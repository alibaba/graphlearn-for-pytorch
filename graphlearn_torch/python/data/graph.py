# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from multiprocessing.reduction import ForkingPickler
from typing import Optional, Tuple, Union, Literal

import torch
import warnings

from .. import py_graphlearn_torch as pywrap
from ..typing import TensorDataType
from ..utils import (
  convert_to_tensor, share_memory, ptr2ind, coo_to_csr, coo_to_csc
)

class Topology(object):
  r""" Graph topology with support for CSC and CSR formats.

  Args:
    edge_index (a 2D torch.Tensor or numpy.ndarray, or a tuple): The edge
      index for graph topology, in the order of first row and then column. 
    edge_ids (torch.Tensor or numpy.ndarray, optional): The edge ids for
      graph edges. If set to ``None``, it will be aranged by the edge size.
      (default: ``None``)
    edge_weights (torch.Tensor or numpy.ndarray, optional): The edge weights for
      graph edges. If set to ``None``, it will be None.
      (default: ``None``)
    input_layout (str): The edge layout representation for the input edge index,
      should be 'COO' (rows and cols uncompressed), 'CSR' (rows compressed)
      or 'CSC' (columns compressed). (default: 'COO')
    layout ('CSR' or 'CSC'): The target edge layout representation for 
      the output. (default: 'CSR')
  """
  def __init__(self,
               edge_index: Union[TensorDataType,
                                 Tuple[TensorDataType, TensorDataType]],
               edge_ids: Optional[TensorDataType] = None,
               edge_weights: Optional[TensorDataType] = None,
               input_layout: str = 'COO',
               layout: Literal['CSR', 'CSC'] = 'CSR'):
    
    edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
    row, col = edge_index[0], edge_index[1]
    input_layout = str(input_layout).upper()
    if input_layout == 'COO':
        assert row.numel() == col.numel()
        num_edges = row.numel()
    elif input_layout == 'CSR':
        num_edges = col.numel()
    elif input_layout == 'CSC':
        num_edges = row.numel()
    else:
      raise RuntimeError(f"'{self.__class__.__name__}': got "
                         f"invalid edge layout {input_layout}")

    edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)
    if edge_ids is None:
      edge_ids = torch.arange(num_edges, dtype=torch.int64, device=row.device)
    else:
      assert edge_ids.numel() == num_edges

    edge_weights = convert_to_tensor(edge_weights, dtype=torch.float)
    if edge_weights is not None:
      assert edge_weights.numel() == num_edges

    self._layout = layout
    
    if input_layout == layout:
      if input_layout == 'CSC':
        self._indices, self._indptr = row, col
      elif input_layout == 'CSR':
        self._indptr, self._indices = row, col
      self._edge_ids = edge_ids
      self._edge_weights = edge_weights
      return
    elif input_layout == 'CSC':
      col = ptr2ind(col)
    elif input_layout == 'CSR':
      row = ptr2ind(row)
    # COO format data is prepared.
    
    if layout == 'CSR':
      self._indptr, self._indices, self._edge_ids, self._edge_weights = \
        coo_to_csr(row, col, edge_id=edge_ids, edge_weight=edge_weights)
    elif layout == 'CSC':
      self._indices, self._indptr, self._edge_ids, self._edge_weights = \
        coo_to_csc(row, col, edge_id=edge_ids, edge_weight=edge_weights)

  
  def to_coo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r""" Convert to COO format.

    Returns:
      row indice tensor, column indice tensor, edge id tensor, edge weight tensor
    """
    if self._layout == 'CSR':
      return ptr2ind(self._indptr), self._indices, \
             self._edge_ids, self._edge_weights
    elif self._layout == 'CSC':
      return self._indices, ptr2ind(self._indptr), \
             self._edge_ids, self._edge_weights

  def to_csc(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r""" Convert to CSC format.

    Returns:
      row indice tensor, column ptr tensor, edge id tensor, edge weight tensor
    """
    if self._layout == 'CSR':
      row, col, edge_id, edge_weights = self.to_coo()
      return coo_to_csc(row, col, edge_id=edge_id, edge_weight=edge_weights)
    elif self._layout == 'CSC':
      return self._indices, self._indptr, self._edge_ids, self._edge_weights
  
  def to_csr(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r""" Convert to CSR format.

    Returns:
      row ptr tensor, column indice tensor, edge id tensor, edge weight tensor
    """
    if self._layout == 'CSR':
      return self._indptr, self._indices, self._edge_ids, self._edge_weights
    elif self._layout == 'CSC':
      row, col, edge_ids, edge_weights = self.to_coo()
      return coo_to_csr(row, col, edge_id=edge_ids, edge_weight=edge_weights)

  @property
  def indptr(self):
    return self._indptr

  @property
  def indices(self):
    return self._indices

  @property
  def edge_ids(self):
    r""" local edge ids.
    """
    return self._edge_ids
  
  @property
  def edge_weights(self):
    r""" local edge weights.
    """
    return self._edge_weights

  @property
  def degrees(self):
    return self._indptr[1:] - self._indptr[:-1]

  @property
  def row_count(self):
    return self._indptr.shape[0] - 1

  @property
  def edge_count(self):
    return self._indices.shape[0]

  def share_memory_(self):
    self._indptr = share_memory(self._indptr)
    self._indices = share_memory(self._indices)
    self._edge_ids = share_memory(self._edge_ids)
    self._edge_weights = share_memory(self._edge_weights)

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)


class Graph(object):
  r""" A graph object used for graph operations such as sampling.

  There are three modes supported:
    1.'CPU': graph data are stored in the CPU memory and graph
      operations are also executed on CPU.
    2.'ZERO_COPY': graph data are stored in the pinned CPU memory and graph
      operations are executed on GPU.
    3.'CUDA': graph data are stored in the GPU memory and graph operations
      are executed on GPU.

  Args:
    topo (Topology): An instance of ``Topology`` with graph topology data.
    mode (str): The graph operation mode, must be 'CPU', 'ZERO_COPY' or 'CUDA'.
      (Default: 'ZERO_COPY').
    device (int, optional): The target cuda device rank to perform graph
      operations. Note that this parameter will be ignored if the graph mode
      set to 'CPU'. The value of ``torch.cuda.current_device()`` will be used
      if set to ``None``. (Default: ``None``).
  """
  def __init__(self, topo: Topology, mode = 'ZERO_COPY',
               device: Optional[int] = None):
    self.topo = topo
    self.topo.share_memory_()
    self.mode = mode.upper()
    self.device = device

    if self.mode != 'CPU' and self.device is not None:
      self.device = int(self.device)
      assert (
        self.device >= 0 and self.device < torch.cuda.device_count()
      ), f"'{self.__class__.__name__}': invalid device rank {self.device}"

    self._graph = None

  def lazy_init(self):
    if self._graph is not None:
      return

    self._graph = pywrap.Graph()
    indptr = self.topo.indptr
    indices = self.topo.indices
    if self.topo.edge_ids is not None:
      edge_ids = self.topo.edge_ids
    else:
      edge_ids = torch.empty(0)

    if self.topo.edge_weights is not None:
      edge_weights = self.topo.edge_weights
    else:
      edge_weights = torch.empty(0)

    if self.mode == 'CPU':
      self._graph.init_cpu_from_csr(indptr, indices, edge_ids, edge_weights)
    else:
      if self.device is None:
        self.device = torch.cuda.current_device()

      if self.mode == 'CUDA':
        self._graph.init_cuda_from_csr(
          indptr, indices, self.device, pywrap.GraphMode.DMA, edge_ids
        )
      elif self.mode == 'ZERO_COPY':
        self._graph.init_cuda_from_csr(
          indptr, indices, self.device, pywrap.GraphMode.ZERO_COPY, edge_ids
        )
      else:
        raise ValueError(f"'{self.__class__.__name__}': "
                         f"invalid mode {self.mode}")

  def export_topology(self):
    return self.topo.indptr, self.topo.indices, self.topo.edge_ids

  def share_ipc(self):
    r""" Create ipc handle for multiprocessing.

    Returns:
      A tuple of topo and graph mode.
    """
    return self.topo, self.mode

  @classmethod
  def from_ipc_handle(cls, ipc_handle):
    r""" Create from ipc handle.
    """
    topo, mode = ipc_handle
    return cls(topo, mode, device=None)

  @property
  def row_count(self):
    self.lazy_init()
    return self._graph.get_row_count()

  @property
  def col_count(self):
    self.lazy_init()
    return self._graph.get_col_count()

  @property
  def edge_count(self):
    self.lazy_init()
    return self._graph.get_edge_count()

  @property
  def graph_handler(self):
    r""" Get a pointer to the underlying graph object for graph operations
    such as sampling.
    """
    self.lazy_init()
    return self._graph


## Pickling Registration

def rebuild_graph(ipc_handle):
  graph = Graph.from_ipc_handle(ipc_handle)
  return graph

def reduce_graph(graph: Graph):
  ipc_handle = graph.share_ipc()
  return (rebuild_graph, (ipc_handle, ))

ForkingPickler.register(Graph, reduce_graph)
