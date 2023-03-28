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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithPer express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from multiprocessing.reduction import ForkingPickler
from typing import Dict, List, Optional, Union

import torch

from ..typing import NodeType, EdgeType, TensorDataType
from ..utils import convert_to_tensor, share_memory, squeeze

from .feature import DeviceGroup, Feature
from .graph import CSRTopo, Graph


class Dataset(object):
  r""" A dataset manager for all graph topology and feature data.
  """
  def __init__(
    self,
    graph: Union[Graph, Dict[EdgeType, Graph]] = None,
    node_features: Union[Feature, Dict[NodeType, Feature]] = None,
    edge_features: Union[Feature, Dict[EdgeType, Feature]] = None,
    node_labels: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
  ):
    self.graph = graph
    self.node_features = node_features
    self.edge_features = edge_features
    self.node_labels = squeeze(convert_to_tensor(node_labels))

  def init_graph(
    self,
    edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
    edge_ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
    layout: Union[str, Dict[EdgeType, str]] = 'COO',
    graph_mode: str = 'ZERO_COPY',
    directed: bool = False,
    device: Optional[int] = None
  ):
    r""" Initialize the graph storage and build the object of `Graph`.

    Args:
      edge_index (torch.Tensor or numpy.ndarray): Edge index for graph topo,
        2D CPU tensor/numpy.ndarray(homo). A dict should be provided for
        heterogenous graph. (default: ``None``)
      edge_ids (torch.Tensor or numpy.ndarray): Edge ids for graph edges, A
        CPU tensor (homo) or a Dict[EdgeType, torch.Tensor](hetero).
        (default: ``None``)
      layout (str): The edge layout representation for the input edge index,
        should be 'COO', 'CSR' or 'CSC'. (default: 'COO')
      graph_mode (str): Mode in graphlearn_torch's ``Graph``, 'CPU', 'ZERO_COPY'
        or 'CUDA'. (default: 'ZERO_COPY')
      directed (bool): A Boolean value indicating whether the graph topology is
        directed. (default: ``False``)
      device (torch.device): The target cuda device rank used for graph
        operations when graph mode is not "CPU". (default: ``None``)
    """
    edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
    edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)
    self._directed = directed

    if edge_index is not None:
      if isinstance(edge_index, dict):
        # heterogeneous.
        if edge_ids is not None:
          assert isinstance(edge_ids, dict)
        else:
          edge_ids = {}
        if not isinstance(layout, dict):
          layout = {etype: layout for etype in edge_index.keys()}
        csr_topo_dict = {}
        for etype, e_idx in edge_index.items():
          csr_topo_dict[etype] = CSRTopo(
            edge_index=e_idx,
            edge_ids=edge_ids.get(etype, None),
            layout=layout[etype]
          )
        self.graph = {}
        for etype, csr_topo in csr_topo_dict.items():
          g = Graph(csr_topo, graph_mode, device)
          g.lazy_init()
          self.graph[etype] = g
      else:
        # homogeneous.
        csr_topo = CSRTopo(edge_index, edge_ids, layout)
        self.graph = Graph(csr_topo, graph_mode, device)
        self.graph.lazy_init()

  def init_node_features(
    self,
    node_feature_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
    id2idx: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None,
    sort_func = None,
    split_ratio: Union[float, Dict[NodeType, float]] = 0.0,
    device_group_list: Optional[List[DeviceGroup]] = None,
    device: Optional[int] = None,
    with_gpu: bool = True,
    dtype: Optional[torch.dtype] = None
  ):
    r""" Initialize the node feature storage.

    Args:
      node_feature_data (torch.Tensor or numpy.ndarray): A tensor of the raw
        node feature data, should be a dict for heterogenous graph nodes.
        (default: ``None``)
      id2idx (torch.Tensor or numpy.ndarray): A tensor that maps node id to
        index, should be a dict for heterogenous graph nodes.
        (default: ``None``)
      sort_func: Function for reordering node features. Currently, only features
        of homogeneous nodes are supported to reorder. (default: ``None``)
      split_ratio (float): The proportion (between 0 and 1) of node feature data
        allocated to the GPU, should be a dict for heterogenous graph nodes.
        (default: ``0.0``)
      device_group_list (List[DeviceGroup]): A list of device groups used for
        node feature lookups, the GPU part of feature data will be replicated on
        each device group in this list during the initialization. GPUs with
        peer-to-peer access to each other should be set in the same device
        group properly. (default: ``None``)
      device (torch.device): The target cuda device rank used for node feature
        lookups when the GPU part is not None.. (default: `None`)
      with_gpu (bool): A Boolean value indicating whether the ``Feature`` uses
        ``UnifiedTensor``. If True, it means ``Feature`` consists of
        ``UnifiedTensor``, otherwise ``Feature`` is PyTorch CPU Tensor and
        ``split_ratio``, ``device_group_list`` and ``device`` will be invliad.
        (default: ``True``)
      dtype (torch.dtype): The data type of node feature elements, if not
        specified, it will be automatically inferred. (Default: ``None``).
    """
    if node_feature_data is not None:
      node_feature_data = convert_to_tensor(node_feature_data, dtype)
      id2idx = convert_to_tensor(id2idx)
      if id2idx is None and sort_func is not None:
        if isinstance(node_feature_data, dict):
          logging.warning("'%s': reordering heterogenous graph node features "
                          "is not supported now.", self.__class__.__name__)
        elif self.graph is not None:
          # reorder node features of homogeneous graph.
          assert isinstance(self.graph, Graph)
          if self._directed is None or not self._directed:
            csr_topo_rev = self.graph.csr_topo
          else:
            row, col, eids = self.graph.csr_topo.to_coo()
            csr_topo_rev = CSRTopo((col, row), eids, layout='COO')
          node_feature_data, id2idx = \
            sort_func(node_feature_data, split_ratio, csr_topo_rev)
      self.node_features = _build_features(
        node_feature_data, id2idx, split_ratio,
        device_group_list, device, with_gpu, dtype
      )

  def init_edge_features(
    self,
    edge_feature_data: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
    id2idx: Union[TensorDataType, Dict[EdgeType, TensorDataType]] = None,
    split_ratio: Union[float, Dict[EdgeType, float]] = 0.0,
    device_group_list: Optional[List[DeviceGroup]] = None,
    device: Optional[int] = None,
    with_gpu: bool = True,
    dtype: Optional[torch.dtype] = None
  ):
    r""" Initialize the edge feature storage.

    Args:
      edge_feature_data (torch.Tensor or numpy.ndarray): A tensor of the raw
        edge feature data, should be a dict for heterogenous graph edges.
        (default: ``None``)
      id2idx (torch.Tensor or numpy.ndarray): A tensor that maps edge id to
        index, should be a dict for heterogenous graph edges.
        (default: ``None``)
      split_ratio (float): The proportion (between 0 and 1) of edge feature data
        allocated to the GPU, should be a dict for heterogenous graph edges.
        (default: ``0.0``)
      device_group_list (List[DeviceGroup]): A list of device groups used for
        edge feature lookups, the GPU part of feature data will be replicated on
        each device group in this list during the initialization. GPUs with
        peer-to-peer access to each other should be set in the same device
        group properly. (default: ``None``)
      device (torch.device): The target cuda device rank used for edge feature
        lookups when the GPU part is not None.. (default: `None`)
      with_gpu (bool): A Boolean value indicating whether the ``Feature`` uses
        ``UnifiedTensor``. If True, it means ``Feature`` consists of
        ``UnifiedTensor``, otherwise ``Feature`` is PyTorch CPU Tensor and
        ``split_ratio``, ``device_group_list`` and ``device`` will be invliad.
        (default: ``True``)
      dtype (torch.dtype): The data type of edge feature elements, if not
        specified, it will be automatically inferred. (Default: ``None``).
    """
    if edge_feature_data is not None:
      self.edge_features = _build_features(
        convert_to_tensor(edge_feature_data, dtype), convert_to_tensor(id2idx),
        split_ratio, device_group_list, device, with_gpu, dtype
      )

  def init_node_labels(
    self,
    node_label_data: Union[TensorDataType, Dict[NodeType, TensorDataType]] = None
  ):
    r""" Initialize the node label storage.

    Args:
      node_label_data (torch.Tensor or numpy.ndarray): A tensor of the raw
        node label data, should be a dict for heterogenous graph nodes.
        (default: ``None``)
    """
    if node_label_data is not None:
      self.node_labels = squeeze(convert_to_tensor(node_label_data))

  def share_ipc(self):
    self.node_labels = share_memory(self.node_labels)
    return self.graph, self.node_features, self.edge_features, self.node_labels

  @classmethod
  def from_ipc_handle(cls, ipc_handle):
    graph, node_features, edge_features, node_labels = ipc_handle
    return cls(graph, node_features, edge_features, node_labels)

  def get_graph(self, etype: Optional[EdgeType] = None):
    if isinstance(self.graph, Graph):
      return self.graph
    if isinstance(self.graph, dict):
      assert etype is not None
      return self.graph.get(etype, None)
    return None

  def get_node_types(self):
    if isinstance(self.graph, dict):
      if not hasattr(self, '_node_types'):
        ntypes = set()
        for etype in self.graph.keys():
          ntypes.add(etype[0])
          ntypes.add(etype[2])
        self._node_types = list(ntypes)
      return self._node_types
    return None

  def get_edge_types(self):
    if isinstance(self.graph, dict):
      if not hasattr(self, '_edge_types'):
        self._edge_types = list(self.graph.keys())
      return self._edge_types
    return None

  def get_node_feature(self, ntype: Optional[NodeType] = None):
    if isinstance(self.node_features, Feature):
      return self.node_features
    if isinstance(self.node_features, dict):
      assert ntype is not None
      return self.node_features.get(ntype, None)
    return None

  def get_edge_feature(self, etype: Optional[EdgeType] = None):
    if isinstance(self.edge_features, Feature):
      return self.edge_features
    if isinstance(self.edge_features, dict):
      assert etype is not None
      return self.edge_features.get(etype, None)
    return None

  def get_node_label(self, ntype: Optional[NodeType] = None):
    if isinstance(self.node_labels, torch.Tensor):
      return self.node_labels
    if isinstance(self.node_labels, dict):
      assert ntype is not None
      return self.node_labels.get(ntype, None)
    return None

  def __getitem__(self, key):
    return getattr(self, key, None)

  def __setitem__(self, key, value):
    setattr(self, key, value)


def _build_features(feature_data, id2idx, split_ratio,
                    device_group_list, device, with_gpu, dtype):
  r""" Build `Feature`s for node/edge feature data.
  """
  if feature_data is not None:
    if isinstance(feature_data, dict):
      # heterogeneous.
      if not isinstance(split_ratio, dict):
        split_ratio = {
          graph_type: float(split_ratio)
          for graph_type in feature_data.keys()
        }

      if id2idx is not None:
        assert isinstance(id2idx, dict)
      else:
        id2idx = {}

      features = {}
      for graph_type, feat in feature_data.items():
        features[graph_type] = Feature(
          feat, id2idx.get(graph_type, None),
          split_ratio.get(graph_type, 0.0),
          device_group_list, device, with_gpu,
          dtype if dtype is not None else feat.dtype
        )
    else:
      # homogeneous.
      features = Feature(
        feature_data, id2idx, float(split_ratio),
        device_group_list, device, with_gpu,
        dtype if dtype is not None else feature_data.dtype
      )
  else:
    features = None

  return features


## Pickling Registration

def rebuild_dataset(ipc_handle):
  ds = Dataset.from_ipc_handle(ipc_handle)
  return ds

def reduce_dataset(dataset: Dataset):
  ipc_handle = dataset.share_ipc()
  return (rebuild_dataset, (ipc_handle, ))

ForkingPickler.register(Dataset, reduce_dataset)
