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

from abc import ABC, abstractmethod
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Literal

import torch

from ..typing import NodeType, EdgeType, NumNeighbors, Split
from ..utils import CastMixin


class EdgeIndex(NamedTuple):
  r""" PyG's :class:`~torch_geometric.loader.EdgeIndex` used in old data loader
  :class:`~torch_geometric.loader.NeighborSampler`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/neighbor_sampler.py
  """
  edge_index: torch.Tensor
  e_id: Optional[torch.Tensor]
  size: Tuple[int, int]

  def to(self, *args, **kwargs):
    edge_index = self.edge_index.to(*args, **kwargs)
    e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
    return EdgeIndex(edge_index, e_id, self.size)


@dataclass
class NodeSamplerInput(CastMixin):
  r""" The sampling input of
  :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_nodes`.

  This class corresponds to :class:`~torch_geometric.sampler.NodeSamplerInput`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  Args:
    node (torch.Tensor): The indices of seed nodes to start sampling from.
    input_type (str, optional): The input node type (in case of sampling in
      a heterogeneous graph). (default: :obj:`None`).
  """
  node: torch.Tensor
  input_type: Optional[NodeType] = None

  def __getitem__(self, index: Union[torch.Tensor, Any]) -> 'NodeSamplerInput':
    if not isinstance(index, torch.Tensor):
      index = torch.tensor(index, dtype=torch.long)
    index = index.to(self.node.device)
    return NodeSamplerInput(self.node[index], self.input_type)

  def __len__(self):
    return self.node.numel()

  def share_memory(self):
    self.node.share_memory_()
    return self

  def to(self, device: torch.device):
    self.node.to(device)
    return self

class NegativeSamplingMode(Enum):
    # 'binary': Randomly sample negative edges in the graph.
    binary = 'binary'
    # 'triplet': Randomly sample negative destination nodes for each positive
    # source node.
    triplet = 'triplet'


@dataclass
class NegativeSampling(CastMixin):
    r"""The negative sampling configuration of a
    :class:`~torch_geometric.sampler.BaseSampler` when calling
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

    Args:
        mode (str): The negative sampling mode
            (:obj:`"binary"` or :obj:`"triplet"`).
            If set to :obj:`"binary"`, will randomly sample negative links
            from the graph.
            If set to :obj:`"triplet"`, will randomly sample negative
            destination nodes for each positive source node.
        amount (int or float, optional): The ratio of sampled negative edges to
            the number of positive edges. (default: :obj:`1`)
        weight (torch.Tensor, optional): A node-level vector determining the
            sampling of nodes. Does not necessariyl need to sum up to one.
            If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
    """
    mode: NegativeSamplingMode
    amount: Union[int, float] = 1
    weight: Optional[torch.Tensor] = None

    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        amount: Union[int, float] = 1,
        weight: Optional[torch.Tensor] = None,
    ):
        self.mode = NegativeSamplingMode(mode)
        self.amount = amount
        self.weight = weight

        if self.amount <= 0:
            raise ValueError(f"The attribute 'amount' needs to be positive "
                             f"for '{self.__class__.__name__}' "
                             f"(got {self.amount})")

        if self.is_triplet():
            if self.amount != math.ceil(self.amount):
                raise ValueError(f"The attribute 'amount' needs to be an "
                                 f"integer for '{self.__class__.__name__}' "
                                 f"with 'triplet' negative sampling "
                                 f"(got {self.amount}).")
            self.amount = math.ceil(self.amount)

    def is_binary(self) -> bool:
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def share_memory(self):
      if self.weight is not None:
        self.weight.share_memory_()
      return self

    def to(self, device: torch.device):
      if self.weight is not None:
        self.weight.to(device)
      return self


@dataclass
class EdgeSamplerInput(CastMixin):
  r""" The sampling input of
  :meth:`~graphlearn_torch.sampler.BaseSampler.sample_from_edges`.

  This class corresponds to :class:`~torch_geometric.sampler.EdgeSamplerInput`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  Args:
    row (torch.Tensor): The source node indices of seed links to start
      sampling from.
    col (torch.Tensor): The destination node indices of seed links to start
      sampling from.
    label (torch.Tensor, optional): The label for the seed links.
      (default: :obj:`None`).
    input_type (Tuple[str, str, str], optional): The input edge type (in
      case of sampling in a heterogeneous graph). (default: :obj:`None`).
  """
  row: torch.Tensor
  col: torch.Tensor
  label: Optional[torch.Tensor] = None
  input_type: Optional[EdgeType] = None
  neg_sampling: Optional[NegativeSampling] = None

  def __getitem__(self, index: Union[torch.Tensor, Any]) -> 'EdgeSamplerInput':
    if not isinstance(index, torch.Tensor):
      index = torch.tensor(index, dtype=torch.long)
    index = index.to(self.row.device)
    return EdgeSamplerInput(
      self.row[index],
      self.col[index],
      self.label[index] if self.label is not None else None,
      self.input_type,
      self.neg_sampling
    )

  def __len__(self):
    return self.row.numel()

  def share_memory(self):
    self.row.share_memory_()
    self.col.share_memory_()
    if self.label is not None:
      self.label.share_memory_()
    if self.label is not None:
      self.neg_sampling.share_memory()
    return self

  def to(self, device: torch.device):
    self.row.to(device)
    self.col.to(device)
    if self.label is not None:
      self.label.to(device)
    if self.label is not None:
      self.neg_sampling.to(device)
    return self


@dataclass
class SamplerOutput(CastMixin):
  r""" The sampling output of a :class:`~graphlearn_torch.sampler.BaseSampler` on
  homogeneous graphs.

  This class corresponds to :class:`~torch_geometric.sampler.SamplerOutput`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  Args:
    node (torch.Tensor): The sampled nodes in the original graph.
    row (torch.Tensor): The source node indices of the sampled subgraph.
      Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
      corresponding to the nodes in the :obj:`node` tensor.
    col (torch.Tensor): The destination node indices of the sampled subgraph.
      Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
      corresponding to the nodes in the :obj:`node` tensor.
    edge (torch.Tensor, optional): The sampled edges in the original graph.
      This tensor is used to obtain edge features from the original
      graph. If no edge attributes are present, it may be omitted.
    batch (torch.Tensor, optional): The vector to identify the seed node
      for each sampled node. Can be present in case of disjoint subgraph
      sampling per seed node. (default: :obj:`None`).
    device (torch.device, optional): The device that all data of this output
      resides in. (default: :obj:`None`).
    metadata: (Any, optional): Additional metadata information.
      (default: :obj:`None`).
  """
  node: torch.Tensor
  row: torch.Tensor
  col: torch.Tensor
  edge: Optional[torch.Tensor] = None
  batch: Optional[torch.Tensor] = None
  num_sampled_nodes: Optional[Union[List[int], torch.Tensor]] = None
  num_sampled_edges: Optional[Union[List[int], torch.Tensor]] = None
  device: Optional[torch.device] = None
  metadata: Optional[Any] = None


@dataclass
class HeteroSamplerOutput(CastMixin):
  r""" The sampling output of a :class:`~graphlearn_torch.sampler.BaseSampler` on
  heterogeneous graphs.

  This class corresponds to
  :class:`~torch_geometric.sampler.HeteroSamplerOutput`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  Args:
    node (Dict[str, torch.Tensor]): The sampled nodes in the original graph
      for each node type.
    row (Dict[Tuple[str, str, str], torch.Tensor]): The source node indices
      of the sampled subgraph for each edge type. Indices must be re-indexed
      to :obj:`{ 0, ..., num_nodes - 1 }` corresponding to the nodes in the
      :obj:`node` tensor of the source node type.
    col (Dict[Tuple[str, str, str], torch.Tensor]): The destination node
      indices of the sampled subgraph for each edge type. Indices must be
      re-indexed to :obj:`{ 0, ..., num_nodes - 1 }` corresponding to the nodes
      in the :obj:`node` tensor of the destination node type.
    edge (Dict[Tuple[str, str, str], torch.Tensor], optional): The sampled
      edges in the original graph for each edge type. This tensor is used to
      obtain edge features from the original graph. If no edge attributes are
      present, it may be omitted. (default: :obj:`None`).
    batch (Dict[str, torch.Tensor], optional): The vector to identify the
      seed node for each sampled node for each node type. Can be present
      in case of disjoint subgraph sampling per seed node.
      (default: :obj:`None`).
    edge_types: (List[Tuple[str, str, str]], optional): The list of edge types
      of the sampled subgraph. (default: :obj:`None`).
    input_type: (Union[NodeType, EdgeType], optional): The input type of seed
      nodes or edge_label_index.
      (default: :obj:`None`).
    device (torch.device, optional): The device that all data of this output
      resides in. (default: :obj:`None`).
    metadata: (Any, optional): Additional metadata information.
      (default: :obj:`None`)
  """
  node: Dict[NodeType, torch.Tensor]
  row: Dict[EdgeType, torch.Tensor]
  col: Dict[EdgeType, torch.Tensor]
  edge: Optional[Dict[EdgeType, torch.Tensor]] = None
  batch: Optional[Dict[NodeType, torch.Tensor]] = None
  num_sampled_nodes: Optional[Dict[NodeType, Union[List[int], torch.Tensor]]] = None
  num_sampled_edges: Optional[Dict[EdgeType, Union[List[int], torch.Tensor]]] = None
  edge_types: Optional[List[EdgeType]] = None
  input_type: Optional[Union[NodeType, EdgeType]] = None
  device: Optional[torch.device] = None
  metadata: Optional[Any] = None

  def get_edge_index(self):
    edge_index = {k: torch.stack([v, self.col[k]]) for k, v in self.row.items()}
    if self.edge_types is not None:
      for etype in self.edge_types:
        if edge_index.get(etype, None) is None:
          edge_index[etype] = \
            torch.empty((2, 0), dtype=torch.long).to(self.device)
    return edge_index


@dataclass
class NeighborOutput(CastMixin):
  r""" The output of sampled neighbor results for a single hop sampling.

  Args:
    nbr (torch.Tensor): A 1D tensor of all sampled neighborhood node ids.
    nbr_num (torch.Tensor): A 1D tensor that identify the number of
      neighborhood nodes for each source nodes. Must be the same length as
      the source nodes of this sampling hop.
    nbr_num (torch.Tensor, optional): The edge ids corresponding to the sampled
      edges (from source node to the sampled neighborhood node). Should be the
      same length as :obj:`nbr` if provided.
  """
  nbr: torch.Tensor
  nbr_num: torch.Tensor
  edge: Optional[torch.Tensor]

  def to(self, device: torch.device):
    return NeighborOutput(
      nbr=self.nbr.to(device),
      nbr_num=self.nbr_num.to(device),
      edge=(self.edge.to(device) if self.edge is not None else None)
    )


class SamplingType(Enum):
  r""" Enum class for sampling types.
  """
  NODE = 0
  LINK = 1
  SUBGRAPH = 2
  RANDOM_WALK = 3


@dataclass
class SamplingConfig:
  r""" Configuration info for sampling.
  """
  sampling_type: SamplingType
  num_neighbors: Optional[NumNeighbors]
  batch_size: int
  shuffle: bool
  drop_last: bool
  with_edge: bool
  collect_features: bool
  with_neg: bool
  with_weight: bool
  edge_dir: Literal['in', 'out']


class BaseSampler(ABC):
  r""" A base class that initializes a graph sampler and provides
  :meth:`sample_from_nodes` and :meth:`sample_from_edges` routines.

  This class corresponds to :class:`~torch_geometric.sampler.BaseSampler`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/sampler/base.py

  """
  @abstractmethod
  def sample_from_nodes(
    self,
    inputs: NodeSamplerInput,
    **kwargs
  ) -> Union[HeteroSamplerOutput, SamplerOutput]:
    r""" Performs sampling from the nodes specified in :obj:`inputs`,
    returning a sampled subgraph(egograph) in the specified output format.

    Args:
      inputs (torch.Tensor): The input data with node indices to start
        sampling from.
    """

  @abstractmethod
  def sample_from_edges(
    self,
    inputs: EdgeSamplerInput,
    **kwargs,
  ) -> Union[HeteroSamplerOutput, SamplerOutput]:
    r""" Performs sampling from the edges specified in :obj:`inputs`,
    returning a sampled subgraph(egograph) in the specified output format.

    Args:
      inputs (EdgeSamplerInput): The input data for sampling from edges
        including the (1) source node indices, the (2) destination node
        indices, the (3) optional edge labels and the (4) input edge type.
    """

  @abstractmethod
  def subgraph(
    self,
    inputs: NodeSamplerInput,
  ) -> SamplerOutput:
    r""" Induce an enclosing subgraph based on inputs and their neighbors(if
      num_neighbors is not None).

    Args:
      inputs (torch.Tensor): The input data with node indices to induce subgraph
        from.
    Returns:
      The sampled unique nodes, relabeled rows and cols, original edge_ids,
      and a mapping from indices in `inputs` to new indices in output nodes,
      i.e. nodes[mapping] = inputs.
    """

class RemoteSamplerInput(ABC):
  """A base class that provides the `to_local_sampler_input` method for the server
  to obtain the sampler input.
  """
  
  @abstractmethod
  def to_local_sampler_input(
    self,
    dataset,
    **kwargs
  ) -> Union[NodeSamplerInput, EdgeSamplerInput]:
    r"""
    Abstract method to convert the sampler input to local format.
    """


class RemoteNodePathSamplerInput(RemoteSamplerInput):
  r"""RemoteNodePathSamplerInput passes the node path to the server, where the server
  can load node seeds from it.
  """
  def __init__(self, node_path: str, input_type: str ) -> None:
    self.node_path = node_path
    self.input_type = input_type

  def to_local_sampler_input(
    self,
    dataset,
    **kwargs,
  ) -> NodeSamplerInput:
    node = torch.load(self.node_path)
    return NodeSamplerInput(node=node, input_type=self.input_type)

class RemoteNodeSplitSamplerInput(RemoteSamplerInput):
  r"""RemoteNodeSplitSamplerInput passes the split category to the server and the server 
  loads seeds from the dataset.
  """
  def __init__(self, split: Split, input_type: str ) -> None:
    self.split = split
    self.input_type = input_type

  def to_local_sampler_input(
    self,
    dataset,
    **kwargs,
  ) -> NodeSamplerInput:
    if self.split == Split.train:
      idx = dataset.train_idx
    elif self.split == Split.valid:
      idx = dataset.val_idx
    elif self.split == Split.test:
      idx = dataset.test_idx 
    if isinstance(idx, dict):
      idx = idx[self.input_type]
    return NodeSamplerInput(node=idx, input_type=self.input_type)
