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

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import numpy as np
from enum import Enum

# Types for basic graph entity #################################################

# Node-types are denoted by a single string
NodeType = str

# Edge-types are denotes by a triplet of strings.
EdgeType = Tuple[str, str, str]

EDGE_TYPE_STR_SPLIT = '__'

def as_str(type: Union[NodeType, EdgeType]) -> str:
  if isinstance(type, NodeType):
    return type
  elif isinstance(type, (list, tuple)) and len(type) == 3:
    return EDGE_TYPE_STR_SPLIT.join(type)
  return ''

def reverse_edge_type(etype: EdgeType):
  src, edge, dst = etype
  if not src == dst:
    if edge.split("_", 1)[0] == 'rev': # undirected edge with `rev_` prefix.
      edge = edge.split("_", 1)[1]
    else:
      edge = 'rev_' + edge
  return (dst, edge, src)


# A representation of tensor data
TensorDataType = Union[torch.Tensor, np.ndarray]

NodeLabel = Union[TensorDataType, Dict[NodeType, TensorDataType]]
NodeIndex = Union[TensorDataType, Dict[NodeType, TensorDataType]]

class Split(Enum):
  train = 'train'
  valid = 'valid'
  test = 'test'

# Types for partition data #####################################################

class GraphPartitionData(NamedTuple):
  r""" Data and indexing info of a graph partition.
  """
  # edge index (rows, cols)
  edge_index: Tuple[torch.Tensor, torch.Tensor]
  # edge ids tensor corresponding to `edge_index`
  eids: torch.Tensor
  # weights tensor corresponding to `edge_index`
  weights: Optional[torch.Tensor] = None

class FeaturePartitionData(NamedTuple):
  r""" Data and indexing info of a node/edge feature partition.
  """
  # node/edge feature tensor
  feats: Optional[torch.Tensor]
  # node/edge ids tensor corresponding to `feats`
  ids: Optional[torch.Tensor]
  # feature cache tensor
  cache_feats: Optional[torch.Tensor]
  # cached node/edge ids tensor corresponding to `cache_feats`
  cache_ids: Optional[torch.Tensor]

HeteroGraphPartitionData = Dict[EdgeType, GraphPartitionData]
HeteroFeaturePartitionData = Dict[Union[NodeType, EdgeType], FeaturePartitionData]

# Types for partition book #####################################################

PartitionBook = torch.Tensor
HeteroNodePartitionDict = Dict[NodeType, PartitionBook]
HeteroEdgePartitionDict = Dict[EdgeType, PartitionBook]

# Types for neighbor sampling ##################################################

Seeds = Union[torch.Tensor, str] 
InputNodes = Union[Seeds, NodeType, Tuple[NodeType, Seeds], Tuple[NodeType, List[Seeds]]]
EdgeIndexTensor = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
InputEdges = Union[EdgeIndexTensor, EdgeType, Tuple[EdgeType, EdgeIndexTensor]]
NumNeighbors = Union[List[int], Dict[EdgeType, List[int]]]
