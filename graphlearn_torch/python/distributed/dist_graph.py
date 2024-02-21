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

from typing import Dict, Optional, Union

import torch

from ..data import Graph
from ..typing import (NodeType, EdgeType)

from ..partition import (
  PartitionBook, GLTPartitionBook,
  HeteroNodePartitionDict, HeteroEdgePartitionDict
)

class DistGraph(object):
  r""" Simple wrapper for graph data with distributed context.

  TODO: support graph operations.

  Args:
    num_partitions: Number of data partitions.
    partition_id: Data partition idx of current process.
    local_graph: local `Graph` instance.
    node_pb: Partition book which records vertex ids to worker node ids.
    edge_pb: Partition book which records edge ids to worker node ids.

  Note that`local_graph`, `node_pb` and `edge_pb` should be a dictionary
  for hetero data.
  """
  def __init__(self,
               num_partitions: int,
               partition_idx: int,
               local_graph: Union[Graph, Dict[EdgeType, Graph]],
               node_pb: Union[PartitionBook, HeteroNodePartitionDict],
               edge_pb: Union[PartitionBook, HeteroEdgePartitionDict]=None):
    self.num_partitions = num_partitions
    self.partition_idx = partition_idx
    self.local_graph = local_graph
    if isinstance(self.local_graph, dict):
      self.data_cls = 'hetero'
      for _, graph in self.local_graph.items():
        graph.lazy_init()
    elif isinstance(self.local_graph, Graph):
      self.data_cls = 'homo'
      self.local_graph.lazy_init()
    else:
      raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                       f"graph type '{type(self.local_graph)}'")
    self.node_pb = node_pb
    if self.node_pb is not None:
      if isinstance(self.node_pb, dict):
        assert self.data_cls == 'hetero'
        for key, feat in self.node_pb.items():
          if not isinstance(feat, PartitionBook):
            self.node_pb[key] = GLTPartitionBook(feat)
      elif isinstance(self.node_pb, PartitionBook):
        assert self.data_cls == 'homo'
      elif isinstance(self.node_pb, torch.Tensor):
        assert self.data_cls == 'homo'
        self.node_pb = GLTPartitionBook(self.node_pb)
      else:
        raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                        f"node patition book type '{type(self.node_pb)}'")
    self.edge_pb = edge_pb
    if self.edge_pb is not None:
      if isinstance(self.edge_pb, dict):
        assert self.data_cls == 'hetero'
        for key, feat in self.edge_pb.items():
          if not isinstance(feat, PartitionBook):
            self.edge_pb[key] = GLTPartitionBook(feat)
      elif isinstance(self.edge_pb, PartitionBook):
        assert self.data_cls == 'homo'
      elif isinstance(self.edge_pb, torch.Tensor):
        assert self.data_cls == 'homo'
        self.edge_pb = GLTPartitionBook(self.edge_pb)
      else:
        raise ValueError(f"'{self.__class__.__name__}': found invalid input "
                        f"edge patition book type '{type(self.edge_pb)}'")

  def get_local_graph(self, etype: Optional[EdgeType]=None):
    r""" Get a `Graph` obj of a specific edge type.
    """
    if self.data_cls == 'hetero':
      assert etype is not None
      return self.local_graph[etype]
    return self.local_graph

  def get_node_partitions(self, ids: torch.Tensor,
                          ntype: Optional[NodeType]=None):
    r""" Get the partition ids of node ids with a specific node type.
    """
    if self.data_cls == 'hetero':
      assert ntype is not None
      pb = self.node_pb[ntype]
    else:
      pb = self.node_pb
    return pb[ids.to(pb.device)]

  def get_edge_partitions(self, eids: torch.Tensor,
                          etype: Optional[EdgeType]=None):
    r""" Get the partition ids of edge ids with a specific edge type.
         PS: tehre is no edge pb implementation when loading graph from v6d
    """
    if self.data_cls == 'hetero':
      assert etype is not None
      assert isinstance(self.edge_pb[etype], GLTPartitionBook)
      pb = self.edge_pb[etype]
    else:
      assert isinstance(self.edge_pb[etype], GLTPartitionBook)
      pb = self.edge_pb
    return pb[eids.to(pb.device)]
