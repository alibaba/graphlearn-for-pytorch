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

from multiprocessing.reduction import ForkingPickler
from typing import Dict, List, Optional, Union, Literal, Tuple, Callable

import torch

from ..data import Dataset, Graph, Feature, DeviceGroup, vineyard_utils
from ..partition import (
  load_partition, cat_feature_cache,
  PartitionBook, HeteroNodePartitionDict, HeteroEdgePartitionDict
)
from ..typing import (NodeType, EdgeType, NodeLabel, NodeIndex)
from ..utils import share_memory, default_id_filter, default_id_select


class DistDataset(Dataset):
  r""" Graph and feature dataset with distributed partition info.

  """
  def __init__(
    self,
    num_partitions: int = 1,
    partition_idx: int = 0,
    graph_partition: Union[Graph, Dict[EdgeType, Graph]] = None,
    node_feature_partition: Union[Feature, Dict[NodeType, Feature]] = None,
    edge_feature_partition: Union[Feature, Dict[EdgeType, Feature]] = None,
    whole_node_labels: NodeLabel = None,
    node_pb: Union[PartitionBook, HeteroNodePartitionDict] = None,
    edge_pb: Union[PartitionBook, HeteroEdgePartitionDict] = None,
    node_feat_pb: Union[PartitionBook, HeteroNodePartitionDict] = None,
    edge_feat_pb: Union[PartitionBook, HeteroEdgePartitionDict] = None,
    edge_dir: Literal['in', 'out'] = 'out',
    graph_caching: bool = False,
    node_split: Tuple[NodeIndex, NodeIndex, NodeIndex] = None,
    id_filter: Callable = default_id_filter,
    id_select: Callable = default_id_select
  ):
    super().__init__(
      graph_partition,
      node_feature_partition,
      edge_feature_partition,
      whole_node_labels,
      edge_dir,
      node_split,
    )
    self.id_filter = id_filter
    self.id_select = id_select
    self.num_partitions = num_partitions
    self.partition_idx = partition_idx
    self.graph_caching = graph_caching

    self.node_pb = node_pb
    self.edge_pb = edge_pb

    # As the loaded feature partition may be concatenated with its cached
    # features and the partition book for features will be modified, thus we
    # need to distinguish them with the original graph partition books.
    #
    # If the `node_feat_pb` or `edge_feat_pb` is not provided, the `node_pb`
    # or `edge_pb` will be used instead for feature lookups.
    self._node_feat_pb = node_feat_pb
    self._edge_feat_pb = edge_feat_pb

    if self.graph is not None:
      assert self.node_pb is not None
    if self.node_features is not None:
      assert self.node_pb is not None or self._node_feat_pb is not None
    if self.edge_features is not None:
      assert self.edge_pb is not None or self._edge_feat_pb is not None

  def load(
    self,
    root_dir: str,
    partition_idx: int,
    graph_mode: str = 'ZERO_COPY',
    input_layout: Literal['COO', 'CSR', 'CSC'] = 'COO',
    feature_with_gpu: bool = True,
    graph_caching: bool = False,
    device_group_list: Optional[List[DeviceGroup]] = None,
    whole_node_label_file: Union[str, Dict[NodeType, str]] = None,
    device: Optional[int] = None
  ):
    r""" Load a certain dataset partition from partitioned files and create
    in-memory objects (``Graph``, ``Feature`` or ``torch.Tensor``).

    Args:
      root_dir (str): The directory path to load the graph and feature
        partition data.
      partition_idx (int): Partition idx to load.
      graph_mode (str): Mode for creating graphlearn_torch's ``Graph``, including
        ``CPU``, ``ZERO_COPY`` or ``CUDA``. (default: ``ZERO_COPY``)
      input_layout (str): layout of the input graph, including ``CSR``, ``CSC`` 
        or ``COO``. (default: ``COO``)
      feature_with_gpu (bool): A Boolean value indicating whether the created
        ``Feature`` objects of node/edge features use ``UnifiedTensor``.
        If True, it means ``Feature`` consists of ``UnifiedTensor``, otherwise
        ``Feature`` is a PyTorch CPU Tensor, the ``device_group_list`` and
        ``device`` will be invliad. (default: ``True``)
      graph_caching (bool): A Boolean value indicating whether to load the full
        graph totoploy instead of partitioned one.
      device_group_list (List[DeviceGroup], optional): A list of device groups
        used for feature lookups, the GPU part of feature data will be
        replicated on each device group in this list during the initialization.
        GPUs with peer-to-peer access to each other should be set in the same
        device group properly.  (default: ``None``)
      whole_node_label_file (str): The path to the whole node labels which are
        not partitioned. (default: ``None``)
      device: The target cuda device rank used for graph operations when graph
        mode is not "CPU" and feature lookups when the GPU part is not None.
        (default: ``None``)
    """
    (
      self.num_partitions,
      self.partition_idx,
      graph_data,
      node_feat_data,
      edge_feat_data,
      self.node_pb,
      self.edge_pb
    ) = load_partition(root_dir, partition_idx, graph_caching)

    # init graph partition
    if isinstance(graph_data, dict):
      # heterogeneous.
      edge_index, edge_ids, edge_weights = {}, {}, {}
      for k, v in graph_data.items():
        edge_index[k] = v.edge_index
        edge_ids[k] = v.eids
        edge_weights[k] = v.weights
    else:
      # homogeneous.
      edge_index = graph_data.edge_index
      edge_ids = graph_data.eids
      edge_weights = graph_data.weights
    self.init_graph(edge_index, edge_ids, edge_weights, layout=input_layout,
                    graph_mode=graph_mode, device=device)
    self.graph_caching = graph_caching

    # load node feature partition
    if node_feat_data is not None:
      node_cache_ratio, node_feat, node_feat_id2idx, node_feat_pb = \
        _cat_feature_cache(partition_idx, node_feat_data, self.node_pb)
      self.init_node_features(
        node_feat, node_feat_id2idx, None, node_cache_ratio,
        device_group_list, device, feature_with_gpu, dtype=None
      )
      self._node_feat_pb = node_feat_pb

    # load edge feature partition
    if edge_feat_data is not None:
      edge_cache_ratio, edge_feat, edge_feat_id2idx, edge_feat_pb = \
        _cat_feature_cache(partition_idx, edge_feat_data, self.edge_pb)
      self.init_edge_features(
        edge_feat, edge_feat_id2idx, edge_cache_ratio,
        device_group_list, device, feature_with_gpu, dtype=None
      )
      self._edge_feat_pb = edge_feat_pb

    # load whole node labels
    if whole_node_label_file is not None:
      if isinstance(whole_node_label_file, dict):
        whole_node_labels = {}
        for ntype, file in whole_node_label_file.items():
          whole_node_labels[ntype] = torch.load(file)
      else:
        whole_node_labels = torch.load(whole_node_label_file)
      self.init_node_labels(whole_node_labels)

  def random_node_split(
    self,
    num_val: Union[float, int],
    num_test: Union[float, int],
  ):
    r"""Performs a node-level random split by adding :obj:`train_idx`,
    :obj:`val_idx` and :obj:`test_idx` attributes to the
    :class:`~graphlearn_torch.distributed.DistDataset` object. All nodes except 
    those in the validation and test sets will be used for training.
    
    Args:
      num_val (int or float): The number of validation samples.
        If float, it represents the ratio of samples to include in the
        validation set.
      num_test (int or float): The number of test samples in case
        of :obj:`"train_rest"` and :obj:`"random"` split. If float, it
        represents the ratio of samples to include in the test set.
    """

    if isinstance(self.node_labels, dict):
      train_idx = {}
      val_idx = {}
      test_idx = {}
  
      for node_type, _ in self.node_labels.items():
        indices = self.id_filter(self.node_pb[node_type], self.partition_idx)
        train_idx[node_type], val_idx[node_type], test_idx[node_type] = random_split(indices, num_val, num_test)
    else:
      indices = self.id_filter(self.node_pb, self.partition_idx)
      train_idx, val_idx, test_idx = random_split(indices, num_val, num_test)
    self.init_node_split((train_idx, val_idx, test_idx))

  def load_vineyard(
    self,
    vineyard_id: str,
    vineyard_socket: str,
    edges: List[EdgeType],
    edge_weights: Dict[EdgeType, str] = None,
    node_features: Dict[NodeType, List[str]] = None,
    edge_features: Dict[EdgeType, List[str]] = None,
    node_labels: Dict[NodeType, str] = None,
  ):
    super().load_vineyard(vineyard_id=vineyard_id, vineyard_socket=vineyard_socket, 
                          edges=edges, edge_weights=edge_weights, node_features=node_features, 
                          edge_features=edge_features, node_labels=node_labels)
    if isinstance(self.graph, dict):
      # hetero
      self._node_feat_pb = {}
      if node_features:
        for ntype, _ in self.node_features.items():
          if self.node_pb is not None:
            self._node_feat_pb[ntype] = self.node_pb[ntype]
          else:
            self._node_feat_pb[ntype] = None
    else:
      # homo
      if node_features:
        self._node_feat_pb = self.node_pb
    
    self.id_select = vineyard_utils.v6d_id_select
    self.id_filter = vineyard_utils.v6d_id_filter

  def share_ipc(self):
    super().share_ipc()
    if isinstance(self.node_pb, torch.Tensor):
      self.node_pb = share_memory(self.node_pb)
      self.edge_pb = share_memory(self.edge_pb)
      self._node_feat_pb = share_memory(self._node_feat_pb)
      self._edge_feat_pb = share_memory(self._edge_feat_pb)
    ipc_hanlde = (
      self.num_partitions, self.partition_idx,
      self.graph, self.node_features, self.edge_features, self.node_labels,
      self.node_pb, self.edge_pb, self._node_feat_pb, self._edge_feat_pb, 
      self.edge_dir, self.graph_caching,
      (self.train_idx, self.val_idx, self.test_idx)
    )
    return ipc_hanlde

  @classmethod
  def from_ipc_handle(cls, ipc_handle):
    return cls(*ipc_handle)

  @property
  def node_feat_pb(self):
    if self._node_feat_pb is None:
      return self.node_pb
    return self._node_feat_pb

  @property
  def edge_feat_pb(self):
    if self._edge_feat_pb is None:
      return self.edge_pb
    return self._edge_feat_pb


def _cat_feature_cache(partition_idx, raw_feat_data, raw_feat_pb):
  r""" Cat a feature partition with its cached features.
  """
  if isinstance(raw_feat_data, dict):
    # heterogeneous.
    cache_ratio, feat_data, feat_id2idx, feat_pb = {}, {}, {}, {}
    for graph_type, raw_feat in raw_feat_data.items():
      cache_ratio[graph_type], feat_data[graph_type], \
      feat_id2idx[graph_type], feat_pb[graph_type] = \
        cat_feature_cache(partition_idx, raw_feat, raw_feat_pb[graph_type])
  else:
    # homogeneous.
    cache_ratio, feat_data, feat_id2idx, feat_pb = \
      cat_feature_cache(partition_idx, raw_feat_data, raw_feat_pb)
  return cache_ratio, feat_data, feat_id2idx, feat_pb


## Pickling Registration

def rebuild_dist_dataset(ipc_handle):
  ds = DistDataset.from_ipc_handle(ipc_handle)
  return ds

def reduce_dist_dataset(dataset: DistDataset):
  ipc_handle = dataset.share_ipc()
  return (rebuild_dist_dataset, (ipc_handle, ))

ForkingPickler.register(DistDataset, reduce_dist_dataset)

def random_split(
  indices: torch.Tensor,
  num_val: Union[float, int],
  num_test: Union[float, int],
):  
  num_total = indices.shape[0]
  num_val = round(num_total * num_val) if isinstance(num_val, float) else num_val
  num_test = round(num_total * num_test) if isinstance(num_test, float) else num_test
  perm = torch.randperm(num_total)
  val_idx = indices[perm[:num_val]].clone()
  test_idx = indices[perm[num_val:num_val + num_test]].clone()
  train_idx = indices[perm[num_val + num_test:]].clone()
  return train_idx, val_idx, test_idx