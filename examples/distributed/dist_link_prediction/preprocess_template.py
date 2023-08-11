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

import argparse
import os.path as osp
from typing import Tuple, Optional

import torch
import graphlearn_torch as glt


# Implement this class to preprocess your dataset
class FakeDataset(object):
  def __init__(self, feat_src_path, graph_src_path):
    self.feats = self.get_feat(feat_src_path)
    self.graph = self.get_graph(graph_src_path)
    self.edge_weights = self.get_edge_weight()
    self.train_idx, self.val_idx, self.test_idx = \
        self.get_train_test_split_idx()

  def get_feat(self, feat_src_path) -> torch.Tensor:
    pass

  def get_graph(self, graph_src_path) -> torch.Tensor:
    pass

  def get_edge_weight(self, edge_weight_src_path='') -> Optional[torch.Tensor]:
    pass

  def get_train_test_split_idx(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r''' This function should return three tensors, which are the indices or
     edges of training, validation and testing for the link prediction task.
     If returned tensors are of shape [2, num_edges], then the first row is
     the source nodes and the second row is the destination nodes.
    '''
    pass

  @property
  def node_num(self) -> int:
    pass


def partition_dataset(
  dataset_name: str, feat_src_path: str, graph_src_path: str,
  dst_path: str = '', num_partitions: int = 2
):
  # Substitute FakeDataset with your own dataset class
  data = FakeDataset(feat_src_path, graph_src_path)
  print('-- Partitioning training idx / training edges ...')
  train_idx = data.train_idx
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx = [idx.T for idx in train_idx]
  train_idx_partitions_dir = osp.join(dst_path, f'{dataset_name}-train-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
      torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning validation idx / validation edges...')
  train_idx = data.val_idx
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx = [idx.T for idx in train_idx]
  train_idx_partitions_dir = osp.join(dst_path, f'{dataset_name}-val-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
      torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning test idx / testing edges ...')
  test_idx = data.test_idx
  test_idx = test_idx.split(test_idx.size(0) // num_partitions)
  test_idx = [idx.T for idx in test_idx]
  test_idx_partitions_dir = osp.join(dst_path, f'{dataset_name}-test-partitions')
  glt.utils.ensure_dir(test_idx_partitions_dir)
  for pidx in range(num_partitions):
      torch.save(test_idx[pidx], osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning graph and features ...')
  partitions_dir = osp.join(dst_path, f'{dataset_name}-partitions')
  partitioner = glt.partition.RandomPartitioner(
      output_dir=partitions_dir,
      num_parts=num_partitions,
      num_nodes=data.node_num,
      edge_index=data.graph,
      node_feat=data.feats,
      edge_weights=data.edge_weights,
      edge_assign_strategy='by_src'
  )
  partitioner.partition()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Arguments for partition of unsupervised SAGE."
  )
  parser.add_argument(
    '--dataset_name', type=str, default=''
  )
  parser.add_argument(
    '--feat_src_path', type=str, default=''
  )
  parser.add_argument(
    '--graph_src_path', type=str, default=''
  )
  parser.add_argument(
    '--dst_path', type=str, default=''
  )
  parser.add_argument(
    '--num_partitions', type=int, default=2
  )
  args = parser.parse_args()
  dataset_name = args.dataset_name
  feat_src_path = args.feat_src_path
  graph_src_path = args.graph_src_path
  dst_path = args.dst_path
  num_partitions = args.num_partitions
     
  partition_dataset(
     dataset_name, feat_src_path, graph_src_path, dst_path, num_partitions)
