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

import graphlearn_torch as glt
import torch

from dataset import IGBHeteroDataset


def partition_dataset(path: str,
                      num_partitions: int,
                      chunk_size: int,
                      dataset_size: str='tiny',
                      in_memory: bool=True,
                      edge_assign_strategy: str='by_src',
                      use_label_2K: bool=False):
  print(f'-- Loading igbh_{dataset_size} ...')
  data = IGBHeteroDataset(path, dataset_size, in_memory, use_label_2K)
  node_num = {k : v.shape[0] for k, v in data.feat_dict.items()}

  print('-- Saving label ...')
  label_dir = osp.join(path, f'{dataset_size}-label')
  glt.utils.ensure_dir(label_dir)
  torch.save(data.label.squeeze(), osp.join(label_dir, 'label.pt'))

  print('-- Partitioning training idx ...')
  train_idx = data.train_idx
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx_partitions_dir = osp.join(path, f'{dataset_size}-train-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning validation idx ...')
  train_idx = data.val_idx
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx_partitions_dir = osp.join(path, f'{dataset_size}-val-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning test idx ...')
  test_idx = data.test_idx
  test_idx = test_idx.split(test_idx.size(0) // num_partitions)
  test_idx_partitions_dir = osp.join(path, f'{dataset_size}-test-partitions')
  glt.utils.ensure_dir(test_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(test_idx[pidx], osp.join(test_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning graph and features ...')
  partitions_dir = osp.join(path, f'{dataset_size}-partitions')
  partitioner = glt.partition.RandomPartitioner(
    output_dir=partitions_dir,
    num_parts=num_partitions,
    num_nodes=node_num,
    edge_index=data.edge_dict,
    node_feat=data.feat_dict,
    edge_assign_strategy=edge_assign_strategy,
    chunk_size=chunk_size,
  )
  partitioner.partition()


if __name__ == '__main__':
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser = argparse.ArgumentParser(description="Arguments for partitioning ogbn datasets.")
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset_size', type=str, default='tiny',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--num_classes', type=int, default=19,
      choices=[19, 2983], help='number of classes')
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  parser.add_argument("--num_partitions", type=int, default=2,
      help="Number of partitions")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")
  parser.add_argument("--edge_assign_strategy", type=str, default='by_src',
      help="edge assign strategy can be either 'by_src' or 'by_dst'")

  args = parser.parse_args()

  partition_dataset(
    args.path,
    num_partitions=args.num_partitions,
    chunk_size=args.chunk_size,
    dataset_size=args.dataset_size,
    in_memory=args.in_memory,
    edge_assign_strategy=args.edge_assign_strategy,
    use_label_2K=args.num_classes==2983,
  )
