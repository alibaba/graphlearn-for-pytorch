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

from dataset import IGBHeteroDataset


def partition_feature(src_path: str,
                      dst_path: str,
                      partition_idx: int,
                      chunk_size: int,
                      dataset_size: str='tiny',
                      in_memory: bool=True):
  print(f'-- Loading igbh_{dataset_size} ...')
  data = IGBHeteroDataset(src_path, dataset_size, in_memory, with_edges=False)

  print(f'-- Build feature for partition {partition_idx} ...')
  dst_path = osp.join(dst_path, f'{dataset_size}-partitions')
  glt.partition.base.build_partition_feature(dst_path, partition_idx, chunk_size, data.feat_dict)


if __name__ == '__main__':
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser = argparse.ArgumentParser(description="Arguments for partitioning ogbn datasets.")
  parser.add_argument('--src_path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dst_path', type=str, default=root,
      help='path containing the partitioned datasets')
  parser.add_argument('--dataset_size', type=str, default='tiny',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  parser.add_argument("--partition_idx", type=int, default=0,
      help="Index of a partition")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")


  args = parser.parse_args()

  partition_feature(
    args.src_path,
    args.dst_path,
    partition_idx=args.partition_idx,
    chunk_size=args.chunk_size,
    dataset_size=args.dataset_size,
    in_memory=args.in_memory==1
  )
