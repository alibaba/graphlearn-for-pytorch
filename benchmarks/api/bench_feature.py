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
import time
import torch

import graphlearn_torch as glt
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler


def test_glt_ogbnproducts(split_ratio):
  root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                  '..', 'data', 'products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  train_idx = dataset.get_idx_split()["train"]
  train_loader = torch.utils.data.DataLoader(train_idx,
                                             batch_size=1024,
                                             pin_memory=True,
                                             shuffle=True)
  csr_topo = glt.data.Topology(dataset[0].edge_index)

  g = glt.data.Graph(csr_topo, 'CUDA', device=0)
  device = torch.device('cuda:0')
  sampler = glt.sampler.NeighborSampler(g, [15, 10, 5], device=device)

  cpu_tensor, id2index = glt.data.sort_by_in_degree(
      dataset[0].x, split_ratio, csr_topo)
  feature = glt.data.Feature(cpu_tensor,
                             id2index,
                             split_ratio,
                             device_group_list=[glt.data.DeviceGroup(0, [0])],
                             device=0)
  total_num = 0
  total_time = 0
  for seeds in train_loader:
    nid = sampler.sample_from_nodes(seeds).node
    torch.cuda.synchronize()
    start = time.time()
    res = feature[nid]
    torch.cuda.synchronize()
    total_time += time.time() - start
    total_num += res.numel()
  torch.cuda.synchronize()
  print('Lookup {} ids, takes {} secs, Throughput {} GB/s.'\
    .format(total_num, total_time, total_num * 4 / total_time/ (1024**3)))


def test_quiver_ogbnproducts(split_ratio):
  import quiver
  cache_size = str(950 * split_ratio) + 'M'
  root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                  '..', 'data', 'products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  train_idx = dataset.get_idx_split()["train"]
  train_loader = torch.utils.data.DataLoader(train_idx,
                                             batch_size=1024,
                                             pin_memory=True,
                                             shuffle=True)
  csr_topo = quiver.CSRTopo(dataset[0].edge_index)
  quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, [15, 10, 5],
                                               device=0,
                                               mode="GPU")
  quiver_feature = quiver.Feature(rank=0,
                                  device_list=[0],
                                  device_cache_size=cache_size,
                                  cache_policy="device_replicate",
                                  csr_topo=csr_topo)
  quiver_feature.from_cpu_tensor(dataset[0].x)
  total_num = 0
  total_time = 0
  for seeds in train_loader:
    nid, _, _ = quiver_sampler.sample(seeds)
    torch.cuda.synchronize()
    start = time.time()
    res = quiver_feature[nid]
    torch.cuda.synchronize()
    total_time += time.time() - start
    total_num += res.numel()
  torch.cuda.synchronize()
  print('Lookup {} ids, takes {} secs, Throughput {} GB/s.'\
    .format(total_num, total_time, total_num * 4 / total_time/ (1024**3)))


def test_pyg_ogbnproducts():
  root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                  '..', 'data', 'products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  feature = dataset[0].x
  train_idx = dataset.get_idx_split()["train"]
  train_loader = NeighborSampler(dataset[0].edge_index,
                                 node_idx=train_idx,
                                 sizes=[15, 10, 5],
                                 batch_size=1024,
                                 shuffle=True)
  total_num = 0
  total_time = 0
  for _, n_id, _ in train_loader:
    start = time.time()
    res = feature[n_id]
    # torch.cuda.synchronize()
    total_time += time.time() - start
    total_num += res.numel()
  # torch.cuda.synchronize()
  print('Lookup {} ids, takes {} secs, Throughput {} GB/s.'\
    .format(total_num, total_time, total_num * 4 / total_time/ (1024**3)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Test Feature Lookup benchmarks.")
  parser.add_argument('--backend', type=str, default='glt',
                      help='glt, quiver, or pyg')
  parser.add_argument('--split_ratio', type=float, default=0.2)
  args = parser.parse_args()
  if args.backend == 'glt':
    test_glt_ogbnproducts(args.split_ratio)
  elif args.backend == 'quiver':
    test_quiver_ogbnproducts(args.split_ratio)
  else:
    test_pyg_ogbnproducts()