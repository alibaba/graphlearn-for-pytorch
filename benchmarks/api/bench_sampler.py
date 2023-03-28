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


def test_glt_ogbnproducts(mode='GPU'):
  if mode == 'GPU':
    graph_mode = 'CUDA'
  else:
    graph_mode = 'ZERO_COPY'
  root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                  '..', 'data', 'products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  train_idx = dataset.get_idx_split()["train"]
  train_loader = torch.utils.data.DataLoader(train_idx,
                                             batch_size=1024,
                                             pin_memory=True,
                                             shuffle=True)
  csr_topo = glt.data.CSRTopo(dataset[0].edge_index)
  g = glt.data.Graph(csr_topo, graph_mode, device=0)
  device = torch.device(0)
  sampler = glt.sampler.NeighborSampler(g, [15, 10, 5], device=device)
  total_time = 0
  sampled_edges = 0
  for seeds in train_loader:
    seeds = seeds.to(0)
    torch.cuda.synchronize()
    start = time.time()
    row = sampler.sample_from_nodes(seeds).row
    torch.cuda.synchronize()
    total_time += time.time() - start
    sampled_edges += row.shape[0]
  print('Sampled Edges per secs: {} M'.format(sampled_edges / total_time / 1000000))

def test_quiver_ogbnproducts(mode='GPU'):
  import quiver
  if mode == 'GPU':
    quiver_mode = 'GPU'
  else:
    quiver_mode = 'UVA'
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
                                               mode=quiver_mode)
  total_time = 0
  sampled_edges = 0
  for seeds in train_loader:
    seeds = seeds.to(0)
    torch.cuda.synchronize()
    start = time.time()
    _, _, adjs = quiver_sampler.sample(seeds)
    torch.cuda.synchronize()
    total_time += time.time() - start
    for adj in adjs:
      sampled_edges += adj.edge_index.shape[1]
  print('Sampled Edges per secs: {} M'.format(sampled_edges / total_time / 1000000))


def test_pyg_ogbnproducts():
  root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))),
                  '..', 'data', 'products')
  dataset = PygNodePropPredDataset('ogbn-products', root)
  train_idx = dataset.get_idx_split()["train"]
  train_loader = NeighborSampler(dataset[0].edge_index,
                                 node_idx=train_idx,
                                 sizes=[15, 10, 5],
                                 batch_size=1024,
                                 shuffle=True)
  total_time = 0
  sampled_edges = 0
  start = time.time()
  for _, _, adjs in train_loader:
    total_time += time.time() - start
    for adj in adjs:
      sampled_edges += adj.edge_index.shape[1]
    start = time.time()
  print('Sampled Edges per secs: {} M'.format(sampled_edges / total_time / 1000000))


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Test Sampler benchmarks.")
  parser.add_argument('--backend', type=str, default='glt',
                      help='glt, quiver, or pyg')
  parser.add_argument('--sample_mode', type=str, default='GPU',
                      help='GPU or ZERO_COPY')
  args = parser.parse_args()
  if args.backend == 'glt':
    test_glt_ogbnproducts(args.sample_mode)
  elif args.backend == 'quiver':
    test_quiver_ogbnproducts(args.sample_mode)
  else:
    test_pyg_ogbnproducts()