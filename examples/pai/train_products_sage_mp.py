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
import common_io
import os
import time
import torch

import graphlearn_torch as glt
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
    super(SAGE, self).__init__()

    self.num_layers = num_layers

    self.convs = torch.nn.ModuleList()
    self.convs.append(SAGEConv(in_channels, hidden_channels))
    for _ in range(num_layers - 2):
      self.convs.append(SAGEConv(hidden_channels, hidden_channels))
    self.convs.append(SAGEConv(hidden_channels, out_channels))


  def forward(self, x, adjs):
    # `train_loader` computes the k-hop neighborhood of a batch of nodes,
    # and returns, for each layer, a bipartite graph object, holding the
    # bipartite edges `edge_index`, the index `e_id` of the original edges,
    # and the size/shape `size` of the bipartite graph.
    # Target nodes are also included in the source nodes so that one can
    # easily apply skip-connections or add self-loops.
    for i, (edge_index, _, size) in enumerate(adjs):
      x_target = x[:size[1]]  # Target nodes are always placed first.
      x = self.convs[i]((x, x_target), edge_index)
      if i != self.num_layers - 1:
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    return x.log_softmax(dim=-1)


def train(rank, model, loader, train_idx, sampler, feature, label,
          optimizer):
  model.train()
  total_loss = total_correct = 0
  for seeds in loader:
    batch_size, n_id, adjs = sampler.sample_pyg_v1(seeds)
    # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    adjs = [adj.to(rank) for adj in adjs]
    optimizer.zero_grad()
    out = model(feature[n_id], adjs)
    loss = F.nll_loss(out, label[n_id[:batch_size]])
    loss.backward()
    optimizer.step()
    total_loss += float(loss)
    total_correct += int(out.argmax(dim=-1).eq(label[n_id[:batch_size]]).sum())
  loss = total_loss / len(loader)
  approx_acc = total_correct / train_idx.size(0)
  return loss, approx_acc

def run(rank, world_size, graph, feature, label, train_idx,
        lr, nbrs_num, features_num, hidden_dim, class_num, depth, epochs):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
  label = label.to(rank)
  train_loader = torch.utils.data.DataLoader(train_idx,
                                             batch_size=1024,
                                             shuffle=True,
                                             drop_last=True)
  # init graphlearn_torch Sampler.
  glt_sampler = glt.sampler.NeighborSampler(graph,
                                            nbrs_num,
                                            device=torch.device(rank))
  # init model.
  model = SAGE(features_num, hidden_dim, class_num, depth).to(rank)
  model = DistributedDataParallel(model, device_ids=[rank])
  # training.
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  for epoch in range(0, epochs):
    epoch_start = time.time()
    loss, acc = train(rank, model, train_loader, train_idx, glt_sampler,
                      feature, label, optimizer)
    if rank == 0:
      print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}',
            f'Epoch Time: {time.time() - epoch_start}')


def read_train_idx_and_label(table):
  id_label = []
  reader = common_io.table.TableReader(table,
                                       num_threads=10,
                                       capacity=10240)
  while True:
    try:
      data = reader.read(1024, allow_smaller_final_batch=True)
      id_label.extend(data)
    except common_io.exception.OutOfRangeException:
      reader.close()
      break
  ids = torch.tensor([e[0] for e in id_label], dtype=torch.long)
  label = torch.tensor([e[1] for e in id_label], dtype=torch.long)
  return ids.squeeze(), label.squeeze()

if __name__ == "__main__":
  argparser = argparse.ArgumentParser("Train GCN.")
  argparser.add_argument('--tables', type=str, default="" ,
                         help='ODPS input table names')
  argparser.add_argument('--class_num', type=int, default=47)
  argparser.add_argument('--features_num', type=int, default=100)
  argparser.add_argument('--hidden_dim', type=int, default=256)
  argparser.add_argument('--depth', type=int, default=3)
  argparser.add_argument('--nbrs_num', type=list, default=[15, 10, 5])
  argparser.add_argument('--learning_rate', type=float, default=0.003)
  argparser.add_argument('--epoch', type=int, default=10)
  argparser.add_argument('--split_ratio', type=float, default=0.2)
  args = argparser.parse_args()

  world_size = torch.cuda.device_count()
  node_table, edge_table, train_table = args.tables.split(',')
  train_idx, label = read_train_idx_and_label(train_table)
  label.share_memory_()
  train_idx.share_memory_()
  node_tables = {'i': node_table}
  edge_tables = {('i', 'i-i', 'i') : edge_table}
  # init glt Dataset.
  glt_dataset = glt.data.TableDataset(edge_tables=edge_tables,
                                      node_tables=node_tables,
                                      graph_mode='ZERO_COPY',
                                      sort_func=glt.data.sort_by_in_degree,
                                      split_ratio=args.split_ratio,
                                      device_group_list=[glt.data.DeviceGroup(0,[0,1,2,3])],
                                      directed=False,
                                      device=0)
  mp.spawn(run,
           args=(world_size,
                 glt_dataset.graph,
                 glt_dataset.node_features,
                 label,
                 train_idx,
                 args.learning_rate,
                 args.nbrs_num,
                 args.features_num,
                 args.hidden_dim,
                 args.class_num,
                 args.depth,
                 args.epoch),
           nprocs=world_size,
           join=True)