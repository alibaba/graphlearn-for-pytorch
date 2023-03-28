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

import os
import time
import torch

import graphlearn_torch as glt
import os.path as osp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch_geometric.transforms as T

from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import Linear, HGTConv


class HGT(torch.nn.Module):
  def __init__(self, hidden_channels, out_channels, num_heads, num_layers,
               node_types, edge_types):
    super().__init__()

    self.lin_dict = torch.nn.ModuleDict()
    for node_type in node_types:
      self.lin_dict[node_type] = Linear(-1, hidden_channels)

    self.convs = torch.nn.ModuleList()
    for _ in range(num_layers):
      conv = HGTConv(hidden_channels, hidden_channels, (node_types, edge_types),
                     num_heads, group='sum')
      self.convs.append(conv)
    self.lin = Linear(hidden_channels, out_channels)

  def forward(self, x_dict, edge_index_dict):
    x_dict = {
      node_type: self.lin_dict[node_type](x).relu_()
      for node_type, x in x_dict.items()
    }
    for conv in self.convs:
      x_dict = conv(x_dict, edge_index_dict)
    return self.lin(x_dict['paper'])


@torch.no_grad()
def init_params(loader, model):
  # Initialize lazy parameters via forwarding a single batch to the model:
  batch = next(iter(loader))
  model(batch.x_dict, batch.edge_index_dict)


def train(model, loader, optimizer):
  model.train()
  total_examples = total_loss = 0
  for batch in loader:
    optimizer.zero_grad()
    batch_size = batch['paper'].batch_size
    out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
    loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
    loss.backward()
    optimizer.step()
    total_examples += batch_size
    total_loss += float(loss) * batch_size
  return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
  model.eval()
  total_examples = total_correct = 0
  for batch in loader:
    batch_size = batch['paper'].batch_size
    out = model(batch.x_dict, batch.edge_index_dict)[:batch_size]
    pred = out.argmax(dim=-1)
    total_examples += batch_size
    total_correct += int((pred == batch['paper'].y[:batch_size]).sum())
  return total_correct / total_examples


def run(rank, world_size, glt_ds, train_idx,
        val_idx, num_classes, node_types, edge_types):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  # init NeighborLoader
  train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
  train_loader = glt.loader.NeighborLoader(glt_ds,
                                           [10] * 2,
                                           ('paper', train_idx),
                                           batch_size=1024,
                                           shuffle=True,
                                           device=torch.device(rank))
  val_loader = glt.loader.NeighborLoader(glt_ds,
                                         [10] * 2,
                                         ('paper', val_idx),
                                         batch_size=1024,
                                         shuffle=True,
                                         device=torch.device(rank))
  # model
  model = HGT(hidden_channels=64,
              out_channels=num_classes,
              num_heads=2,
              num_layers=2,
              node_types=node_types,
              edge_types=edge_types).to(rank)
  init_params(train_loader, model)
  model = DistributedDataParallel(model, device_ids=[rank],
                                  find_unused_parameters=True)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  for epoch in range(1, 21):
    start_time = time.time()
    loss = train(model, train_loader, optimizer)
    epoch_time = time.time() - start_time
    dist.barrier()
    if rank == 0:
      val_acc = test(model, val_loader)
      print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}',
            f'Time: {epoch_time:.4f}, Val: {val_acc:.4f}')
    dist.barrier()

if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/')
  transform = T.ToUndirected(merge=True)
  dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)
  data = dataset[0]
  train_idx = data['paper'].train_mask.nonzero(as_tuple=False).view(-1)
  val_idx = data['paper'].val_mask.nonzero(as_tuple=False).view(-1)

  # init graphlearn_torch Dataset.
  edge_dict, x_dict = {}, {}
  for etype in data.edge_types:
    edge_dict[etype] = data[etype]['edge_index']
  for ntype in data.node_types:
    x_dict[ntype] = data[ntype].x.clone(memory_format=torch.contiguous_format)

  glt_dataset = glt.data.Dataset()
  glt_dataset.init_graph(
    edge_index=edge_dict,
    graph_mode='ZERO_COPY'
  )
  glt_dataset.init_node_features(
    node_feature_data=x_dict,
    split_ratio=0.2,
    device_group_list=[glt.data.DeviceGroup(i, [i]) for i in range(world_size)]
  )
  glt_dataset.init_node_labels(node_label_data={'paper': data['paper'].y})

  train_idx.share_memory_()
  val_idx.share_memory_()

  mp.spawn(run,
           args=(world_size,
                 glt_dataset,
                 train_idx,
                 val_idx,
                 dataset.num_classes,
                 data.node_types,
                 data.edge_types),
           nprocs=world_size,
           join=True)