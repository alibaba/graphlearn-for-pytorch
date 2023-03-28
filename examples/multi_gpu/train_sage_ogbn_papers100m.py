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

import numpy as np
import os.path as osp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from numpy import genfromtxt
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import GraphSAGE

import graphlearn_torch as glt


def run(rank, world_size, glt_ds, train_idx,
        num_features, num_classes):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(42)
  print(f'Rank {rank} init graphlearn_torch NeighborLoader...')
  train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
  train_loader = glt.loader.NeighborLoader(glt_ds,
                                           [15, 10, 5],
                                           train_idx,
                                           batch_size=1024,
                                           shuffle=True,
                                           device=torch.device(rank))
  print(f'Rank {rank} build graphlearn_torch NeighborLoader Done.')
  model = GraphSAGE(
    in_channels=num_features,
    hidden_channels=256,
    num_layers=3,
    out_channels=num_classes,
  ).to(rank)
  model = DistributedDataParallel(model, device_ids=[rank])
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(1, 10):
    model.train()
    start = time.time()
    total_examples = total_loss = 0
    for batch in train_loader:
      optimizer.zero_grad()
      out = model(batch.x, batch.edge_index)[:batch.batch_size].log_softmax(dim=-1)
      loss = F.nll_loss(out, batch.y[:batch.batch_size])
      loss.backward()
      optimizer.step()
      total_examples += batch.batch_size
      total_loss += float(loss) * batch.batch_size
    end = time.time()
    dist.barrier()
    if rank == 0:
      print(f'Epoch: {epoch:03d}, Loss: {(total_loss / total_examples):.4f},',
            f'Epoch Time: {end - start}')
    dist.barrier()


if __name__ == '__main__':
  world_size = torch.cuda.device_count()
  start = time.time()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))),
                  'data',
                  'papers100M-bin')
  label = np.load(osp.join(root, "raw/node-label.npz"))
  train_idx = genfromtxt(root+'/split/time/train.csv',
                         delimiter='\n')
  train_idx = torch.from_numpy(train_idx.astype(np.long))
  data = np.load(osp.join(root, "raw/data.npz"))
  edge_index = data["edge_index"]
  feature = torch.from_numpy(data["node_feat"]).type(torch.float)
  label = torch.from_numpy(label["node_label"]).type(torch.long).squeeze()
  print(f'Load data cost {time.time()-start} s.')

  start = time.time()
  print('Build graphlearn_torch dataset...')
  glt_dataset = glt.data.Dataset()
  glt_dataset.init_graph(
    edge_index=edge_index,
    graph_mode='ZERO_COPY',
    directed=True
  )
  glt_dataset.init_node_features(
    node_feature_data=feature,
    sort_func=glt.data.sort_by_in_degree,
    split_ratio=0.15 * min(world_size, 4),
    device_group_list=[glt.data.DeviceGroup(0, [0, 1, 2, 3]),
                       glt.data.DeviceGroup(1, [4, 5, 6, 7])],
  )
  glt_dataset.init_node_labels(node_label_data=label)
  print(f'Build graphlearn_torch csr_topo and feature cost {time.time() - start} s.')
  train_idx.share_memory_()
  mp.spawn(run,
           args=(world_size,
                 glt_dataset,
                 train_idx,
                 128,
                 172),
           nprocs=world_size,
           join=True)
