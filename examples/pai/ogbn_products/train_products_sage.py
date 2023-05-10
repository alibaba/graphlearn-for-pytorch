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
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import GraphSAGE



def run(rank, world_size, dataset, train_idx, train_label, batch_size,
        lr, nbrs_num, features_num, hidden_dim, class_num, depth, epochs,):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)
  torch.manual_seed(42)

  train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
  train_label = train_label.to(rank)

  # Create neighbor loader for training
  train_loader = glt.loader.NeighborLoader(dataset,
                                           nbrs_num,
                                           train_idx,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           device=torch.device(rank))
  # Define model and optimizer.
  model = GraphSAGE(
    in_channels=features_num,
    hidden_channels=hidden_dim,
    num_layers=depth,
    out_channels=class_num,
  ).to(rank)
  model = DistributedDataParallel(model, device_ids=[rank])
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Train and test.
  for epoch in range(0, epochs):
    model.train()
    start = time.time()
    for batch in train_loader:
      optimizer.zero_grad()
      out = model(batch.x, batch.edge_index)[:batch.batch_size].log_softmax(dim=-1)
      loss = F.nll_loss(out, train_label[batch.batch])
      loss.backward()
      optimizer.step()
    end = time.time()
    print(f'-- [Trainer {rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n')
    torch.cuda.synchronize()
    torch.distributed.barrier()


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
  argparser = argparse.ArgumentParser("Train GraphSAGE.")
  argparser.add_argument('--tables', type=str, default="" ,
                         help='ODPS input table names')
  argparser.add_argument('--class_num', type=int, default=47)
  argparser.add_argument('--features_num', type=int, default=100)
  argparser.add_argument('--hidden_dim', type=int, default=256)
  argparser.add_argument('--depth', type=int, default=3)
  argparser.add_argument('--nbrs_num', type=list, default=[15, 10, 5])
  argparser.add_argument('--learning_rate', type=float, default=0.003)
  argparser.add_argument('--epoch', type=int, default=10)
  argparser.add_argument('--batch_size', type=int, default=512)
  argparser.add_argument('--split_ratio', type=float, default=0.2)
  args = argparser.parse_args()

  world_size = torch.cuda.device_count()
  node_table, edge_table, train_table = args.tables.split(',')
  train_idx, train_label = read_train_idx_and_label(train_table)
  train_idx.share_memory_()
  train_label.share_memory_()
  node_tables = {'i': node_table}
  edge_tables = {('i', 'i-i', 'i') : edge_table}
  # init glt Dataset.
  glt_dataset = glt.data.TableDataset()
  glt_dataset.load(edge_tables=edge_tables,
                   node_tables=node_tables,
                   graph_mode='ZERO_COPY',
                   sort_func=glt.data.sort_by_in_degree,
                   split_ratio=args.split_ratio,
                   directed=False,
                   device=0)
  mp.spawn(run,
           args=(world_size,
                 glt_dataset,
                 train_idx,
                 train_label,
                 args.batch_size,
                 args.learning_rate,
                 args.nbrs_num,
                 args.features_num,
                 args.hidden_dim,
                 args.class_num,
                 args.depth,
                 args.epoch,
                ),
           nprocs=world_size,
           join=True)