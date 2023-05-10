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
from torch_geometric.nn import GraphSAGE


def run(rank, num_nodes, node_rank, num_training_procs_per_node,
        dataset, train_idx, train_label, batch_size,
        lr, nbrs_num, features_num, hidden_dim, class_num, depth, epochs,
        master_addr, training_pg_master_port, train_loader_master_port):
  # Initialize graphlearn_torch distributed worker group context.
  glt.distributed.init_worker_group(
    world_size=num_nodes*num_training_procs_per_node,
    rank=node_rank*num_training_procs_per_node+rank,
    group_name='distributed-sage-supervised-trainer'
  )

  current_ctx = glt.distributed.get_context()
  current_device = torch.device(rank % torch.cuda.device_count())

  # Initialize training process group of PyTorch.
  torch.distributed.init_process_group(
    backend='nccl',
    rank=current_ctx.rank,
    world_size=current_ctx.world_size,
    init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port)
  )

  train_idx = train_idx.split(train_idx.size(0) // num_training_procs_per_node)[rank]
  train_label = train_label.to(current_device)

  # Create distributed neighbor loader for training
  train_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=nbrs_num,
    input_nodes=train_idx,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collect_features=True,
    to_device=current_device,
    worker_options=glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[current_device],
      worker_concurrency=4,
      master_addr=master_addr,
      master_port=train_loader_master_port,
      channel_size='1GB',
      pin_memory=True
    )
  )


  # Define model and optimizer.
  torch.cuda.set_device(current_device)
  model = GraphSAGE(
    in_channels=features_num,
    hidden_channels=hidden_dim,
    num_layers=depth,
    out_channels=class_num,
  ).to(current_device)
  model = DistributedDataParallel(model, device_ids=[current_device.index])
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
    print(f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n')
    torch.cuda.synchronize()
    torch.distributed.barrier()

def read_train_idx_and_label(table, partition_idx, num_partitions):
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
  argparser.add_argument("--num_training_procs", type=int, default=2,
    help="The number of traning processes per node.")
  argparser.add_argument('--num_vertices', type=int, default=2449029)
  argparser.add_argument('--class_num', type=int, default=47)
  argparser.add_argument('--features_num', type=int, default=100)
  argparser.add_argument('--hidden_dim', type=int, default=256)
  argparser.add_argument('--depth', type=int, default=3)
  argparser.add_argument('--nbrs_num', type=list, default=[15, 10, 5])
  argparser.add_argument('--learning_rate', type=float, default=0.003)
  argparser.add_argument('--epoch', type=int, default=10)
  argparser.add_argument('--batch_size', type=int, default=512)

  args = argparser.parse_args()

  # get dist context from PAI environment.
  num_nodes = int(os.getenv('WORLD_SIZE', 1))
  node_rank = int(os.getenv('RANK', 0))
  master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
  master_port = int(os.getenv('MASTER_PORT', 29500))
  train_loader_master_port = 11234
  partitioner_master_port = 11235

  node_table, edge_table, train_table = args.tables.split(',')
  train_idx, train_label = read_train_idx_and_label(train_table, node_rank, num_nodes)
  train_idx.share_memory_()
  train_label.share_memory_()
  node_tables = {'i': node_table}
  edge_tables = {('i', 'i-i', 'i') : edge_table}
  # init glt Dataset.
  print('--- Loading data partition ...')
  glt_dataset = glt.distributed.DistTableDataset()
  glt_dataset.load(num_partitions=num_nodes,
                   partition_idx=node_rank,
                   edge_tables=edge_tables,
                   node_tables=node_tables,
                   num_nodes=args.num_vertices,
                   graph_mode='ZERO_COPY',
                   device_group_list=None,
                   reader_threads=10,
                   reader_capacity=10240,
                   reader_batch_size=1024,
                   feature_with_gpu=True,
                   edge_assign_strategy='by_src',
                   chunk_size=10000,
                   master_addr=master_addr,
                   master_port=partitioner_master_port,
                   num_rpc_threads=16,
                  )
  print('--- Launching training processes ...')
  mp.spawn(run,
           args=(num_nodes,
                 node_rank,
                 args.num_training_procs,
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
                 master_addr,
                 master_port,
                 train_loader_master_port,
                ),
           nprocs=args.num_training_procs,
           join=True)