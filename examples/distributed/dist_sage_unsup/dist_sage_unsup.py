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
import time
import tqdm

import graphlearn_torch as glt
import torch
import torch.distributed
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, recall_score

from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.algorithms.join import Join
from torch_geometric.nn import GraphSAGE


@torch.no_grad()
def test(model, test_loader, current_device):
  model.eval()
  preds, targets = [], []
  for batch in tqdm.tqdm(test_loader):
    batch = batch.to(current_device)
    out = model.module(batch.x, batch.edge_index)
    out_src = out[batch.edge_label_index[0]]
    out_dst = out[batch.edge_label_index[1]]
    pred = (out_src * out_dst).sum(dim=-1).sigmoid().round().view(-1).cpu()
    target = batch.edge_label.long().cpu()
    
    preds.append(pred)
    targets.append(target)

  pred = torch.cat(preds, dim=0).numpy()
  target = torch.cat(targets, dim=0).numpy()

  return roc_auc_score(target, pred), recall_score(target, pred)


def run_training_proc(local_proc_rank: int, num_nodes: int, node_rank: int,
                      num_training_procs_per_node: int, dataset_name: str,
                      in_channels: int, out_channels: int, with_weight: bool,
                      dataset: glt.distributed.DistDataset,
                      train_idx: torch.Tensor, test_idx: torch.Tensor,
                      epochs: int, batch_size: int, master_addr: str,
                      training_pg_master_port: int, train_loader_master_port: int,
                      test_loader_master_port: int):
  # Initialize graphlearn_torch distributed worker group context.
  glt.distributed.init_worker_group(
    world_size=num_nodes*num_training_procs_per_node,
    rank=node_rank*num_training_procs_per_node+local_proc_rank,
    group_name='distributed-sage-unsupervised-trainer'
  )

  current_ctx = glt.distributed.get_context()
  current_device = torch.device('cpu')

  # Initialize training process group of PyTorch.
  torch.distributed.init_process_group(
    backend='gloo',
    rank=current_ctx.rank,
    world_size=current_ctx.world_size,
    init_method='tcp://{}:{}'.format(master_addr, training_pg_master_port)
  )

  # Create distributed neighbor loader for training
  # train_idx is an edge index of shape [2, num_train_edges] in this example.
  train_idx = (train_idx.T.split(train_idx.size(1) // num_training_procs_per_node)
               [local_proc_rank]).T
  
  # with_weight=True only supports CPU sampling currently.
  train_loader = glt.distributed.DistLinkNeighborLoader(
    data=dataset,
    num_neighbors=[15, 10],
    edge_label_index=train_idx,
    batch_size=batch_size,
    with_weight=with_weight, 
    neg_sampling='binary',
    shuffle=True,
    collect_features=True,
    to_device=current_device,
    worker_options=glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[current_device],
      worker_concurrency=4,
      master_addr=master_addr,
      master_port=train_loader_master_port,
      channel_size='8GB',
      pin_memory=False
    )
  )

  # Create distributed neighbor loader for testing.
  # test_idx is an edge index of shape [2, num_test_edges] in this example.
  test_idx = (test_idx.T.split(test_idx.size(1) // num_training_procs_per_node)
              [local_proc_rank]).T

  # with_weight=True only supports CPU sampling currently.
  test_loader = glt.distributed.DistLinkNeighborLoader(
    data=dataset,
    num_neighbors=[15, 10],
    edge_label_index=test_idx,
    neg_sampling='binary',
    batch_size=batch_size,
    with_weight=with_weight,
    shuffle=False,
    collect_features=True,
    to_device=current_device,
    worker_options=glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=1,
      worker_devices=[current_device],
      worker_concurrency=4,
      master_addr=master_addr,
      master_port=test_loader_master_port,
      channel_size='2GB',
      pin_memory=False
    )
  )

  # Define model and optimizer.
  current_device = torch.device(local_proc_rank % torch.cuda.device_count())
  torch.cuda.set_device(current_device)
  model = GraphSAGE(
    in_channels=in_channels,
    hidden_channels=256,
    num_layers=2,
    out_channels=out_channels,
  ).to(current_device)
  model = DistributedDataParallel(model, device_ids=[current_device.index],
                                  find_unused_parameters=True)
  optimizer = ZeRO(model.parameters(), torch.optim.Adam, lr=1e-5)

  # Train and test.
  for epoch in range(0, epochs):
    with Join([model, optimizer]):
      model.train()
      start = time.time()
      for batch in tqdm.tqdm(train_loader):
        batch = batch.to(current_device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        out_src = out[batch.edge_label_index[0]]
        out_dst = out[batch.edge_label_index[1]]
        pred = (out_src * out_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
      end = time.time()
      print(f'-- [Trainer {current_ctx.rank}] Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}\n')
      # torch.cuda.empty_cache() # empty cache when GPU memory is not efficient.
      torch.cuda.synchronize()
      torch.distributed.barrier()
    # Test accuracy.
    if epoch % 2 == 0:
      test_auc, test_recall = test(model, test_loader, current_device)
      print(f'-- [Trainer {current_ctx.rank}] Test AUC: {test_auc:.4f} Test Rec: {test_recall:.4f}\n')
      torch.cuda.synchronize()
      torch.distributed.barrier()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Arguments for distributed training of unsupervised SAGE."
  )
  parser.add_argument(
    "--dataset",
    type=str,
    default='Your dataset name',
    help="The name of the dataset.",
  )
  parser.add_argument(
    "--in_channel",
    type=int,
    default=4,
  )
  parser.add_argument(
    "--out_channel",
    type=int,
    default=32,
  )
  parser.add_argument(
    "--dataset_root_dir",
    type=str,
    default='Your dataset root directory',
    help="The root directory (relative path) of the partitioned dataset.",
  )
  parser.add_argument(
    "--num_dataset_partitions",
    type=int,
    default=2,
    help="The number of partitions of the dataset.",
  )
  parser.add_argument(
    "--num_nodes",
    type=int,
    default=2,
    help="Number of distributed nodes.",
  )
  parser.add_argument(
    "--node_rank",
    type=int,
    default=0,
    help="The current node rank.",
  )
  parser.add_argument(
    "--num_training_procs",
    type=int,
    default=2,
    help="The number of traning processes per node.",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=11,
    help="The number of training epochs.",
  )
  parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="Batch size for the training and testing dataloader.",
  )
  parser.add_argument(
    "--with_weight",
    action="store_true",
    help="Whether to use edge weights.",
  )
  parser.add_argument(
    "--master_addr",
    type=str,
    default='localhost',
    help="The master address for RPC initialization.",
  )
  parser.add_argument(
    "--training_pg_master_port",
    type=int,
    default=12211,
    help="The port used for PyTorch's process group initialization across training processes.",
  )
  parser.add_argument(
    "--train_loader_master_port",
    type=int,
    default=12212,
    help="The port used for RPC initialization across all sampling workers of training loader.",
  )
  parser.add_argument(
    "--test_loader_master_port",
    type=int,
    default=12213,
    help="The port used for RPC initialization across all sampling workers of testing loader.",
  )
  args = parser.parse_args()

  print('--- Distributed training example of unsupervised SAGE ---\n')
  print(f'* dataset: {args.dataset}\n')
  print(f'* dataset root dir: {args.dataset_root_dir}\n')
  print(f'* number of dataset partitions: {args.num_dataset_partitions}\n')
  print(f'* total nodes: {args.num_nodes}\n')
  print(f'* node rank: {args.node_rank}\n')
  print(f'* number of training processes per node: {args.num_training_procs}\n')
  print(f'* epochs: {args.epochs}\n')
  print(f'* batch size: {args.batch_size}\n')
  print(f'* master addr: {args.master_addr}\n')
  print(f'* training process group master port: {args.training_pg_master_port}\n')
  print(f'* training loader master port: {args.train_loader_master_port}\n')
  print(f'* testing loader master port: {args.test_loader_master_port}\n')

  print('--- Loading data partition ...\n')
  root_dir = osp.join(osp.dirname(osp.realpath(__file__)), args.dataset_root_dir)
  data_pidx = args.node_rank % args.num_dataset_partitions
  dataset = glt.distributed.DistDataset()
  dataset.load(
    root_dir=osp.join(root_dir, f'{args.dataset}-partitions'),
    partition_idx=data_pidx,
    graph_mode='CPU',
  )

  # Load train and test edges.
  train_idx = torch.load(
    osp.join(root_dir, f'{args.dataset}-train-partitions', f'partition{data_pidx}.pt')
  )
  
  test_idx = torch.load(
    osp.join(root_dir, f'{args.dataset}-test-partitions', f'partition{data_pidx}.pt')
  )
  train_idx.share_memory_()
  test_idx.share_memory_()

  print('--- Launching training processes ...\n')
  torch.multiprocessing.spawn(
    run_training_proc,
    args=(args.num_nodes, args.node_rank, args.num_training_procs,
          args.dataset, args.in_channel, args.out_channel, args.with_weight,
          dataset, train_idx, test_idx, args.epochs,
          args.batch_size, args.master_addr, args.training_pg_master_port,
          args.train_loader_master_port, args.test_loader_master_port),
    nprocs=args.num_training_procs,
    join=True
  )
