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
import torch.distributed


def run_server_proc(
  num_servers: int, num_clients: int, server_rank: int,
  dataset: glt.distributed.DistDataset, master_addr: str,
  server_client_port: int
):
  print(f'-- [Server {server_rank}] Initializing server ...')
  glt.distributed.init_server(
    num_servers=num_servers,
    num_clients=num_clients,
    server_rank=server_rank,
    dataset=dataset,
    master_addr=master_addr,
    master_port=server_client_port,
    num_rpc_threads=16,
    server_group_name='dist-train-supervised-sage-server'
  )

  print(f'-- [Server {server_rank}] Waiting for exit ...')
  glt.distributed.wait_and_shutdown_server()

  print(f'-- [Server {server_rank}] Exited ...')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Arguments for distributed training of supervised SAGE with servers."
  )
  parser.add_argument(
    "--dataset",
    type=str,
    default='ogbn-products',
    help="The name of ogbn dataset.",
  )
  parser.add_argument(
    "--dataset_root_dir",
    type=str,
    default='../../data/products',
    help="The root directory (relative path) of partitioned ogbn dataset.",
  )
  parser.add_argument(
    "--num_dataset_partitions",
    type=int,
    default=2,
    help="The number of partitions of the dataset.",
  )
  parser.add_argument(
    "--num_server_nodes",
    type=int,
    default=2,
    help="Number of server nodes for remote sampling.",
  )
  parser.add_argument(
    "--num_client_nodes",
    type=int,
    default=2,
    help="Number of client nodes for training.",
  )
  parser.add_argument(
    "--node_rank",
    type=int,
    default=0,
    help="The node rank of the current role.",
  )
  parser.add_argument(
    "--num_server_procs_per_node",
    type=int,
    default=1,
    help="The number of server processes for remote sampling per server node.",
  )
  parser.add_argument(
    "--num_client_procs_per_node",
    type=int,
    default=2,
    help="The number of client processes for training per client node.",
  )
  parser.add_argument(
    "--master_addr",
    type=str,
    default='localhost',
    help="The master address for RPC initialization.",
  )
  parser.add_argument(
    "--server_client_master_port",
    type=int,
    default=11110,
    help="The port used for RPC initialization across all servers and clients.",
  )
  args = parser.parse_args()

  print(
    f'--- Distributed training example of supervised SAGE with server-client mode. Server {args.node_rank} ---'
  )
  print(f'* dataset: {args.dataset}')
  print(f'* dataset root dir: {args.dataset_root_dir}')
  print(f'* total server nodes: {args.num_server_nodes}')
  print(f'* node rank: {args.node_rank}')
  print(
    f'* number of server processes per server node: {args.num_server_procs_per_node}'
  )
  print(
    f'* number of client processes per client node: {args.num_client_procs_per_node}'
  )
  print(f'* master addr: {args.master_addr}')
  print(f'* server-client master port: {args.server_client_master_port}')

  print(f'* number of dataset partitions: {args.num_dataset_partitions}')

  num_servers = args.num_server_nodes * args.num_server_procs_per_node
  num_clients = args.num_client_nodes * args.num_client_procs_per_node
  root_dir = osp.join(
    osp.dirname(osp.realpath(__file__)), args.dataset_root_dir
  )
  data_pidx = args.node_rank % args.num_dataset_partitions

  mp_context = torch.multiprocessing.get_context('spawn')

  print('--- Loading data partition ...')
  dataset = glt.distributed.DistDataset()
  dataset.load(
    root_dir=osp.join(root_dir, f'{args.dataset}-partitions'),
    partition_idx=data_pidx,
    graph_mode='ZERO_COPY',
    whole_node_label_file=osp.join(
      root_dir, f'{args.dataset}-label', 'label.pt'
    )
  )

  print('--- Launching server processes ...')
  server_procs = []
  for local_proc_rank in range(args.num_server_procs_per_node):
    server_rank = args.node_rank * args.num_server_procs_per_node + local_proc_rank
    sproc = mp_context.Process(
      target=run_server_proc,
      args=(
        num_servers, num_clients, server_rank, dataset, args.master_addr,
        args.server_client_master_port
      )
    )
    server_procs.append(sproc)
  for sproc in server_procs:
    sproc.start()
  for sproc in server_procs:
    sproc.join()
