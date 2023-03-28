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
import torch
import torch.distributed as dist

import graphlearn_torch as glt


if __name__ == "__main__":
  print('*** DistNeighborLoader Benchmarks ***')

  parser = argparse.ArgumentParser('DistRandomSampler benchmarks.')
  parser.add_argument('--dataset', type=str, default='products',
                      help='name of the dataset for benchmark')
  parser.add_argument('--num_nodes', type=int, default=2,
                      help='number of worker nodes')
  parser.add_argument('--node_rank', type=int, default=0,
                      help='worker node rank')
  parser.add_argument('--sample_nprocs', type=int, default=2,
                      help='number of processes for sampling')
  parser.add_argument('--epochs', type=int, default=1,
                      help='repeat epochs for sampling')
  parser.add_argument('--batch_size', type=int, default=2048,
                      help='batch size for sampling')
  parser.add_argument('--shuffle', action="store_true",
                      help='whether to shuffle input seeds at each epoch')
  parser.add_argument('--with_edge', action="store_true",
                      help='whether to sample with edge ids')
  parser.add_argument('--collect_features', action='store_true',
                      help='whether to collect features for sampled results')
  parser.add_argument('--worker_concurrency', type=int, default=4,
                      help='concurrency for each sampling worker')
  parser.add_argument('--channel_size', type=str, default='4GB',
                      help='memory used for shared-memory channel')
  parser.add_argument('--master_addr', type=str, default='localhost',
                      help='master ip address for synchronization across all training nodes')
  parser.add_argument('--master_port', type=str, default='11234',
                      help='port for synchronization across all training nodes')
  args = parser.parse_args()
  
  dataset_name = args.dataset
  num_nodes = args.num_nodes
  node_rank = args.node_rank
  sampling_nprocs = args.sample_nprocs
  device_count = torch.cuda.device_count()
  epochs = args.epochs
  batch_size = args.batch_size
  shuffle = args.shuffle
  with_edge = args.with_edge
  collect_features = args.collect_features
  worker_concurrency = args.worker_concurrency
  channel_size = args.channel_size
  master_addr = str(args.master_addr)
  sampling_master_port = int(args.master_port)
  torch_pg_master_port = sampling_master_port + 1
  
  print('- dataset: {}'.format(dataset_name))
  print('- total nodes: {}'.format(num_nodes))
  print('- node rank: {}'.format(node_rank))
  print('- device count: {}'.format(device_count))
  print('- sampling nprocs per training proc: {}'.format(sampling_nprocs))
  print('- epochs: {}'.format(epochs))
  print('- batch size: {}'.format(batch_size))
  print('- shuffle: {}'.format(shuffle))
  print('- sample with edge id: {}'.format(with_edge))
  print('- collect remote features: {}'.format(collect_features))
  print('- sampling concurrency per worker: {}'.format(worker_concurrency))
  print('- channel size: {}'.format(channel_size))
  print('- master addr: {}'.format(master_addr))
  print('- sampling master port: {}'.format(sampling_master_port))

  print('** Loading dist dataset ...')
  root = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', dataset_name)
  dataset = glt.distributed.DistDataset()
  dataset.load(
    root_dir=osp.join(root, 'ogbn-'+dataset_name+'-partitions'),
    partition_idx=node_rank,
    graph_mode='ZERO_COPY',
    device_group_list=[glt.data.DeviceGroup(0, [0]), glt.data.DeviceGroup(1, [1])], # 2 GPUs
    device=0
  )

  print('** Loading input seeds  ...')
  seeds_dir = osp.join(root, 'ogbn-'+dataset_name+'-test-partitions')
  seeds_data = torch.load(osp.join(seeds_dir, f'partition{node_rank}.pt'))

  print('** Initializing worker group context ...')
  glt.distributed.init_worker_group(
    world_size=num_nodes,
    rank=node_rank,
    group_name='dist-neighbor-loader-benchmarks'
  )
  dist_context = glt.distributed.get_context()

  print('** Initializing process group')
  dist.init_process_group('gloo', rank=dist_context.rank,
                          world_size=dist_context.world_size,
                          init_method='tcp://{}:{}'.format(master_addr, torch_pg_master_port))

  print('** Launching dist neighbor loader ...')
  dist_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[15, 10, 5],
    input_nodes=seeds_data,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=True,
    with_edge=with_edge,
    collect_features=collect_features,
    to_device=torch.device(0),
    worker_options=glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=sampling_nprocs,
      worker_devices=[torch.device('cuda', i % device_count) for i in range(sampling_nprocs)],
      worker_concurrency=worker_concurrency,
      master_addr=master_addr,
      master_port=sampling_master_port,
      channel_size=channel_size,
      pin_memory=True
    )
  )

  print('** Benchmarking ...')
  f = open('benchmark.txt', 'a+')
  for epoch in range(epochs):
    num_sampled_nodes = 0
    num_sampled_edges = 0
    num_collected_features = 0
    start = time.time()
    for i, batch in enumerate(dist_loader):
      if i % 100 == 0:
        f.write('Epoch {}, Batch {}\n'.format(epoch, i))
      num_sampled_nodes += batch.node.numel()
      num_sampled_edges += batch.edge_index.size(1)
      if batch.x is not None:
        num_collected_features += batch.x.size(0)
    torch.cuda.synchronize()
    total_time = time.time() - start
    f.write('** Epoch {} **\n'.format(epoch))
    f.write('- total time: {}s\n'.format(total_time))
    f.write('- total sampled nodes: {}\n'.format(num_sampled_nodes))
    f.write('- sampling nodes per sec: {} M\n'.format((num_sampled_nodes / total_time) / 1000000))
    f.write('- total sampled edges: {}\n'.format(num_sampled_edges))
    f.write('- sampling edges per sec: {} M\n'.format((num_sampled_edges / total_time) / 1000000))
    f.write('- total collected features: {}\n'.format(num_collected_features))
    f.write('- collecting features per sec: {} M\n'.format((num_collected_features / total_time) / 1000000))
    dist.barrier()

  time.sleep(1)
  print('** Exit ...')
