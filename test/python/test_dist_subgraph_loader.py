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

import time
import unittest

import torch
import graphlearn_torch as glt

# sampling options
sampling_nprocs = 2
device_num = 2

def _prepare_dataset(rank: int):
  """
  input graph:
    1 1 0 0 0 0 0 1
    0 1 0 1 0 0 1 0
    0 0 1 0 1 0 1 0
    0 0 0 0 0 1 0 0
    0 1 0 1 0 0 0 0
    1 0 0 0 0 1 0 0
    1 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 1
  supppose first 4 rows of above matrix are partitioned to partiton#0 and the
  rest are belong to partition#1.
  """
  # partition
  node_pb = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
  edge_pb = torch.tensor([0] * 10 + [1] * 7, dtype=torch.long)

  # graph
  nodes, rows, cols, eids = [], [], [], []
  if rank == 0:
    nodes = [0, 1, 2, 3]
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    cols = [0, 1, 7, 1, 3, 6, 2, 4, 6, 5]
    eids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  else:
    nodes = [4, 5, 6, 7]
    rows = [4, 4, 5, 5, 6, 6, 7]
    cols = [1, 3, 0, 5, 0, 4, 7]
    eids = [10, 11, 12, 13, 14, 15, 16]
  edge_index = torch.tensor([rows, cols], dtype=torch.int64)
  edge_ids = torch.tensor(eids, dtype=torch.int64)
  csr_topo = glt.data.Topology(edge_index=edge_index, edge_ids=edge_ids)
  graph = glt.data.Graph(csr_topo, 'ZERO_COPY', device=0)
  # feature
  device_group_list = [glt.data.DeviceGroup(0, [0]),
                       glt.data.DeviceGroup(1, [1])]
  split_ratio = 0.2
  nfeat = rank + torch.zeros(len(nodes), 512, dtype=torch.float32)
  nfeat_id2idx = glt.utils.id2idx(nodes)
  node_feature = glt.data.Feature(nfeat, nfeat_id2idx, split_ratio,
                                  device_group_list, device=0)

  efeat = rank + torch.ones(len(eids), 10, dtype=torch.float32)
  efeat_id2idx = glt.utils.id2idx(eids)
  edge_feature = glt.data.Feature(efeat, efeat_id2idx, split_ratio,
                                  device_group_list, device=0)
  # dist dataset
  return glt.distributed.DistDataset(
    2, rank,
    graph, node_feature, edge_feature, None,
    node_pb, edge_pb,
  )

def _check_sample_result(data, rank):
  tc = unittest.TestCase()
  if rank == 0:
    true_node = torch.tensor([0, 1, 3, 5, 6, 7], device='cuda:0')
    true_edge_index = torch.tensor([[0, 1, 5, 1, 2, 4, 3, 0, 3, 0, 5],
                                    [0, 0, 0, 1, 1, 1, 2, 3, 3, 4, 5]],
                                  device='cuda:0')
    true_edge_id = torch.tensor([0, 1, 2, 3, 4, 5, 9, 12, 13, 14, 16], device='cuda:0')
    true_mapping = torch.tensor([0, 2, 5], device='cuda:0')
  else:
    true_node = torch.tensor([0, 1, 3, 5, 6, 7], device='cuda:1')
    true_edge_index = torch.tensor([[0, 3, 0, 5, 0, 1, 5, 1, 2, 4, 3],
                                    [3, 3, 4, 5, 0, 0, 0, 1, 1, 1, 2]],
                                    device='cuda:1')
    true_edge_id = torch.tensor([12, 13, 14, 16, 0, 1, 2, 3, 4, 5, 9], device='cuda:1')
    true_mapping = torch.tensor([0, 2, 5], device='cuda:1')
  tc.assertTrue(glt.utils.tensor_equal_with_device(data.node, true_node))
  tc.assertTrue(glt.utils.tensor_equal_with_device(data.edge_index, true_edge_index))
  tc.assertTrue(glt.utils.tensor_equal_with_device(data.edge, true_edge_id))
  tc.assertTrue(glt.utils.tensor_equal_with_device(data.mapping, true_mapping))

  device = data.node.device
  for i, v in enumerate(data.node):
    expect_feat = torch.zeros(512, device=device, dtype=torch.float32)
    if v > 3: # rank1
      expect_feat += 1
    tc.assertTrue(glt.utils.tensor_equal_with_device(data.x[i], expect_feat))
  tc.assertTrue(data.edge_attr is not None)
  for i, e in enumerate(data.edge):
    expect_feat = torch.ones(10, device=device, dtype=torch.float32)
    if e > 9: # rank1
      expect_feat += 1
    tc.assertTrue(glt.utils.tensor_equal_with_device(data.edge_attr[i], expect_feat))


def run_test_as_worker(world_size: int, rank: int,
                       master_port: int, sampling_master_port: int,
                       dataset: glt.distributed.DistDataset,
                       input_nodes: glt.InputNodes, check_fn,
                       collocated = False):
  # Initialize worker group context
  glt.distributed.init_worker_group(
    world_size, rank, 'dist-neighbor-loader-test'
  )
  dist_context = glt.distributed.get_context()

  # Init RPC
  glt.distributed.init_rpc(
    master_addr='localhost',
    master_port=master_port,
    num_rpc_threads=1,
    rpc_timeout=30
  )

  # dist loader
  if collocated:
    worker_options = glt.distributed.CollocatedDistSamplingWorkerOptions(
      master_addr='localhost',
      master_port=sampling_master_port,
      rpc_timeout=10
    )
  else:
    worker_options = glt.distributed.MpDistSamplingWorkerOptions(
      num_workers=sampling_nprocs,
      worker_devices=[torch.device('cuda', i % device_num)
                      for i in range(sampling_nprocs)],
      worker_concurrency=2,
      master_addr='localhost',
      master_port=sampling_master_port,
      rpc_timeout=10,
      num_rpc_threads=2,
      pin_memory=True
    )
  dist_loader = glt.distributed.DistSubGraphLoader(
    data=dataset,
    num_neighbors=[-1, -1],
    input_nodes=input_nodes,
    batch_size=3,
    shuffle=False,
    drop_last=False,
    with_edge=True,
    collect_features=True,
    to_device=torch.device('cuda', rank % device_num),
    worker_options=worker_options
  )

  # run testing
  for epoch in range(0, 2):
    for res in dist_loader:
      check_fn(res, rank)
      time.sleep(0.1)
    glt.distributed.barrier()
    print(f'[Trainer {dist_context.rank}] epoch {epoch} finished.')

  dist_loader.shutdown()


def run_test_as_server(num_servers: int, num_clients: int, server_rank: int,
                       master_port: int, dataset: glt.distributed.DistDataset):
  print(f'[Server {server_rank}] Initializing server ...')
  glt.distributed.init_server(
    num_servers=num_servers,
    num_clients=num_clients,
    server_rank=server_rank,
    dataset=dataset,
    master_addr='localhost',
    master_port=master_port,
    request_timeout=30,
    num_rpc_threads=2,
    server_group_name='dist-remote-sampling-test-server'
  )

  print(f'[Server {server_rank}] Waiting for exit ...')
  glt.distributed.wait_and_shutdown_server()

  print(f'[Server {server_rank}] Exited ...')


def run_test_as_client(num_servers: int, num_clients: int, client_rank: int,
                       master_port: int, sampling_master_port: int,
                       input_nodes: glt.InputNodes, check_fn):
    print(f'[Client {client_rank}] Initializing client ...')
    glt.distributed.init_client(
      num_servers=num_servers,
      num_clients=num_clients,
      client_rank=client_rank,
      master_addr='localhost',
      master_port=master_port,
      num_rpc_threads=1,
      client_group_name='dist-remote-sampling-test-client'
    )

    print(f'[Client {client_rank}] Creating DistSubGraphLoader ...')
    target_server_rank = client_rank % num_servers
    options = glt.distributed.RemoteDistSamplingWorkerOptions(
      server_rank=target_server_rank,
      num_workers=sampling_nprocs,
      worker_devices=[torch.device('cuda', i % device_num)
                      for i in range(sampling_nprocs)],
      worker_concurrency=2,
      master_addr='localhost',
      master_port=sampling_master_port,
      rpc_timeout=10,
      num_rpc_threads=2,
      prefetch_size=4
    )
    dist_loader = glt.distributed.DistSubGraphLoader(
      data=None,
      num_neighbors=[-1, -1],
      input_nodes=input_nodes,
      batch_size=3,
      shuffle=False,
      drop_last=False,
      with_edge=True,
      collect_features=True,
      to_device=torch.device('cuda', client_rank % device_num),
      worker_options=options
    )

    print(f'[Client {client_rank}] Running tests ...')
    for epoch in range(0, 2):
      for res in dist_loader:
        check_fn(res, client_rank)
        time.sleep(0.1)
      glt.distributed.barrier()
      print(f'[Client {client_rank}] epoch {epoch} finished.')

    print(f'[Client {client_rank}] Shutdowning ...')
    glt.distributed.shutdown_client()

    print(f'[Client {client_rank}] Exited ...')


class DistSubGraphLoaderTestCase(unittest.TestCase):
  def setUp(self):
    self.dataset0 = _prepare_dataset(rank=0)
    self.dataset1 = _prepare_dataset(rank=1)
    self.input_nodes0 = torch.tensor([0, 3, 7], dtype=torch.long)
    self.input_nodes1 = self.input_nodes0
    self.master_port = glt.utils.get_free_port()
    self.sampling_master_port = glt.utils.get_free_port()

  def test_collocated(self):
    print("\n--- DistSubGraphLoader Test (collocated) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.dataset0, self.input_nodes0, _check_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.dataset1, self.input_nodes1, _check_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_mp(self):
    print("\n--- DistSubGraphLoader Test (multiprocessing) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.dataset0, self.input_nodes0, _check_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.dataset1, self.input_nodes1, _check_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_remote_mode(self):
    print("\n--- DistSubGraphLoader Test (server-client mode, remote) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    server0 = mp_context.Process(
      target=run_test_as_server,
      args=(2, 2, 0, self.master_port, self.dataset0)
    )
    server1 = mp_context.Process(
      target=run_test_as_server,
      args=(2, 2, 1, self.master_port, self.dataset1)
    )
    client0 = mp_context.Process(
      target=run_test_as_client,
      args=(2, 2, 0, self.master_port, self.sampling_master_port,
            self.input_nodes0, _check_sample_result)
    )
    client1 = mp_context.Process(
      target=run_test_as_client,
      args=(2, 2, 1, self.master_port, self.sampling_master_port,
            self.input_nodes1, _check_sample_result)
    )
    server0.start()
    server1.start()
    client0.start()
    client1.start()
    server0.join()
    server1.join()
    client0.join()
    client1.join()


if __name__ == "__main__":
  unittest.main()
