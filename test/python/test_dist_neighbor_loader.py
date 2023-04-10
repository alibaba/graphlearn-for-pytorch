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

from dist_test_utils import *
from dist_test_utils import _prepare_dataset, _prepare_hetero_dataset


def _check_sample_result(data):
  tc = unittest.TestCase()
  tc.assertEqual(data.batch_size, 5)
  device = data.node.device
  label = torch.arange(vnum_total).to(device)
  tc.assertTrue(glt.utils.tensor_equal_with_device(data.y, label[data.node]))
  for i, v in enumerate(data.node):
    expect_feat = int(v) % 2 + torch.zeros(
      512, device=device, dtype=torch.float32
    )
    tc.assertTrue(glt.utils.tensor_equal_with_device(data.x[i], expect_feat))
  tc.assertTrue(data.edge is not None)
  tc.assertTrue(data.edge_attr is not None)
  for i, e in enumerate(data.edge):
    expect_feat = ((int(e) // degree) % 2) + torch.ones(
      10, device=device, dtype=torch.float32
    )
    tc.assertTrue(glt.utils.tensor_equal_with_device(data.edge_attr[i], expect_feat))
  rows = data.node[data.edge_index[0]]
  cols = data.node[data.edge_index[1]]
  for i in range(rows.size(0)):
    tc.assertTrue(
      int(rows[i]) == ((int(cols[i]) + 1) % vnum_total) or
      int(rows[i]) == ((int(cols[i]) + 2) % vnum_total)
    )


def _check_hetero_sample_result(data):
  tc = unittest.TestCase()
  tc.assertEqual(data[user_ntype].batch_size, 5)
  device = data[user_ntype].node.device
  user_label = torch.arange(vnum_total).to(device)
  tc.assertTrue(glt.utils.tensor_equal_with_device(
    data[user_ntype].y, user_label[data[user_ntype].node]
  ))
  for i, v in enumerate(data[user_ntype].node):
    expect_feat = int(v) % 2 + torch.zeros(512, device=device, dtype=torch.float32)
    tc.assertTrue(glt.utils.tensor_equal_with_device(
      data.x_dict[user_ntype][i], expect_feat
    ))
  for i, v in enumerate(data[item_ntype].node):
    expect_feat = (int(v) % 2) * 2 + torch.zeros(
      256, device=device, dtype=torch.float32
    )
    tc.assertTrue(glt.utils.tensor_equal_with_device(
      data.x_dict[item_ntype][i], expect_feat
    ))
  rev_u2i_etype = glt.reverse_edge_type(u2i_etype)
  rev_i2i_etype = glt.reverse_edge_type(i2i_etype)
  tc.assertTrue(data[rev_u2i_etype].edge is not None)
  tc.assertTrue(data[rev_u2i_etype].edge_attr is not None)
  for i, e in enumerate(data[rev_u2i_etype].edge):
    expect_feat = ((int(e) // degree) % 2) + torch.ones(
      10, device=device, dtype=torch.float32
    )
    tc.assertTrue(glt.utils.tensor_equal_with_device(
      data.edge_attr_dict[rev_u2i_etype][i], expect_feat
    ))
  tc.assertTrue(data[rev_i2i_etype].edge is not None)
  tc.assertTrue(data[rev_i2i_etype].edge_attr is not None)
  for i, e in enumerate(data[rev_i2i_etype].edge):
    expect_feat = ((int(e) // degree) % 2) * 2 + torch.ones(
      5, device=device, dtype=torch.float32
    )
    tc.assertTrue(glt.utils.tensor_equal_with_device(
      data.edge_attr_dict[rev_i2i_etype][i], expect_feat
    ))
  rev_u2i_rows = data[item_ntype].node[data.edge_index_dict[rev_u2i_etype][0]]
  rev_u2i_cols = data[user_ntype].node[data.edge_index_dict[rev_u2i_etype][1]]
  tc.assertEqual(rev_u2i_rows.size(0), rev_u2i_cols.size(0))
  for i in range(rev_u2i_rows.size(0)):
    tc.assertTrue(
      (int(rev_u2i_rows[i]) == ((int(rev_u2i_cols[i]) + 1) % vnum_total) or
      int(rev_u2i_rows[i]) == ((int(rev_u2i_cols[i]) + 2) % vnum_total))
    )
  rev_i2i_rows = data[item_ntype].node[data.edge_index_dict[rev_i2i_etype][0]]
  rev_i2i_cols = data[item_ntype].node[data.edge_index_dict[rev_i2i_etype][1]]
  tc.assertEqual(rev_i2i_rows.size(0), rev_i2i_cols.size(0))
  for i in range(rev_i2i_rows.size(0)):
    tc.assertTrue(
      int(rev_i2i_rows[i]) == ((int(rev_i2i_cols[i]) + 2) % vnum_total) or
      int(rev_i2i_rows[i]) == ((int(rev_i2i_cols[i]) + 3) % vnum_total)
    )


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
  dist_loader = glt.distributed.DistNeighborLoader(
    data=dataset,
    num_neighbors=[2, 2],
    input_nodes=input_nodes,
    batch_size=5,
    shuffle=True,
    drop_last=False,
    with_edge=True,
    collect_features=True,
    to_device=torch.device('cuda', rank % device_num),
    worker_options=worker_options
  )

  # run testing
  for epoch in range(0, 2):
    for res in dist_loader:
      check_fn(res)
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

    print(f'[Client {client_rank}] Creating DistNeighborLoader ...')
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
    dist_loader = glt.distributed.DistNeighborLoader(
      data=None,
      num_neighbors=[2, 2],
      input_nodes=input_nodes,
      batch_size=5,
      shuffle=True,
      drop_last=False,
      with_edge=True,
      collect_features=True,
      to_device=torch.device('cuda', client_rank % device_num),
      worker_options=options
    )

    print(f'[Client {client_rank}] Running tests ...')
    for epoch in range(0, 2):
      for res in dist_loader:
        check_fn(res)
        time.sleep(0.1)
      glt.distributed.barrier()
      print(f'[Client {client_rank}] epoch {epoch} finished.')

    print(f'[Client {client_rank}] Shutdowning ...')
    glt.distributed.shutdown_client()

    print(f'[Client {client_rank}] Exited ...')


class DistNeighborLoaderTestCase(unittest.TestCase):
  def setUp(self):
    self.dataset0 = _prepare_dataset(rank=0)
    self.dataset1 = _prepare_dataset(rank=1)
    self.input_nodes0 = torch.arange(vnum_per_partition)
    self.input_nodes1 = torch.arange(vnum_per_partition) + vnum_per_partition

    self.hetero_dataset0 = _prepare_hetero_dataset(rank=0)
    self.hetero_dataset1 = _prepare_hetero_dataset(rank=1)
    self.hetero_input_nodes0 = (user_ntype, self.input_nodes0)
    self.hetero_input_nodes1 = (user_ntype, self.input_nodes1)

  def test_homo_collocated(self):
    print("\n--- DistNeighborLoader Test (homogeneous, collocated) ---")
    master_port = glt.utils.get_free_port()
    sampling_master_port = glt.utils.get_free_port()
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, master_port, sampling_master_port,
            self.dataset0, self.input_nodes0, _check_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, master_port, sampling_master_port,
            self.dataset1, self.input_nodes1, _check_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_homo_mp(self):
    print("\n--- DistNeighborLoader Test (homogeneous, multiprocessing) ---")
    master_port = glt.utils.get_free_port()
    sampling_master_port = glt.utils.get_free_port()
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, master_port, sampling_master_port,
            self.dataset0, self.input_nodes0, _check_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, master_port, sampling_master_port,
            self.dataset1, self.input_nodes1, _check_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_collocated(self):
    print("\n--- DistNeighborLoader Test (heterogeneous, collocated) ---")
    master_port = glt.utils.get_free_port()
    sampling_master_port = glt.utils.get_free_port()
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, master_port, sampling_master_port,
            self.hetero_dataset0, self.hetero_input_nodes0,
            _check_hetero_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, master_port, sampling_master_port,
            self.hetero_dataset1, self.hetero_input_nodes1,
            _check_hetero_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_mp(self):
    print("\n--- DistNeighborLoader Test (heterogeneous, multiprocessing) ---")
    master_port = glt.utils.get_free_port()
    sampling_master_port = glt.utils.get_free_port()
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, master_port, sampling_master_port,
            self.hetero_dataset0, self.hetero_input_nodes0,
            _check_hetero_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, master_port, sampling_master_port,
            self.hetero_dataset1, self.hetero_input_nodes1,
            _check_hetero_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_remote_mode(self):
    print("\n--- DistNeighborLoader Test (server-client mode, remote) ---")
    master_port = glt.utils.get_free_port()
    sampling_master_port = glt.utils.get_free_port()
    mp_context = torch.multiprocessing.get_context('spawn')
    server0 = mp_context.Process(
      target=run_test_as_server,
      args=(2, 2, 0, master_port, self.dataset0)
    )
    server1 = mp_context.Process(
      target=run_test_as_server,
      args=(2, 2, 1, master_port, self.dataset1)
    )
    client0 = mp_context.Process(
      target=run_test_as_client,
      args=(2, 2, 0, master_port, sampling_master_port,
            self.input_nodes0, _check_sample_result)
    )
    client1 = mp_context.Process(
      target=run_test_as_client,
      args=(2, 2, 1, master_port, sampling_master_port,
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
