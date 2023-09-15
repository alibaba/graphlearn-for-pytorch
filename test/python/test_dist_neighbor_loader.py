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
import os

import torch
import graphlearn_torch as glt

from dist_test_utils import *
from dist_test_utils import _prepare_dataset, _prepare_hetero_dataset
from parameterized import parameterized
from typing import List, Optional

def _check_sample_result(data, edge_dir):
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

  tc.assertEqual(data.num_sampled_nodes[0].item(), 5)
  tc.assertEqual(data.num_sampled_nodes.size(0), 3)
  tc.assertNotEqual(data.num_sampled_nodes[1].item(), 0)
  tc.assertNotEqual(data.num_sampled_nodes[2].item(), 0)
  tc.assertEqual(data.num_sampled_edges[0].item(), 10)
  tc.assertEqual(data.num_sampled_edges.size(0), 2)
  tc.assertNotEqual(data.num_sampled_edges[1].item(), 0)


def _check_hetero_sample_result(data, edge_dir):
  tc = unittest.TestCase()
  if edge_dir == 'out':
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

    tc.assertEqual(data.num_sampled_nodes['item'][0].item(), 0)
    tc.assertNotEqual(data.num_sampled_nodes['item'][1].item(), 0)
    tc.assertNotEqual(data.num_sampled_nodes['item'][2].item(), 0)
    tc.assertEqual(data.num_sampled_nodes['user'][0].item(), 5)
    tc.assertEqual(data.num_sampled_nodes['user'][1].item(), 0)
    tc.assertEqual(data.num_sampled_nodes['user'][2].item(), 0)
    tc.assertEqual(data.num_sampled_edges['item', 'rev_u2i', 'user'][0].item(), 10)
    tc.assertEqual(data.num_sampled_edges['item', 'rev_u2i', 'user'][1].item(), 0)
    tc.assertEqual(data.num_sampled_edges['item', 'i2i', 'item'][0].item(), 0)
    tc.assertNotEqual(data.num_sampled_edges['item', 'i2i', 'item'][1].item(), 0)
  else:
    tc.assertEqual(data['num_sampled_nodes']['item'].size(0), 3)
    tc.assertEqual(data['num_sampled_nodes']['user'].size(0), 3)
    tc.assertEqual(
      data['num_sampled_edges'][('user', 'u2i', 'item')].size(0), 2)
    tc.assertEqual(
      data['num_sampled_edges'][('item', 'i2i', 'item')].size(0), 2)
    tc.assertTrue(data[('user', 'u2i', 'item')].edge_attr.size(1), 10)
    tc.assertTrue(data[('item', 'i2i', 'item')].edge_attr.size(1), 5)

    u2i_row = data['user'].node[data[('user', 'u2i', 'item')].edge_index[0]]
    u2i_col = data['item'].node[data[('user', 'u2i', 'item')].edge_index[1]]
    i2i_row = data['item'].node[data[('item', 'i2i', 'item')].edge_index[0]]
    i2i_col = data['item'].node[data[('item', 'i2i', 'item')].edge_index[1]]
    tc.assertEqual(u2i_row.size(0), u2i_col.size(0))
    tc.assertEqual(i2i_row.size(0), i2i_row.size(0))
    tc.assertTrue(torch.all(
      ((u2i_row+2)%vnum_total == u2i_col) + ((u2i_row+1)%vnum_total == u2i_col))
    )
    tc.assertTrue(torch.all(
      ((i2i_row+2)%vnum_total == i2i_col) + ((i2i_row+3)%vnum_total == i2i_col))
    )

def run_test_as_worker(world_size: int, rank: int,
                       master_port: int, sampling_master_port: int,
                       dataset: glt.distributed.DistDataset,
                       input_nodes: glt.InputNodes, check_fn,
                       collocated = False, edge_dir='out'):
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
    edge_dir=edge_dir,
    collect_features=True,
    to_device=torch.device('cuda', rank % device_num),
    worker_options=worker_options
  )

  # run testing
  for epoch in range(0, 2):
    for res in dist_loader:
      check_fn(res, edge_dir)
      time.sleep(0.1)
    glt.distributed.barrier()
    print(f'[Trainer {dist_context.rank}] epoch {epoch} finished.')

  dist_loader.shutdown()


def run_test_as_server(num_servers: int, num_clients: int, server_rank: List[int],
                       master_port: int, dataset: glt.distributed.DistDataset, is_dynamic: bool = False):
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
    server_group_name='dist_remote_sampling_test_server',
    is_dynamic=is_dynamic
  )

  print(f'[Server {server_rank}] Waiting for exit ...')
  glt.distributed.wait_and_shutdown_server()

  print(f'[Server {server_rank}] Exited ...')


def run_test_as_client(num_servers: int, num_clients: int, client_rank: int, server_rank: Optional[List[int]],
                       master_port: int, sampling_master_port: int,
                       input_nodes: glt.InputNodes, check_fn, edge_dir='out',
                       is_dynamic: bool = False):
    print(f'[Client {client_rank}] Initializing client ...')
    glt.distributed.init_client(
      num_servers=num_servers,
      num_clients=num_clients,
      client_rank=client_rank,
      master_addr='localhost',
      master_port=master_port,
      num_rpc_threads=1,
      client_group_name='dist_remote_sampling_test_client',
      is_dynamic=is_dynamic
    )

    print(f'[Client {client_rank}] Creating DistNeighborLoader ...')
    
    options = glt.distributed.RemoteDistSamplingWorkerOptions(
      # Automatically assign server_rank (server_rank_list) if server_rank (server_rank_list) is None
      server_rank=server_rank,
      num_workers=sampling_nprocs,
      worker_devices=[torch.device('cuda', i % device_num)
                      for i in range(sampling_nprocs)],
      worker_concurrency=2,
      master_addr='localhost',
      master_port=sampling_master_port,
      rpc_timeout=10,
      num_rpc_threads=2,
      prefetch_size=2,
      worker_key='unittest'
    )
    dist_loader = glt.distributed.DistNeighborLoader(
      data=None,
      num_neighbors=[2, 2],
      input_nodes=input_nodes,
      batch_size=5,
      shuffle=True,
      drop_last=False,
      with_edge=True,
      edge_dir=edge_dir,
      collect_features=True,
      to_device=torch.device('cuda', client_rank % device_num),
      worker_options=options
    )

    print(f'[Client {client_rank}] Running tests ...')
    for epoch in range(0, 2):
      num_batches = 0
      for res in dist_loader:
        num_batches += 1
        check_fn(res, edge_dir)
        time.sleep(0.1)
      glt.distributed.barrier()
      print(f'[Client {client_rank}] epoch {epoch} finished with {num_batches} batches.')

    print(f'[Client {client_rank}] Shutdowning ...')
    glt.distributed.shutdown_client()

    print(f'[Client {client_rank}] Exited ...')


class DistNeighborLoaderTestCase(unittest.TestCase):

  input_nodes0_path = 'input_nodes0.pt'
  input_nodes1_path = 'input_nodes1.pt'

  def setUp(self):
    self.dataset0 = _prepare_dataset(rank=0)
    self.dataset1 = _prepare_dataset(rank=1)

    # all for train
    self.dataset0.random_node_split(0, 0)
    self.dataset1.random_node_split(0, 0)
    
    self.input_nodes0 = torch.arange(vnum_per_partition)
    self.input_nodes1 = torch.arange(vnum_per_partition) + vnum_per_partition

    torch.save(self.input_nodes0, self.input_nodes0_path)
    torch.save(self.input_nodes1, self.input_nodes1_path)

    self.in_hetero_dataset0 = _prepare_hetero_dataset(rank=0, edge_dir='in')
    self.in_hetero_dataset1 = _prepare_hetero_dataset(rank=1, edge_dir='in')
    self.out_hetero_dataset0 = _prepare_hetero_dataset(rank=0, edge_dir='out')
    self.out_hetero_dataset1 = _prepare_hetero_dataset(rank=1, edge_dir='out')

    self.out_hetero_input_nodes0 = (user_ntype, self.input_nodes0)
    self.out_hetero_input_nodes1 = (user_ntype, self.input_nodes1)
    self.in_hetero_input_nodes0 = (item_ntype, self.input_nodes0)
    self.in_hetero_input_nodes1 = (item_ntype, self.input_nodes1)
    self.master_port = glt.utils.get_free_port()
    self.sampling_master_port = glt.utils.get_free_port()

  def tearDown(self):
    for file_path in [self.input_nodes0_path, self.input_nodes1_path]:
      if os.path.exists(file_path):
        os.remove(file_path)

  def test_homo_collocated(self):
    print("\n--- DistNeighborLoader Test (homogeneous, collocated) ---")
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

  def test_homo_mp(self):
    print("\n--- DistNeighborLoader Test (homogeneous, multiprocessing) ---")
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

  def test_hetero_out_sample_collocated(self):
    print("\n--- DistNeighborLoader Test (heterogeneous, collocated) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset0, self.out_hetero_input_nodes0,
            _check_hetero_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset1, self.out_hetero_input_nodes1,
            _check_hetero_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_out_sample_mp(self):
    print("\n--- DistNeighborLoader Test (heterogeneous, multiprocessing) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset0, self.out_hetero_input_nodes0,
            _check_hetero_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset1, self.out_hetero_input_nodes1,
            _check_hetero_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_in_sample_collocated(self):
    print("\n--- DistNeighborLoader Test (in-sample, heterogeneous, collocated) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset0, self.in_hetero_input_nodes0,
            _check_hetero_sample_result, True, 'in')
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset1, self.in_hetero_input_nodes1,
            _check_hetero_sample_result, True, 'in')
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_in_sample_mp(self):
    print("\n--- DistNeighborLoader Test (in-sample, heterogeneous, multiprocessing) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset0, self.in_hetero_input_nodes0,
            _check_hetero_sample_result, False, 'in')
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset1, self.in_hetero_input_nodes1,
            _check_hetero_sample_result, False, 'in')
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  @parameterized.expand([
    ([[0],[1]], 2, 2, "file_path"),
    ([[0, 1]], 1, 2, "file_path"),
    ([[0, 1], [0, 1]], 2, 2, "file_path"),
    ([[0],[1]], 2, 2, "split"),
    ([[0, 1]], 1, 2, "split"),
    ([[0, 1], [0, 1]], 2, 2, "split"),
  ])
  def test_remote_mode(self, servers_for_clients, num_clients, num_servers, input_nodes_type):
    print("\n--- DistNeighborLoader Test (server-client mode, remote) ---")
    print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")
    print(f"--- input_nodes_type: {input_nodes_type} ---")

    self.dataset_list = [self.dataset0, self.dataset1]
    # self.input_nodes_list = [self.input_nodes0, self.input_nodes1]
    self.input_nodes_path_list = [self.input_nodes0_path, self.input_nodes1_path]

    mp_context = torch.multiprocessing.get_context('spawn')
          
    server_procs = []
    for server_rank in range(num_servers):
      server_procs.append(mp_context.Process(
        target=run_test_as_server,
        args=(num_servers, num_clients, server_rank, self.master_port, self.dataset_list[server_rank])
      ))

    client_procs = []
    for client_rank in range(num_clients):
      server_rank_list = servers_for_clients[client_rank]
      if input_nodes_type == "split":
        input_nodes = glt.typing.Split.train
      elif input_nodes_type == "file_path":
        input_nodes = [self.input_nodes_path_list[server_rank] for server_rank in server_rank_list]

      client_procs.append(mp_context.Process(
        target=run_test_as_client,
        args=(num_servers, num_clients, client_rank, server_rank_list, self.master_port, 
            self.sampling_master_port, input_nodes, _check_sample_result)
      ))
    for sproc in server_procs:
      sproc.start()
    for cproc in client_procs:
      cproc.start()
    
    for sproc in server_procs:
      sproc.join()
    for cproc in client_procs:
      cproc.join()
      
  @parameterized.expand([
    ([[0],[1]], 2, 2),
    ([[0, 1]], 1, 2),
    ([[0, 1], [0, 1]], 2, 2),
  ])
  def test_remote_mode_dynamic_world_size(self, servers_for_clients, num_clients, num_servers):
    print("\n--- DistNeighborLoader Test (server-client mode, remote, dynamic world size) ---")
    print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")

    self.dataset_list = [self.dataset0, self.dataset1]
    self.input_nodes_list = [self.input_nodes0_path, self.input_nodes1_path]

    mp_context = torch.multiprocessing.get_context('spawn')

    server_procs = []
    for server_rank in range(num_servers):
      server_procs.append(mp_context.Process(
        target=run_test_as_server,
        # set `num_clients`=0 because this arg is not used in server-client mode with dynamic world size feature(`is_dynamic`=True).
        args=(num_servers, 0, server_rank, self.master_port, self.dataset_list[server_rank], True)
      ))

    client_procs = []
    for client_rank in range(num_clients):
      server_rank_list = servers_for_clients[client_rank]
      client_procs.append(mp_context.Process(
        target=run_test_as_client,
        args=(num_servers, num_clients, client_rank, server_rank_list, self.master_port, self.sampling_master_port,
            [self.input_nodes_list[server_rank] for server_rank in server_rank_list], _check_sample_result, 'out', True)
      ))
    for sproc in server_procs:
      sproc.start()
    for cproc in client_procs:
      cproc.start()
    
    for sproc in server_procs:
      sproc.join()
    for cproc in client_procs:
      cproc.join()
      
  @parameterized.expand([
    (2, 2),
    (1, 2)
  ])
  def test_remote_mode_auto_assign_server(self, num_clients, num_servers):
    print("\n--- DistNeighborLoader Test (server-client mode, remote, dynamic world size) ---")
    print(f"--- num_clients: {num_clients} num_servers: {num_servers} ---")

    self.dataset_list = [self.dataset0, self.dataset1]
    self.input_nodes_list = [self.input_nodes0_path, self.input_nodes1_path]

    mp_context = torch.multiprocessing.get_context('spawn')

    server_procs = []
    for server_rank in range(num_servers):
      server_procs.append(mp_context.Process(
        target=run_test_as_server,
        # set `num_clients`=0 because this arg is not used in server-client mode with dynamic world size feature(`is_dynamic`=True).
        args=(num_servers, 0, server_rank, self.master_port, self.dataset_list[server_rank], True)
      ))

    client_procs = []
    for client_rank in range(num_clients):
      client_procs.append(mp_context.Process(
        target=run_test_as_client,
        # set `server_rank`=None to test assign server rank automatically.
        args=(num_servers, num_clients, client_rank, None, self.master_port, self.sampling_master_port,
            [self.input_nodes_list[server_rank] for server_rank in range(num_servers)], _check_sample_result, 'out', True)
      ))
    for sproc in server_procs:
      sproc.start()
    for cproc in client_procs:
      cproc.start()
    
    for sproc in server_procs:
      sproc.join()
    for cproc in client_procs:
      cproc.join()


if __name__ == "__main__":
  unittest.main()
