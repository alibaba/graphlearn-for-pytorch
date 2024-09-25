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
from parameterized import parameterized

def _check_sample_result(data, edge_dir='out'):
  tc = unittest.TestCase()

  if 'src_index' in data:
    # triplet negative sampling
    tc.assertEqual(data.src_index.size(0), 5)
    tc.assertEqual(data.dst_pos_index.size(0), 5)
    tc.assertEqual(data.dst_neg_index.size(0), 5)
    tc.assertTrue(data.edge_attr is not None)

    pos_index = torch.stack(
      (data.node[data.src_index],
       data.node[data.dst_pos_index]
    ))
    tc.assertTrue(torch.all(
      ((pos_index[0]+1)%40==pos_index[1]) + ((pos_index[0]+2)%40==pos_index[1])
    ))
  else:
    # binary negative sampling
    tc.assertEqual(data.edge_label_index.size(1), 10)
    tc.assertEqual(data.edge_label.size(0), 10)
    tc.assertTrue(data.edge_attr is not None)
    tc.assertEqual(max(data.edge_label), 1)

    out_index = data.edge_label_index
    pos_index = torch.stack(
      (data.node[out_index[0,:5]],
       data.node[out_index[1,:5]]
    ))
    sub_edge_index = torch.stack((data.node[data.edge_index[0]],
                                  data.node[data.edge_index[1]]))
    tc.assertTrue(torch.all(
      ((pos_index[1]+1)%40==pos_index[0]) + ((pos_index[1]+2)%40==pos_index[0])
    ))
    tc.assertTrue(torch.all(
      ((sub_edge_index[1]+1)%40==sub_edge_index[0]) +
      ((sub_edge_index[1]+2)%40==sub_edge_index[0])
    ))


def _check_hetero_sample_result(data, edge_dir='out'):
  tc = unittest.TestCase()
  if edge_dir == 'out':
    if len(data[user_ntype]) > 2:
      # triplet negative sampling
      tc.assertEqual(data[user_ntype].node.size(0), 5)
      tc.assertEqual(data[user_ntype].src_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_pos_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_neg_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_neg_index.size(1), 2)
      tc.assertEqual(data[rev_u2i_etype].edge.size(0), 10)
      tc.assertTrue(data[rev_u2i_etype].edge_attr is not None)
      tc.assertTrue(data[i2i_etype].edge_attr is not None)
      tc.assertLess(max(data[user_ntype].src_index), 5)
      tc.assertLess(max(data[rev_u2i_etype].edge_index[1]), 5)

      pos_index = torch.stack(
        (data[user_ntype].node[data[user_ntype].src_index],
        data[item_ntype].node[data[item_ntype].dst_pos_index]
      ))
      tc.assertTrue(torch.all(
        ((pos_index[0]+1)%40==pos_index[1]) + ((pos_index[0]+2)%40==pos_index[1])
      ))

    else:
      # binary negative sampling
      tc.assertLessEqual(data[user_ntype].node.size(0), 10)
      tc.assertEqual(data[rev_u2i_etype].edge_label_index.size(0), 2)
      tc.assertEqual(data[rev_u2i_etype].edge_label_index.size(1), 10)
      tc.assertEqual(data[rev_u2i_etype].edge_label.size(0), 10)
      tc.assertEqual(max(data[rev_u2i_etype].edge_label), 1)
      tc.assertTrue(data[rev_u2i_etype].edge_attr is not None)
      tc.assertTrue(data[i2i_etype].edge_attr is not None)

      out_index = data[rev_u2i_etype].edge_label_index
      pos_index = torch.stack(
        (data[item_ntype].node[out_index[0,:int(out_index.size(1)/2)]],
        data[user_ntype].node[out_index[1,:int(out_index.size(1)/2)]])
      )
      neg_index = torch.stack(
        (data[item_ntype].node[out_index[0,int(out_index.size(1)/2):]],
        data[user_ntype].node[out_index[1,int(out_index.size(1)/2):]])
      )

      tc.assertTrue(torch.all(
        ((pos_index[1]+1)%40==pos_index[0]) + ((pos_index[1]+2)%40==pos_index[0])
      ))
      tc.assertEqual(neg_index.size(0), pos_index.size(0))
      tc.assertEqual(neg_index.size(1), pos_index.size(1))

      sub_edge_index = data[rev_u2i_etype].edge_index
      glob_edge_index = torch.stack((data[item_ntype].node[sub_edge_index[0]],
                                    data[user_ntype].node[sub_edge_index[1]]))
      tc.assertTrue(torch.all(
        ((glob_edge_index[1]+1)%40==glob_edge_index[0]) +
        ((glob_edge_index[1]+2)%40==glob_edge_index[0])
      ))

  elif edge_dir == 'in':
    if len(data[user_ntype]) > 2:
      tc.assertTrue(data[u2i_etype].edge_attr.size(1), 10)
      tc.assertTrue(data[i2i_etype].edge_attr.size(1), 5)
      tc.assertEqual(data[user_ntype].src_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_pos_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_neg_index.size(0), 5)
      tc.assertEqual(data[item_ntype].dst_neg_index.size(1), 2)
      u2i_row = data[user_ntype].node[data[u2i_etype].edge_index[0]]
      u2i_col = data[item_ntype].node[data[u2i_etype].edge_index[1]]
      i2i_row = data[item_ntype].node[data[i2i_etype].edge_index[0]]
      i2i_col = data[item_ntype].node[data[i2i_etype].edge_index[1]]
      pos_index = torch.stack(
        (data[user_ntype].node[data[user_ntype].src_index],
         data[item_ntype].node[data[item_ntype].dst_pos_index])
      )

      tc.assertTrue(torch.all(
        ((u2i_row+1)%vnum_total == u2i_col) + ((u2i_row+2)%vnum_total == u2i_col)
      ))
      tc.assertTrue(torch.all(
        ((i2i_row+1)%vnum_total == i2i_col) + ((i2i_row+2)%vnum_total == i2i_col)
      ))
      tc.assertTrue(torch.all(
        ((pos_index[0]+1)%vnum_total==pos_index[1]) + ((pos_index[0]+2)%vnum_total==pos_index[1])
      ))

    else:
      tc.assertEqual(max(data[u2i_etype].edge_label), 1)
      tc.assertTrue(data[u2i_etype].edge_attr.size(1), 10)
      tc.assertTrue(data[i2i_etype].edge_attr.size(1), 5)
      tc.assertEqual(data[u2i_etype].edge_label_index.size(0), 2)
      tc.assertEqual(data[u2i_etype].edge_label_index.size(1), 10)
      tc.assertEqual(data[u2i_etype].edge_label.size(0), 10)
      u2i_row = data[user_ntype].node[data[u2i_etype].edge_index[0]]
      u2i_col = data[item_ntype].node[data[u2i_etype].edge_index[1]]
      i2i_row = data[item_ntype].node[data[i2i_etype].edge_index[0]]
      i2i_col = data[item_ntype].node[data[i2i_etype].edge_index[1]]
      out_index = data[u2i_etype].edge_label_index
      pos_index = torch.stack(
        (data[user_ntype].node[out_index[0,:int(out_index.size(1)/2)]],
        data[item_ntype].node[out_index[1,:int(out_index.size(1)/2)]])
      )
      neg_index = torch.stack(
        (data[user_ntype].node[out_index[0,int(out_index.size(1)/2):]],
        data[item_ntype].node[out_index[1,int(out_index.size(1)/2):]])
      )

      tc.assertTrue(torch.all(
        ((u2i_row+1)%vnum_total == u2i_col) + ((u2i_row+2)%vnum_total == u2i_col)
      ))
      tc.assertTrue(torch.all(
        ((i2i_row+1)%vnum_total == i2i_col) + ((i2i_row+2)%vnum_total == i2i_col)
      ))

      tc.assertTrue(torch.all(
        ((pos_index[0]+1)%vnum_total==pos_index[1]) +
        ((pos_index[0]+2)%vnum_total==pos_index[1])
      ))
      tc.assertEqual(neg_index.size(0), pos_index.size(0))
      tc.assertEqual(neg_index.size(1), pos_index.size(1))
      sub_edge_index = data[u2i_etype].edge_index
      glob_edge_index = torch.stack((data[user_ntype].node[sub_edge_index[0]],
                                    data[item_ntype].node[sub_edge_index[1]]))
      tc.assertTrue(torch.all(
        ((glob_edge_index[0]+1)%vnum_total==glob_edge_index[1]) +
        ((glob_edge_index[0]+2)%vnum_total==glob_edge_index[1])
      ))


def run_test_as_worker(world_size: int, rank: int,
                       master_port: int, sampling_master_port: int,
                       dataset: glt.distributed.DistDataset,
                       neg_sampling: glt.sampler.NegativeSampling,
                       input_edges: glt.InputEdges, check_fn,
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
  dist_loader = glt.distributed.DistLinkNeighborLoader(
    data=dataset,
    num_neighbors=[2, 1],
    edge_label_index=input_edges,
    neg_sampling=neg_sampling,
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
                       neg_sampling: glt.sampler.NegativeSampling,
                       input_edges: glt.InputEdges, check_fn):
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

    print(f'[Client {client_rank}] Creating DistLinkNeighborLoader ...')
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
    dist_loader = glt.distributed.DistLinkNeighborLoader(
      data=None,
      num_neighbors=[2, 1],
      edge_label_index=input_edges,
      neg_sampling=neg_sampling,
      batch_size=5,
      shuffle=True,
      drop_last=False,
      with_edge=True,
      edge_dir='out',
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


class DistLinkNeighborLoaderTestCase(unittest.TestCase):
  def setUp(self):
    self.dataset0 = _prepare_dataset(rank=0)
    self.dataset1 = _prepare_dataset(rank=1)
    self.range_partition_dataset0 = _prepare_dataset(rank=0, is_range_partition=True)
    self.range_partition_dataset1 = _prepare_dataset(rank=1, is_range_partition=True)
    self.input_edges0 = torch.stack(
      (torch.arange(vnum_per_partition), torch.arange(vnum_per_partition)+1)
    ).to(dtype=torch.long)
    self.input_edges1 = torch.stack(
      (torch.arange(vnum_per_partition)+vnum_per_partition,
      (torch.arange(vnum_per_partition)+vnum_per_partition+1)%vnum_total)
    ).to(dtype=torch.long)
    self.out_hetero_dataset0 = _prepare_hetero_dataset(rank=0, edge_dir='out')
    self.out_hetero_dataset1 = _prepare_hetero_dataset(rank=1, edge_dir='out')
    self.in_hetero_dataset0 = _prepare_hetero_dataset(rank=0, edge_dir='in')
    self.in_hetero_dataset1 = _prepare_hetero_dataset(rank=1, edge_dir='in')
    self.hetero_input_edges0 = (u2i_etype, self.input_edges0)
    self.hetero_input_edges1 = (u2i_etype, self.input_edges1)
    self.bin_neg_sampling = glt.sampler.NegativeSampling('binary')
    self.tri_neg_sampling = glt.sampler.NegativeSampling('triplet', amount=2)

    self.master_port = glt.utils.get_free_port()
    self.sampling_master_port = glt.utils.get_free_port()

  def _get_homo_datasets(self, is_range_partition):
    return (self.range_partition_dataset0, self.range_partition_dataset1) if is_range_partition else (self.dataset0, self.dataset1)

  @parameterized.expand([
    (True),
    (False),
  ])
  def test_homo_out_sample_collocated(self, is_range_partition):
    print("\n--- DistLinkNeighborLoader Test (homogeneous, collocated) ---")
    dataset0, dataset1 = self._get_homo_datasets(is_range_partition)

    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            dataset0, self.bin_neg_sampling, self.input_edges0, _check_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            dataset1, self.bin_neg_sampling, self.input_edges1, _check_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()
    
  @parameterized.expand([
    (True),
    (False),
  ])
  def test_homo_out_sample_mp(self, is_range_partition):
    print("\n--- DistLinkNeighborLoader Test (homogeneous, multiprocessing) ---")
    dataset0, dataset1 = self._get_homo_datasets(is_range_partition)

    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            dataset0, self.tri_neg_sampling, self.input_edges0,
            _check_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            dataset1, self.tri_neg_sampling, self.input_edges1,
            _check_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_out_sample_collocated(self):
    print("\n--- DistLinkNeighborLoader Test (heterogeneous, collocated) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset0, self.tri_neg_sampling, self.hetero_input_edges0,
            _check_hetero_sample_result, True)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset1, self.tri_neg_sampling, self.hetero_input_edges1,
            _check_hetero_sample_result, True)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_out_sample_mp(self):
    print("\n--- DistLinkNeighborLoader Test (heterogeneous, multiprocessing) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset0, self.bin_neg_sampling, self.hetero_input_edges0,
            _check_hetero_sample_result, False)
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.out_hetero_dataset1, self.bin_neg_sampling, self.hetero_input_edges1,
            _check_hetero_sample_result, False)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_in_sample_collocated(self):
    print("\n--- DistLinkNeighborLoader Test (in-sample, heterogeneous, collocated) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset0, self.tri_neg_sampling, self.hetero_input_edges0,
            _check_hetero_sample_result, True, 'in')
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset1, self.tri_neg_sampling, self.hetero_input_edges1,
            _check_hetero_sample_result, True, 'in')
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_hetero_in_sample_mp(self):
    print("\n--- DistLinkNeighborLoader Test (in-sample, heterogeneous, multiprocessing) ---")
    mp_context = torch.multiprocessing.get_context('spawn')
    w0 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 0, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset0, self.bin_neg_sampling, self.hetero_input_edges0,
            _check_hetero_sample_result, False, 'in')
    )
    w1 = mp_context.Process(
      target=run_test_as_worker,
      args=(2, 1, self.master_port, self.sampling_master_port,
            self.in_hetero_dataset1, self.bin_neg_sampling, self.hetero_input_edges1,
            _check_hetero_sample_result, False, 'in')
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()

  def test_remote_mode(self):
    print("\n--- DistLinkNeighborLoader Test (server-client mode, remote) ---")
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
            self.bin_neg_sampling, self.input_edges0, _check_sample_result)
    )
    client1 = mp_context.Process(
      target=run_test_as_client,
      args=(2, 2, 1, self.master_port, self.sampling_master_port,
            self.bin_neg_sampling, self.input_edges1, _check_sample_result)
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
