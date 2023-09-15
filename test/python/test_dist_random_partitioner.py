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
import shutil
import time
import unittest

import torch
import torch.multiprocessing as mp
import graphlearn_torch as glt


# for hetero dataset
user_ntype = 'user'
item_ntype = 'item'
u2i_etype = ('user', 'u2i', 'item')
i2i_etype = ('item', 'i2i', 'item')

def _get_part_of_node(rank, world_size, num_nodes, nfeat_num=10):
  per_num_nodes = num_nodes // world_size
  nids, nfeat = [], []
  for v in range(per_num_nodes * rank, min(num_nodes, per_num_nodes * (rank + 1))):
    nids.append(v)
    nfeat.append([v for _ in range(nfeat_num)])
  node_ids = torch.tensor(nids, dtype=torch.int64)
  node_feat = torch.tensor(nfeat, dtype=torch.float32)
  return node_ids, node_feat

def _get_part_of_edge(rank, world_size, num_src, num_dst, degree=2, efeat_num=5):
  per_num_src = num_src // world_size
  rows, cols, eids, efeat = [], [], [], []
  for v in range(per_num_src * rank, min(num_src, per_num_src * (rank + 1))):
    rows.extend([v for _ in range(degree)])
    cols.extend([((v + i + 1) % num_dst) for i in range(degree)])
    eids.extend([(v * degree + i) for i in range(degree)])
    efeat.extend([[(v * degree + i) for _ in range(efeat_num)] for i in range(degree)])
  edge_index = torch.tensor([rows, cols], dtype=torch.int64)
  edge_ids = torch.tensor(eids, dtype=torch.int64)
  edge_feat = torch.tensor(efeat, dtype=torch.float32)
  return edge_index, edge_ids, edge_feat

def _get_part_of_graph(rank, world_size, graph_type='homo'):
  if graph_type == 'homo':
    num_nodes = 100
    node_ids, node_feat = _get_part_of_node(rank, world_size, 100)
    edge_index, edge_ids, edge_feat = _get_part_of_edge(rank, world_size, 100, 100)
  else:
    num_nodes = {user_ntype: 100, item_ntype: 60}
    node_ids, node_feat = {}, {}
    node_ids[user_ntype], node_feat[user_ntype] = \
      _get_part_of_node(rank, world_size, 100)
    node_ids[item_ntype], node_feat[item_ntype] = \
      _get_part_of_node(rank, world_size, 60)
    edge_index, edge_ids, edge_feat = {}, {}, {}
    edge_index[u2i_etype], edge_ids[u2i_etype], edge_feat[u2i_etype] = \
      _get_part_of_edge(rank, world_size, 100, 60)
    edge_index[i2i_etype], edge_ids[i2i_etype], edge_feat[i2i_etype] = \
      _get_part_of_edge(rank, world_size, 60, 60)
  return num_nodes, node_ids, node_feat, edge_index, edge_ids, edge_feat

def _check_partition(dir, pidx, num_parts, graph_type='homo'):
  loaded_num_parts, _, graph, node_feat, edge_feat, node_pb, edge_pb = \
    glt.partition.load_partition(dir, pidx)
  tc = unittest.TestCase()
  tc.assertEqual(loaded_num_parts, num_parts)
  if graph_type == 'homo':
    node_ids = torch.unique(graph.edge_index[0])
    tc.assertEqual(node_ids.size(0), 50)
    tc.assertTrue(torch.equal(torch.sort(node_ids)[0],
                              torch.sort(node_feat.ids)[0]))
    expect_node_pidx = torch.ones(50, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(node_pb[node_ids], expect_node_pidx))
    tc.assertEqual(node_feat.feats.size(0), 50)
    tc.assertEqual(node_feat.feats.size(1), 10)
    tc.assertTrue(node_feat.cache_feats is None)
    tc.assertTrue(node_feat.cache_ids is None)
    for idx, n_id in enumerate(node_feat.ids):
      tc.assertTrue(torch.equal(node_feat.feats[idx],
                                torch.ones(10, dtype=torch.float32) * n_id))
    edge_ids = graph.eids
    tc.assertEqual(edge_ids.size(0), 50 * 2)
    tc.assertTrue(torch.equal(torch.sort(edge_ids)[0],
                              torch.sort(edge_feat.ids)[0]))
    expect_edge_pidx = torch.ones(50 * 2, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(edge_pb[edge_ids], expect_edge_pidx))

    tc.assertEqual(edge_feat.feats.size(0), 50 * 2)
    tc.assertEqual(edge_feat.feats.size(1), 5)
    tc.assertTrue(edge_feat.cache_feats is None)
    tc.assertTrue(edge_feat.cache_ids is None)
    for idx, e_id in enumerate(edge_feat.ids):
      tc.assertTrue(torch.equal(edge_feat.feats[idx],
                                torch.ones(5, dtype=torch.float32) * e_id))
  else:
    # user
    user_ids = torch.unique(graph[u2i_etype].edge_index[0])
    tc.assertEqual(user_ids.size(0), 50)
    tc.assertTrue(torch.equal(torch.sort(user_ids)[0],
                              torch.sort(node_feat[user_ntype].ids)[0]))
    expect_user_pidx = torch.ones(50, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(node_pb[user_ntype][user_ids], expect_user_pidx))
    tc.assertEqual(node_feat[user_ntype].feats.size(0), 50)
    tc.assertEqual(node_feat[user_ntype].feats.size(1), 10)
    tc.assertTrue(node_feat[user_ntype].cache_feats is None)
    tc.assertTrue(node_feat[user_ntype].cache_ids is None)
    for idx, user_id in enumerate(node_feat[user_ntype].ids):
      tc.assertTrue(torch.equal(node_feat[user_ntype].feats[idx],
                                torch.ones(10, dtype=torch.float32) * user_id))
    # item
    item_ids = torch.unique(graph[i2i_etype].edge_index[0])
    tc.assertEqual(item_ids.size(0), 30)
    tc.assertTrue(torch.equal(torch.sort(item_ids)[0],
                              torch.sort(node_feat[item_ntype].ids)[0]))
    expect_item_pidx = torch.ones(30, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(node_pb[item_ntype][item_ids], expect_item_pidx))
    tc.assertEqual(node_feat[item_ntype].feats.size(0), 30)
    tc.assertEqual(node_feat[item_ntype].feats.size(1), 10)
    tc.assertTrue(node_feat[item_ntype].cache_feats is None)
    tc.assertTrue(node_feat[item_ntype].cache_ids is None)
    for idx, item_id in enumerate(node_feat[item_ntype].ids):
      tc.assertTrue(torch.equal(node_feat[item_ntype].feats[idx],
                                torch.ones(10, dtype=torch.float32) * item_id))
    # u2i
    u2i_eids = graph[u2i_etype].eids
    tc.assertEqual(u2i_eids.size(0), 50 * 2)
    expect_u2i_pidx = torch.ones(50 * 2, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(edge_pb[u2i_etype][u2i_eids], expect_u2i_pidx))
    tc.assertEqual(edge_feat[u2i_etype].feats.size(0), 50 * 2)
    tc.assertEqual(edge_feat[u2i_etype].feats.size(1), 5)
    tc.assertTrue(edge_feat[u2i_etype].cache_feats is None)
    tc.assertTrue(edge_feat[u2i_etype].cache_ids is None)
    for idx, u2i_eid in enumerate(edge_feat[u2i_etype].ids):
      tc.assertTrue(torch.equal(edge_feat[u2i_etype].feats[idx],
                                torch.ones(5, dtype=torch.float32) * u2i_eid))
    # i2i
    i2i_eids = graph[i2i_etype].eids
    tc.assertEqual(i2i_eids.size(0), 30 * 2)
    expect_i2i_pidx = torch.ones(30 * 2, dtype=torch.int64) * pidx
    tc.assertTrue(torch.equal(edge_pb[i2i_etype][i2i_eids], expect_i2i_pidx))
    tc.assertEqual(edge_feat[i2i_etype].feats.size(0), 30 * 2)
    tc.assertEqual(edge_feat[i2i_etype].feats.size(1), 5)
    tc.assertTrue(edge_feat[i2i_etype].cache_feats is None)
    tc.assertTrue(edge_feat[i2i_etype].cache_ids is None)
    for idx, i2i_eid in enumerate(edge_feat[i2i_etype].ids):
      tc.assertTrue(torch.equal(edge_feat[i2i_etype].feats[idx],
                                torch.ones(5, dtype=torch.float32) * i2i_eid))


def run_dist_partitioner(rank, world_size, master_addr, master_port,
                         root_dir, graph_type='homo'):
  num_nodes, node_ids, node_feat, edge_index, edge_ids, edge_feat = \
    _get_part_of_graph(rank, world_size, graph_type)
  output_dir = os.path.join(root_dir, f'p{rank}')
  dist_partitioner = glt.distributed.DistRandomPartitioner(
    output_dir, num_nodes, edge_index, edge_ids, node_feat, node_ids,
    edge_feat, edge_ids, num_parts=world_size, current_partition_idx=rank,
    chunk_size=7, master_addr=master_addr, master_port=master_port,
    num_rpc_threads=4
  )
  dist_partitioner.partition()
  _check_partition(output_dir, rank, world_size, graph_type)


class DistRandomPartitionerTestCase(unittest.TestCase):
  def test_with_homo_graph(self):
    root_dir = 'dist_random_partitioner_ut_homo'
    master_addr = 'localhost'
    master_port = glt.utils.get_free_port(master_addr)
    time.sleep(1)
    n_partitioners = 2
    mp.spawn(
      run_dist_partitioner,
      args=(n_partitioners, master_addr, master_port, root_dir, 'homo'),
      nprocs=n_partitioners,
      join=True,
    )
    shutil.rmtree(root_dir)

  def test_with_hetero_graph(self):
    root_dir = 'dist_random_partitioner_ut_hetero'
    master_addr = 'localhost'
    master_port = glt.utils.get_free_port(master_addr)
    time.sleep(1)
    n_partitioners = 2
    mp.spawn(
      run_dist_partitioner,
      args=(n_partitioners, master_addr, master_port, root_dir, 'hetero'),
      nprocs=n_partitioners,
      join=True,
    )
    shutil.rmtree(root_dir)


if __name__ == "__main__":
  unittest.main()
