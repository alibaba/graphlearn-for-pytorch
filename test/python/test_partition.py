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
import unittest

import torch

from graphlearn_torch.typing import *
from graphlearn_torch.partition import *


class PartitionTestCase(unittest.TestCase):
  def _create_edge_index(self, src_num, dst_num, degree):
    rows = []
    cols = []
    for v in range(src_num):
      rows.extend([v for _ in range(degree)])
      cols.extend([((v + i + 1) % dst_num) for i in range(degree)])
    return torch.tensor([rows, cols], dtype=torch.int64)
  
  def _check_dir_and_del(self, dir):
    if os.path.exists(dir):
      try:
        shutil.rmtree(dir)
      except OSError as e:
          print(f'Error when deleting {dir}: {e}')
  
  def test_random_homo_partition(self):
    dir = 'random_homo_partition_ut'
    self._check_dir_and_del(dir)
    nparts = 4

    node_num = 20
    edge_index = self._create_edge_index(node_num, node_num, 2)
    edge_num = len(edge_index[0])
    node_feat = torch.stack(
      [torch.ones(10) * i for i in range(node_num)], dim=0
    )
    edge_feat = torch.stack(
      [torch.ones(2) * i for i in range(edge_num)], dim=0
    )

    edge_weights = torch.arange(0, edge_num, dtype=torch.float)
    random_partitioner = RandomPartitioner(
      dir, nparts, node_num, edge_index, node_feat=node_feat,
      edge_feat=edge_feat, edge_weights=edge_weights, chunk_size=3)
    random_partitioner.partition()

    for pidx in range(nparts):
      _, _, p_graph, p_node_feat, p_edge_feat, node_pb, edge_pb = \
        load_partition(dir, pidx)

      # node
      node_ids = torch.unique(p_graph.edge_index[0])
      self.assertEqual(node_ids.size(0), 5)
      self.assertTrue(torch.equal(torch.sort(node_ids)[0],
                                  torch.sort(p_node_feat.ids)[0]))

      expect_node_pids = torch.ones(5, dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(node_pb[node_ids], expect_node_pids))

      self.assertEqual(p_node_feat.feats.size(0), 5)
      self.assertEqual(p_node_feat.feats.size(1), 10)
      self.assertEqual(p_node_feat.ids.size(0), 5)
      self.assertTrue(p_node_feat.cache_feats is None)
      self.assertTrue(p_node_feat.cache_ids is None)
      for idx, n_id in enumerate(p_node_feat.ids):
        self.assertTrue(torch.equal(p_node_feat.feats[idx], node_feat[n_id]))

      # edge & weight
      edge_ids = p_graph.eids
      edge_weight = p_graph.weights
      
      self.assertEqual(edge_ids.size(0), 10)
      self.assertTrue(torch.equal(torch.sort(edge_ids)[0],
                                  torch.sort(p_edge_feat.ids)[0]))
      self.assertTrue(torch.allclose(edge_ids.to(torch.float), edge_weight))

      expect_edge_pids = torch.ones(10, dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(edge_pb[edge_ids], expect_edge_pids))

      self.assertEqual(p_edge_feat.feats.size(0), 10)
      self.assertEqual(p_edge_feat.feats.size(1), 2)
      self.assertEqual(p_edge_feat.ids.size(0), 10)
      self.assertTrue(p_edge_feat.cache_feats is None)
      self.assertTrue(p_edge_feat.cache_ids is None)
      for idx, e_id in enumerate(p_edge_feat.ids):
        self.assertTrue(torch.equal(p_edge_feat.feats[idx], edge_feat[e_id]))

    shutil.rmtree(dir)

  def test_random_hetero_partition(self):
    dir = 'random_hetero_partition_ut'
    self._check_dir_and_del(dir)
    nparts = 4

    user_num = 20
    item_num = 12
    node_num_dict = {'user': user_num, 'item': item_num}
    u2i_type = ('user', 'u2i', 'item')
    i2i_type = ('item', 'i2i', 'item')

    edge_index_dict = {
      u2i_type: self._create_edge_index(user_num, item_num, 2),
      i2i_type: self._create_edge_index(item_num, item_num, 2)
    }
    u2i_num = len(edge_index_dict[u2i_type][0])
    i2i_num = len(edge_index_dict[i2i_type][0])

    user_feats = [torch.ones(10, dtype=torch.float) * i
                  for i in range(user_num)]
    item_feats = [torch.ones(10, dtype=torch.float) * 2 * i
                  for i in range(item_num)]
    node_feat_dict = {
      'user': torch.stack(user_feats, dim=0),
      'item': torch.stack(item_feats, dim=0)
    }

    u2i_feats = [torch.ones(2, dtype=torch.float) * i
                 for i in range(u2i_num)]
    i2i_feats = [torch.ones(2, dtype=torch.float) * 2 * i
                 for i in range(i2i_num)]
    edge_feat_dict = {
      u2i_type: torch.stack(u2i_feats, dim=0),
      i2i_type: torch.stack(i2i_feats, dim=0)
    }

    u2i_weights = torch.arange(0, edge_index_dict[u2i_type].size(1), dtype=torch.float)
    i2i_weights = torch.arange(0, edge_index_dict[i2i_type].size(1), dtype=torch.float)
    edge_weight_dict = {
      u2i_type: u2i_weights,
      i2i_type: i2i_weights
    }

    random_partitioner = RandomPartitioner(
      dir, nparts, node_num_dict, edge_index_dict, node_feat=node_feat_dict,
      edge_feat=edge_feat_dict, edge_weights=edge_weight_dict, chunk_size=3
    )
    random_partitioner.partition()

    for pidx in range(nparts):
      (
        _, _,
        p_graph_dict, p_node_feat_dict, p_edge_feat_dict,
        node_pb_dict, edge_pb_dict
      ) = load_partition(dir, pidx)

      # user
      user_ids = torch.unique(p_graph_dict[u2i_type].edge_index[0])
      self.assertEqual(user_ids.size(0), 5)
      self.assertTrue(torch.equal(torch.sort(user_ids)[0],
                                  torch.sort(p_node_feat_dict['user'].ids)[0]))

      expect_user_pids = torch.ones(5, dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(node_pb_dict['user'][user_ids],
                                  expect_user_pids))

      self.assertEqual(p_node_feat_dict['user'].feats.size(0), 5)
      self.assertEqual(p_node_feat_dict['user'].feats.size(1), 10)
      self.assertEqual(p_node_feat_dict['user'].ids.size(0), 5)
      self.assertTrue(p_node_feat_dict['user'].cache_feats is None)
      self.assertTrue(p_node_feat_dict['user'].cache_ids is None)
      for idx, user_id in enumerate(p_node_feat_dict['user'].ids):
        self.assertTrue(torch.equal(p_node_feat_dict['user'].feats[idx],
                                    node_feat_dict['user'][user_id]))

      # item
      item_ids = torch.unique(p_graph_dict[i2i_type].edge_index[0])
      self.assertEqual(item_ids.size(0), 3)
      self.assertTrue(torch.equal(torch.sort(item_ids)[0],
                                  torch.sort(p_node_feat_dict['item'].ids)[0]))

      expect_item_pids = torch.ones(3, dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(node_pb_dict['item'][item_ids],
                                  expect_item_pids))

      self.assertEqual(p_node_feat_dict['item'].feats.size(0), 3)
      self.assertEqual(p_node_feat_dict['item'].feats.size(1), 10)
      self.assertEqual(p_node_feat_dict['item'].ids.size(0), 3)
      self.assertTrue(p_node_feat_dict['item'].cache_feats is None)
      self.assertTrue(p_node_feat_dict['item'].cache_ids is None)
      for idx, item_id in enumerate(p_node_feat_dict['item'].ids):
        self.assertTrue(torch.equal(p_node_feat_dict['item'].feats[idx],
                                    node_feat_dict['item'][item_id]))

      # u2i
      p_u2i_eids = p_graph_dict[u2i_type].eids
      p_u2i_weights = p_graph_dict[u2i_type].weights
      
      self.assertEqual(p_u2i_eids.size(0), 10)
      self.assertTrue(torch.allclose(p_u2i_eids.to(torch.float), p_u2i_weights))

      expect_u2i_pids = torch.ones(10, dtype=torch.long) * pidx
      self.assertTrue(torch.equal(edge_pb_dict[u2i_type][p_u2i_eids],
                                  expect_u2i_pids))

      self.assertEqual(p_edge_feat_dict[u2i_type].feats.size(0), 10)
      self.assertEqual(p_edge_feat_dict[u2i_type].feats.size(1), 2)
      self.assertEqual(p_edge_feat_dict[u2i_type].ids.size(0), 10)
      self.assertTrue(p_edge_feat_dict[u2i_type].cache_feats is None)
      self.assertTrue(p_edge_feat_dict[u2i_type].cache_ids is None)
      for idx, e_id in enumerate(p_edge_feat_dict[u2i_type].ids):
        self.assertTrue(torch.equal(p_edge_feat_dict[u2i_type].feats[idx],
                                    edge_feat_dict[u2i_type][e_id]))

      # i2i
      p_i2i_eids = p_graph_dict[i2i_type].eids
      p_i2i_weights = p_graph_dict[i2i_type].weights

      self.assertEqual(p_i2i_eids.size(0), 6)
      self.assertTrue(torch.allclose(p_i2i_eids.to(torch.float), p_i2i_weights))

      expect_i2i_pids = torch.ones(6, dtype=torch.long) * pidx
      self.assertTrue(torch.equal(edge_pb_dict[i2i_type][p_i2i_eids],
                                  expect_i2i_pids))

      self.assertEqual(p_edge_feat_dict[i2i_type].feats.size(0), 6)
      self.assertEqual(p_edge_feat_dict[i2i_type].feats.size(1), 2)
      self.assertEqual(p_edge_feat_dict[i2i_type].ids.size(0), 6)
      self.assertTrue(p_edge_feat_dict[i2i_type].cache_feats is None)
      self.assertTrue(p_edge_feat_dict[i2i_type].cache_ids is None)
      for idx, e_id in enumerate(p_edge_feat_dict[i2i_type].ids):
        self.assertTrue(torch.equal(p_edge_feat_dict[i2i_type].feats[idx],
                                    edge_feat_dict[i2i_type][e_id]))

    shutil.rmtree(dir)

  def test_frequency_partition(self):
    dir = 'frequency_partition_ut'
    self._check_dir_and_del(dir)
    nparts = 4

    node_num = 20
    edge_index = self._create_edge_index(node_num, node_num, 2)
    edge_num = len(edge_index[0])

    node_feat = torch.stack(
      [torch.ones(10) * i for i in range(node_num)], dim=0
    )
    node_probs = [torch.rand(node_num) for _ in range(nparts)]
    cache_budget_bytes = 4 * node_feat.size(1) * node_feat.element_size()

    edge_feat = torch.stack(
      [torch.ones(2) * i for i in range(edge_num)], dim=0
    )

    edge_weights = torch.arange(0, edge_num, dtype=torch.float)
    freq_partitioner = FrequencyPartitioner(
      dir, nparts, node_num, edge_index, node_probs,
      node_feat=node_feat, edge_feat=edge_feat, edge_weights=edge_weights,
      cache_memory_budget=cache_budget_bytes,
      chunk_size=3)
    freq_partitioner.partition()

    all_node_ids = []
    all_edge_ids = []
    for pidx in range(nparts):
      _, _, p_graph, p_node_feat, p_edge_feat, node_pb, edge_pb = \
        load_partition(dir, pidx)

      node_ids = torch.unique(p_graph.edge_index[0])
      self.assertTrue(torch.equal(torch.sort(node_ids)[0],
                                  torch.sort(p_node_feat.ids)[0]))
      all_node_ids.append(node_ids)

      expect_node_pids = torch.ones(node_ids.size(0), dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(node_pb[node_ids], expect_node_pids))

      self.assertTrue(p_node_feat.cache_feats is not None)
      self.assertTrue(p_node_feat.cache_ids is not None)
      for idx in range(p_node_feat.cache_ids.size(0) - 1):
        self.assertGreaterEqual(
          node_probs[pidx][p_node_feat.cache_ids[idx]].item(),
          node_probs[pidx][p_node_feat.cache_ids[idx + 1]].item()
        )

      # edge & weight
      edge_ids = p_graph.eids
      edge_weights = p_graph.weights

      self.assertTrue(torch.equal(torch.sort(edge_ids)[0],
                                  torch.sort(p_edge_feat.ids)[0]))
      self.assertTrue(torch.allclose(edge_ids.to(torch.float), edge_weights))

      all_edge_ids.append(edge_ids)

      expect_edge_pids = torch.ones(edge_ids.size(0), dtype=torch.int64) * pidx
      self.assertTrue(torch.equal(edge_pb[edge_ids], expect_edge_pids))

      self.assertTrue(p_edge_feat.cache_feats is None)
      self.assertTrue(p_edge_feat.cache_ids is None)

    all_node_ids = torch.cat(all_node_ids)
    self.assertTrue(torch.equal(torch.sort(all_node_ids)[0],
                                torch.arange(node_num)))
    all_edge_ids = torch.cat(all_edge_ids)
    self.assertTrue(torch.equal(torch.sort(all_edge_ids)[0],
                                torch.arange(edge_num)))

    shutil.rmtree(dir)

  def test_cat_feature_cache(self):
    feat_pdata = FeaturePartitionData(
      feats=torch.rand(4, 10),
      ids=torch.tensor([0, 2, 4, 6], dtype=torch.long),
      cache_feats=torch.rand(2, 10),
      cache_ids=torch.tensor([3, 4], dtype=torch.long)
    )
    feat_pb = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

    res = cat_feature_cache(0, feat_pdata, feat_pb)
    cache_ratio, new_feats, id2idx, new_feat_pb = res
    self.assertEqual(cache_ratio, 2 / 6)
    self.assertEqual(new_feats.size(0), 6)
    self.assertEqual(id2idx[3].item(), 0)
    self.assertEqual(id2idx[4].item(), 1)
    self.assertEqual(new_feat_pb[3].item(), 0)


if __name__ == '__main__':
  unittest.main()
