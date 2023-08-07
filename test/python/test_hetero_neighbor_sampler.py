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


class RandomSamplerTestCase(unittest.TestCase):
  def setUp(self):
    # options for dataset generation
    self.vnum_total = 40
    self.degree = 2
    self.enum_total = 80

    # for hetero dataset
    self.user_ntype = 'user'
    self.item_ntype = 'item'
    self.u2i_etype = ('user', 'u2i', 'item')
    self.i2i_etype = ('item', 'i2i', 'item')
    self.rev_u2i_etype = ('item', 'rev_u2i', 'user')

    # graph
    user_nodes, u2i_rows, u2i_cols, u2i_eids = [], [], [], []
    for v in range(self.vnum_total):
      user_nodes.append(v)
      u2i_rows.extend([v for _ in range(self.degree)])
      u2i_cols.extend([((v + i + 1) % self.vnum_total) for i in range(self.degree)])
      u2i_eids.extend([(v * self.degree + i) for i in range(self.degree)])
    u2i_edge_index = torch.tensor([u2i_rows, u2i_cols], dtype=torch.int64)

    u2i_edge_ids = torch.tensor(u2i_eids, dtype=torch.int64)
    u2i_edge_weights = (u2i_edge_ids % 2).to(torch.float) + .5
    u2i_csr_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids)
    u2i_csc_topo = glt.data.Topology(
      edge_index=u2i_edge_index, edge_ids=u2i_edge_ids, layout='CSC')
    u2i_weighted_csr_topo = glt.data.Topology(
      edge_index=u2i_edge_index,
      edge_ids=u2i_edge_ids, edge_weights=u2i_edge_weights)
    u2i_graph = glt.data.Graph(u2i_csr_topo, 'ZERO_COPY', device=0)
    u2i_in_graph = glt.data.Graph(u2i_csc_topo, 'ZERO_COPY', device=0)
    u2i_weighted_graph = glt.data.Graph(u2i_weighted_csr_topo, 'CPU')

    item_nodes, i2i_rows, i2i_cols, i2i_eids = [], [], [], []
    for v in range(self.vnum_total):
      item_nodes.append(v)
      i2i_rows.extend([v for _ in range(self.degree)])
      i2i_cols.extend([((v + i + 2) % self.vnum_total) for i in range(self.degree)])
      i2i_eids.extend([(v * self.degree + i) for i in range(self.degree)])
    i2i_edge_index = torch.tensor([i2i_rows, i2i_cols], dtype=torch.int64)

    i2i_edge_ids = torch.tensor(i2i_eids, dtype=torch.int64)
    i2i_edge_weights = (i2i_edge_ids  % 2).to(torch.float) + .5
    i2i_csr_topo = glt.data.Topology(edge_index=i2i_edge_index, edge_ids=i2i_edge_ids)
    i2i_csc_topo = glt.data.Topology(
      edge_index=i2i_edge_index, edge_ids=i2i_edge_ids, layout='CSC'
    )
    i2i_weighted_csr_topo = glt.data.Topology(
      edge_index=i2i_edge_index,
      edge_ids=i2i_edge_ids, edge_weights=i2i_edge_weights)
    i2i_graph = glt.data.Graph(i2i_csr_topo, 'ZERO_COPY', device=0)
    i2i_in_graph = glt.data.Graph(i2i_csc_topo, 'ZERO_COPY', device=0)
    i2i_weighted_graph = glt.data.Graph(i2i_weighted_csr_topo, 'CPU')

    self.graph_dict = {
      self.u2i_etype: u2i_graph,
      self.i2i_etype: i2i_graph
    }

    self.graph_in_dict = {
      self.u2i_etype: u2i_in_graph,
      self.i2i_etype: i2i_in_graph
    }

    self.weighted_graph_dict = {
      self.u2i_etype: u2i_weighted_graph,
      self.i2i_etype: i2i_weighted_graph
    }

    # feature
    device_group_list = [glt.data.DeviceGroup(0, [0])]
    split_ratio = 0.2

    user_nfeat = torch.zeros(len(user_nodes), 512, dtype=torch.float32)
    user_nfeat_id2idx = glt.utils.id2idx(user_nodes)
    user_feature = glt.data.Feature(user_nfeat, user_nfeat_id2idx,
                                    split_ratio, device_group_list, device=0)

    item_nfeat = torch.ones(len(item_nodes), 256, dtype=torch.float32) + 1
    item_nfeat_id2idx = glt.utils.id2idx(item_nodes)
    item_feature = glt.data.Feature(item_nfeat, item_nfeat_id2idx,
                                    split_ratio, device_group_list, device=0)

    self.node_feature_dict = {
      self.user_ntype: user_feature,
      self.item_ntype: item_feature
    }

    u2i_efeat = torch.ones(len(u2i_eids), 10, dtype=torch.float32) + 1
    u2i_efeat_id2idx = glt.utils.id2idx(u2i_eids)
    u2i_feature = glt.data.Feature(u2i_efeat, u2i_efeat_id2idx,
                                   split_ratio, device_group_list, device=0)

    i2i_efeat = torch.ones(len(i2i_eids), 5, dtype=torch.float32) + 3
    i2i_efeat_id2idx = glt.utils.id2idx(i2i_eids)
    i2i_feature = glt.data.Feature(i2i_efeat, i2i_efeat_id2idx,
                                   split_ratio, device_group_list, device=0)

    self.edge_feature_dict = {
      self.u2i_etype: u2i_feature,
      self.i2i_etype: i2i_feature
    }

    # node label
    self.node_label_dict = {
      self.user_ntype: torch.arange(self.vnum_total),
      self.item_ntype: torch.arange(self.vnum_total)
    }


  def test_hetero_sample_from_nodes(self):
    node_sampler = glt.sampler.NeighborSampler(
      graph=self.graph_dict,
      num_neighbors=[2,1],
      with_edge=True,
      edge_dir='out',
    )
    sampler_input = glt.sampler.NodeSamplerInput(
      node=torch.tensor([1,5,9,13,17,21,25,29]), input_type=self.user_ntype)
    sample_out = node_sampler.sample_from_nodes(sampler_input)

    base_homo_edge_index = torch.stack((
      sample_out.node['item'][sample_out.row[self.i2i_etype]],
      sample_out.node['item'][sample_out.col[self.i2i_etype]]
    ))
    base_hetero_edge_index = torch.stack((
      sample_out.node['item'][sample_out.row[self.rev_u2i_etype]],
      sample_out.node['user'][sample_out.col[self.rev_u2i_etype]]
    ))
    self.assertTrue(torch.all(
      ((base_homo_edge_index[1]+2)%40==base_homo_edge_index[0]) +
      ((base_homo_edge_index[1]+3)%40==base_homo_edge_index[0])
    ))
    self.assertTrue(torch.all(
      ((base_hetero_edge_index[1]+1)%40==base_hetero_edge_index[0]) +
      ((base_hetero_edge_index[1]+2)%40==base_hetero_edge_index[0])
    ))

    base_edge_ids = torch.cat(
       (sample_out.node['user'] * 2, sample_out.node['user'] * 2 + 1)
    ).unique()
    self.assertTrue(glt.utils.tensor_equal_with_device(
       base_edge_ids, sample_out.edge[self.rev_u2i_etype].unique())
    )
  
  def test_weighted_hetero_sample_from_nodes(self):
    node_out_sampler = glt.sampler.NeighborSampler(
      graph=self.weighted_graph_dict,
      device='CPU',
      num_neighbors=[1],
      with_edge=True,
      with_weight=True,
      edge_dir='out',
    )
    sampler_out_input = glt.sampler.NodeSamplerInput(
      node=torch.tensor([1,5,9,13,21,29,37,38]), input_type=self.user_ntype)
    stats = torch.zeros(80)
    for _ in range(1000):
      sample_out = node_out_sampler.sample_from_nodes(
        sampler_out_input, device=torch.device('cpu'))
      edges = sample_out.edge[('item', 'rev_u2i', 'user')]
      stats.scatter_add_(0, edges, torch.ones(80))

    # with high probability
    self.assertTrue(stats[2] < 350 and stats[10] < 350 and stats[18] < 350 and \
                    stats[26] < 350 and stats[42] < 350 and stats[58] < 350 \
                    and stats[74] < 350 and stats[76] < 350)
    self.assertTrue(stats[3] > 650 and stats[11] > 650 and stats[19] > 650 and \
                    stats[27] > 650 and stats[43] > 650 and stats[59] > 650 \
                    and stats[75] > 650 and stats[77] > 650)
    self.assertEqual(sum(stats), 8000)

  def test_hetero_insample_from_items(self):
    node_sampler = glt.sampler.NeighborSampler(
      graph=self.graph_in_dict,
      num_neighbors=[1],
      with_edge=True,
      edge_dir='in'
    )
    # sample from 'user' can't get other nodes
    sampler_input = glt.sampler.NodeSamplerInput(
      node=torch.tensor([1,5,9,13,17,21,25,29]), input_type=self.user_ntype)
    sample_out = node_sampler.sample_from_nodes(sampler_input)
    self.assertTrue(len(sample_out.num_sampled_edges) == 0 and \
                    sample_out.node['user'].numel() == 8)

    # sampler from 'item', we can get 'user' and 'item'
    sampler_input = glt.sampler.NodeSamplerInput(
      node=torch.tensor([1,5,9,13,17,21,25,29]), input_type=self.item_ntype)
    sample_out = node_sampler.sample_from_nodes(sampler_input)

    base_homo_edge_index = torch.stack((
      sample_out.node['item'][sample_out.row[self.i2i_etype]],
      sample_out.node['item'][sample_out.col[self.i2i_etype]]
    ))
    base_hetero_edge_index = torch.stack((
      sample_out.node['user'][sample_out.row[self.u2i_etype]],
      sample_out.node['item'][sample_out.col[self.u2i_etype]]  
    ))
    
    self.assertTrue(torch.all(
      ((base_homo_edge_index[0]+2)%40==base_homo_edge_index[1]) +
      ((base_homo_edge_index[0]+3)%40==base_homo_edge_index[1])
    ))
    self.assertTrue(torch.all(
      ((base_hetero_edge_index[0]+1)%40==base_hetero_edge_index[1]) +
      ((base_hetero_edge_index[0]+2)%40==base_hetero_edge_index[1])
    ))


  def test_hetero_sample_from_edges(self):
    edge_sampler = glt.sampler.NeighborSampler(
      graph=self.graph_dict,
      num_neighbors=[2,1],
      with_edge=True,
      with_neg=True
    )
    bin_neg_sampling = glt.sampler.NegativeSampling(mode='binary')
    tri_neg_sampling = glt.sampler.NegativeSampling(mode='triplet', amount=2)
    bin_sampler_input = glt.sampler.EdgeSamplerInput(
      row=torch.tensor([1, 3, 4, 7, 12, 18, 27, 32, 38], device=0),
      col=torch.tensor([2, 5, 5, 8, 13, 20, 29, 33, 0], device=0),
      input_type=self.u2i_etype,
      neg_sampling=bin_neg_sampling
    )
    tri_sampler_input = glt.sampler.EdgeSamplerInput(
      row=torch.tensor([1, 3, 4, 7, 12, 18, 27, 32, 38], device=0),
      col=torch.tensor([2, 5, 5, 8, 13, 20, 29, 33, 0], device=0),
      input_type=self.u2i_etype,
      neg_sampling=tri_neg_sampling
    )
    
    # check binary cases
    bin_sampler_out = edge_sampler.sample_from_edges(bin_sampler_input)
    base_homo_edge_index = torch.stack((
      bin_sampler_out.node['item'][bin_sampler_out.row[self.i2i_etype]],
      bin_sampler_out.node['item'][bin_sampler_out.col[self.i2i_etype]]
    ))
    base_hetero_edge_index = torch.stack((
      bin_sampler_out.node['item'][bin_sampler_out.row[self.rev_u2i_etype]],
      bin_sampler_out.node['user'][bin_sampler_out.col[self.rev_u2i_etype]]
    ))
    self.assertTrue(torch.all(
      ((base_homo_edge_index[1]+2)%40==base_homo_edge_index[0]) +
      ((base_homo_edge_index[1]+3)%40==base_homo_edge_index[0])
    ))
    self.assertTrue(torch.all(
      ((base_hetero_edge_index[1]+1)%40==base_hetero_edge_index[0]) +
      ((base_hetero_edge_index[1]+2)%40==base_hetero_edge_index[0])
    ))

    base_edge_ids = torch.cat(
      (bin_sampler_out.node['user'] * 2, bin_sampler_out.node['user'] * 2 + 1)
    ).unique()
    self.assertTrue(glt.utils.tensor_equal_with_device(
      base_edge_ids, bin_sampler_out.edge[self.rev_u2i_etype].unique())
    )
    
    self.assertTrue(glt.utils.tensor_equal_with_device(
      bin_sampler_out.metadata['edge_label'][:9],
      torch.ones(9, dtype=torch.float, device=0))
    )
    self.assertTrue(glt.utils.tensor_equal_with_device(
      bin_sampler_out.metadata['edge_label'][9:],
      torch.zeros(9, dtype=torch.float, device=0))
    )

    base_edge_label_index = torch.stack((
      bin_sampler_input.row, bin_sampler_input.col
    ))
    pos_index = torch.stack((
      bin_sampler_out.node['user'][bin_sampler_out.metadata['edge_label_index'][0,:9]],
      bin_sampler_out.node['item'][bin_sampler_out.metadata['edge_label_index'][1,:9]]
    ))
    self.assertTrue(glt.utils.tensor_equal_with_device(
      base_edge_label_index, pos_index
    ))
    neg_index = torch.stack((
      bin_sampler_out.node['user'][bin_sampler_out.metadata['edge_label_index'][0,9:]],
      bin_sampler_out.node['item'][bin_sampler_out.metadata['edge_label_index'][1,9:]]
    ))
    self.assertFalse(torch.any(
      ((neg_index[0]+1)%40==neg_index[1]) + ((neg_index[0]+2)%40==neg_index[1])
    ))
    
    # check triplet cases
    tri_sampler_out = edge_sampler.sample_from_edges(tri_sampler_input)
    base_homo_edge_index = torch.stack((
      tri_sampler_out.node['item'][tri_sampler_out.row[self.i2i_etype]],
      tri_sampler_out.node['item'][tri_sampler_out.col[self.i2i_etype]]
    ))
    base_hetero_edge_index = torch.stack((
      tri_sampler_out.node['item'][tri_sampler_out.row[self.rev_u2i_etype]],
      tri_sampler_out.node['user'][tri_sampler_out.col[self.rev_u2i_etype]]
    ))
    self.assertTrue(torch.all(
      ((base_homo_edge_index[1]+2)%40==base_homo_edge_index[0]) +
      ((base_homo_edge_index[1]+3)%40==base_homo_edge_index[0])
    ))
    self.assertTrue(torch.all(
      ((base_hetero_edge_index[1]+1)%40==base_hetero_edge_index[0]) +
      ((base_hetero_edge_index[1]+2)%40==base_hetero_edge_index[0])
    ))

    base_edge_ids = torch.cat(
      (tri_sampler_out.node['user'] * 2, tri_sampler_out.node['user'] * 2 + 1)
    ).unique()
    self.assertTrue(glt.utils.tensor_equal_with_device(
      base_edge_ids, tri_sampler_out.edge[self.rev_u2i_etype].unique())
    )

    base_src_index = tri_sampler_out.node['user'][
      tri_sampler_out.metadata['src_index']]
    base_dst_index = tri_sampler_out.node['item'][
      tri_sampler_out.metadata['dst_pos_index']]
    self.assertTrue(glt.utils.tensor_equal_with_device(
      base_src_index, tri_sampler_input.row
    ))
    self.assertTrue(glt.utils.tensor_equal_with_device(
      base_dst_index, tri_sampler_input.col
    ))
    self.assertEqual(
      tri_sampler_out.metadata['dst_neg_index'].size(), torch.Size([9, 2])
    )


if __name__ == "__main__":
  unittest.main()
