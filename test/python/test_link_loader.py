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

import unittest
import torch

from torch_geometric.data import Data, HeteroData

from graphlearn_torch.data import Dataset, DeviceGroup
from graphlearn_torch.sampler import NegativeSampling
from graphlearn_torch.loader import LinkNeighborLoader


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)

def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


class LinkLoaderTestCase(unittest.TestCase):
  def setUp(self) -> None:
    self.bin_neg_sampling = NegativeSampling('binary')
    self.tri_neg_sampling = NegativeSampling('triplet', amount=3)

  def test_homo_link_neighbor_loader(self):
    edge_label_index = get_edge_index(100, 50, 500)

    data = Data()

    data.edge_index = edge_label_index
    data.x = torch.arange(100)
    data.edge_attr = torch.arange(500)

    dataset = Dataset()
    dataset.init_graph(
      edge_index=data.edge_index,
      graph_mode='ZERO_COPY',
      directed=False
    )
    dataset.init_node_features(
      node_feature_data=data.x,
      split_ratio=0.2,
      device_group_list=[DeviceGroup(0, [0])],
    )
    dataset.init_edge_features(
      edge_feature_data=data.edge_attr,
      device_group_list=[DeviceGroup(0, [0])],
      device=0)

    loader = LinkNeighborLoader(
      dataset,
      num_neighbors=[3] * 2,
      batch_size=20,
      edge_label_index=edge_label_index,
      neg_sampling=self.bin_neg_sampling,
      shuffle=True,
      with_edge=True
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader._seed_loader) == 500 / 20

    for batch in loader:
      assert isinstance(batch, Data)

      assert batch.node.size() == (batch.num_nodes, )
      assert batch.edge.size() == (batch.num_edges, )
      assert batch.x.size(0) <= 100
      assert batch.x.min() >= 0 and batch.x.max() < 100
      assert batch.edge_index.min() >= 0
      assert batch.edge_index.max() < batch.num_nodes
      assert batch.edge_attr.min() >= 0
      assert batch.edge_attr.max() < 500
      assert batch.edge_label_index.size(1) == 40
      assert torch.all(batch.edge_label[:20] == 1)
      assert torch.all(batch.edge_label[20:] == 0)

  def test_hetero_link_neighbor_loader(self):
    hetero_data, hetero_dataset = HeteroData(), Dataset()

    hetero_data['paper'].x = torch.arange(100, dtype=torch.float32)
    hetero_data['author'].x = torch.arange(100, 300, dtype=torch.float32)

    hetero_data['paper', 'to', 'paper'].edge_index = get_edge_index(100, 100, 500)
    hetero_data['paper', 'to', 'paper'].edge_attr = torch.arange(500, dtype=torch.float32)
    hetero_data['paper', 'to', 'author'].edge_index = get_edge_index(100, 200, 1000)
    hetero_data['paper', 'to', 'author'].edge_attr = torch.arange(500, 1500, dtype=torch.float32)
    hetero_data['author', 'to', 'paper'].edge_index = get_edge_index(200, 100, 1000)
    hetero_data['author', 'to', 'paper'].edge_attr = torch.arange(1500, 2500, dtype=torch.float32)

    edge_dict, node_feature_dict, edge_feature_dict = {}, {}, {}
    for etype in hetero_data.edge_types:
      edge_dict[etype] = hetero_data[etype]['edge_index']
      edge_feature_dict[etype] = hetero_data[etype]['edge_attr']
    for ntype in hetero_data.node_types:
      node_feature_dict[ntype] = hetero_data[ntype].x.clone(memory_format=torch.contiguous_format)

    hetero_dataset.init_graph(
      edge_index=edge_dict,
      graph_mode='CUDA',
      device=0)

    hetero_dataset.init_node_features(
      node_feature_data=node_feature_dict,
      device_group_list=[DeviceGroup(0, [0])],
      device=0)

    hetero_dataset.init_edge_features(
      edge_feature_data=edge_feature_dict,
      device_group_list=[DeviceGroup(0, [0])],
      device=0)

    bin_loader = LinkNeighborLoader(
      hetero_dataset,
      num_neighbors=[3] * 3,
      edge_label_index=('paper', 'to', 'author'),
      batch_size=20,
      neg_sampling=self.bin_neg_sampling,
      with_edge=True,
      shuffle=True,
    )

    homo_seeds_loader = LinkNeighborLoader(
      hetero_dataset,
      num_neighbors=[3] * 3,
      edge_label_index=('paper', 'to', 'paper'),
      batch_size=20,
      neg_sampling=self.bin_neg_sampling,
      with_edge=True,
      shuffle=True,
    )

    assert str(bin_loader) == 'LinkNeighborLoader()'
    assert len(bin_loader._seed_loader) == 1000 / 20

    for batch in bin_loader:
      assert isinstance(batch, HeteroData)
      batch_a_rev_p = batch['author', 'rev_to', 'paper']
      assert batch_a_rev_p.edge_label_index.size(1) == 40
      assert torch.all(batch['author', 'rev_to', 'paper'].edge_label[:20] == 1)
      assert torch.all(batch['author', 'rev_to', 'paper'].edge_label[20:] == 0)
      assert batch_a_rev_p.edge.size(0) == batch_a_rev_p.edge_attr.size(0)

    for batch in homo_seeds_loader:
      assert isinstance(batch, HeteroData)
      assert batch['paper', 'to', 'paper'].edge_label_index.size(1) == 40
      assert torch.all(batch['paper', 'to', 'paper'].edge_label[:20] == 1)
      assert torch.all(batch['paper', 'to', 'paper'].edge_label[20:] == 0)


    tri_loader = LinkNeighborLoader(
      hetero_dataset,
      num_neighbors=[3] * 3,
      edge_label_index=('paper', 'to', 'author'),
      batch_size=20,
      neg_sampling=self.tri_neg_sampling,
      with_edge=True,
      shuffle=True,
    )

    for batch in tri_loader:
      assert isinstance(batch, HeteroData)
      assert batch['paper'].src_index.size(0) == 20
      assert batch['author'].dst_pos_index.size(0) == 20
      assert batch['author'].dst_neg_index.size(0) == 20
      assert batch['author'].dst_neg_index.size(1) == 3

  def test_hetero_link_neighbor_loader_with_insampling(self):
    hetero_data, hetero_dataset = HeteroData(), Dataset(edge_dir='in')

    hetero_data['paper'].x = torch.arange(100, dtype=torch.float32)
    hetero_data['author'].x = torch.arange(100, 300, dtype=torch.float32)
    hetero_data['institute'].x = torch.arange(300, 350, dtype=torch.float32)

    hetero_data['paper', 'to', 'author'].edge_index = get_edge_index(100, 200, 1000)
    hetero_data['paper', 'to', 'author'].edge_attr = torch.arange(500, 1500, dtype=torch.float32)
    hetero_data['author', 'to', 'institute'].edge_index = get_edge_index(200, 50, 100)
    hetero_data['author', 'to', 'institute'].edge_attr = torch.arange(1500, 1600, dtype=torch.float32)

    edge_dict, node_feature_dict, edge_feature_dict = {}, {}, {}
    for etype in hetero_data.edge_types:
      edge_dict[etype] = hetero_data[etype]['edge_index']
      edge_feature_dict[etype] = hetero_data[etype]['edge_attr']
    for ntype in hetero_data.node_types:
      node_feature_dict[ntype] = hetero_data[ntype].x.clone(memory_format=torch.contiguous_format)

    hetero_dataset.init_graph(
      edge_index=edge_dict,
      graph_mode='CUDA',
      device=0)

    hetero_dataset.init_node_features(
      node_feature_data=node_feature_dict,
      device_group_list=[DeviceGroup(0, [0])],
      device=0)

    hetero_dataset.init_edge_features(
      edge_feature_data=edge_feature_dict,
      device_group_list=[DeviceGroup(0, [0])],
      device=0)

    loader1 = LinkNeighborLoader(
      hetero_dataset,
      num_neighbors=[3] * 2,
      edge_label_index=('paper', 'to', 'author'),
      batch_size=20,
      neg_sampling=self.bin_neg_sampling,
      with_edge=True,
      shuffle=True,
    )

    assert str(loader1) == 'LinkNeighborLoader()'

    for batch in loader1:
      self.assertTrue(set(batch.node_types) == set(['paper', 'author']))
      assert isinstance(batch, HeteroData)
      batch_p2a = batch['paper', 'to', 'author']
      assert batch_p2a.edge_label_index.size(1) == 40
      assert torch.all(batch['paper', 'to', 'author'].edge_label[:20] == 1)
      assert torch.all(batch['paper', 'to', 'author'].edge_label[20:] == 0)
      assert batch_p2a.edge.size(0) == batch_p2a.edge_attr.size(0)
    
    loader2 = LinkNeighborLoader(
      hetero_dataset,
      num_neighbors=[3] * 2,
      edge_label_index=('author', 'to', 'institute'),
      batch_size=20,
      neg_sampling=self.bin_neg_sampling,
      with_edge=True,
      shuffle=True,
    )

    for batch in loader2:
      self.assertTrue(set(batch.node_types) == set(['author', 'institute', 'paper']))
      assert isinstance(batch, HeteroData)
      batch_p2a = batch['author', 'to', 'institute']
      assert batch_p2a.edge_label_index.size(1) == 40
      assert torch.all(batch['author', 'to', 'institute'].edge_label[:20] == 1)
      assert torch.all(batch['author', 'to', 'institute'].edge_label[20:] == 0)
      assert batch_p2a.edge.size(0) == batch_p2a.edge_attr.size(0)


if __name__ == "__main__":
  unittest.main()