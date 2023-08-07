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

from graphlearn_torch.data import Topology, Graph
from graphlearn_torch.sampler import NeighborSampler, NegativeSampling, EdgeSamplerInput
from graphlearn_torch.utils import tensor_equal_with_device


class RandomSamplerTestCase(unittest.TestCase):
  def setUp(self):
    """
    input graph:
      1 1 0 0 0 0
      0 1 0 1 0 0
      0 0 1 0 1 0
      0 0 0 0 0 1
    """
    indptr = torch.tensor([0, 2, 4, 6, 7, 7, 7], dtype=torch.int64)
    indices = torch.tensor([0, 1, 1, 3, 2, 4, 5], dtype=torch.int64)
    self.csr_topo = Topology(edge_index=(indptr, indices), input_layout='CSR')

    self.input_seeds = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    self.num_neighbors = [2]

    self.nodes = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    self.edge_index = torch.tensor([[0, 1, 1, 3, 2, 4, 5],
                                    [0, 0, 1, 1, 2, 2, 3]],
                                   dtype=torch.int64)
    self.edge_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
    self.edge_weights = torch.tensor([.1, .9, .1, .9, .1, .9, .1], dtype=torch.float)

    self.csr_topo_with_weight = Topology(edge_index=(indptr, indices),
                                         input_layout='CSR',
                                         edge_weights=self.edge_weights)
    self.device = torch.device('cuda:0')
    self.cuda_nodes = torch.tensor([0, 1, 2, 3, 4, 5],
                                   dtype=torch.int64, device=self.device)
    self.cuda_edge_index = torch.tensor([[0, 1, 1, 3, 2, 4, 5],
                                         [0, 0, 1, 1, 2, 2, 3]],
                                        dtype=torch.int64, device=self.device)
    self.cuda_edge_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6],
                                      dtype=torch.int64, device=self.device)
    self.edge_label_index = torch.tensor([[0, 3],
                                          [1, 5]],
                                          dtype=torch.int64)
    self.cuda_edge_label_index = torch.tensor([[0, 3],
                                               [1, 5]],
                                              dtype=torch.int64,
                                              device=self.device)

  def test_cpu_sample_from_node(self):
    g = Graph(self.csr_topo, mode='CPU')
    sampler = NeighborSampler(g, self.num_neighbors, device=torch.device('cpu'))
    sample_out = sampler.sample_from_nodes(self.input_seeds)
    nodes = sample_out.node
    edge_index = torch.stack([sample_out.row, sample_out.col])
    self.assertTrue(tensor_equal_with_device(nodes, self.nodes))
    self.assertTrue(tensor_equal_with_device(edge_index, self.edge_index))

  def test_cuda_sample_from_node(self):
    g = Graph(self.csr_topo, 'CUDA', 0)
    sampler = NeighborSampler(g, self.num_neighbors, device=self.device)
    sample_out = sampler.sample_from_nodes(self.input_seeds)
    nodes = sample_out.node
    edge_index = torch.stack([sample_out.row, sample_out.col])
    self.assertTrue(tensor_equal_with_device(nodes, self.cuda_nodes))
    self.assertTrue(tensor_equal_with_device(edge_index, self.cuda_edge_index))

  def test_pin_sample_from_node(self):
    g = Graph(self.csr_topo, 'ZERO_COPY', 0)
    sampler = NeighborSampler(g, self.num_neighbors, device=self.device)
    sample_out = sampler.sample_from_nodes(self.input_seeds)
    nodes = sample_out.node
    edge_index = torch.stack([sample_out.row, sample_out.col])
    self.assertTrue(tensor_equal_with_device(nodes, self.cuda_nodes))
    self.assertTrue(tensor_equal_with_device(edge_index, self.cuda_edge_index))

  def test_cuda_sample_prob(self):
    g = Graph(self.csr_topo, 'CUDA', 0)
    sampler = NeighborSampler(g, [2, 2], device=self.device)
    probs = sampler.sample_prob(
      torch.tensor([0, 4], dtype=torch.int64),
      self.nodes.size(0)
    )
    print(probs)

  def test_cpu_sample_from_node_with_edge(self):
    g = Graph(self.csr_topo, mode='CPU')
    sampler = NeighborSampler(
      g, self.num_neighbors, device=torch.device('cpu'), with_edge=True
    )
    sample_out = sampler.sample_from_nodes(self.input_seeds)
    self.assertTrue(tensor_equal_with_device(sample_out.node, self.nodes))
    self.assertTrue(tensor_equal_with_device(
      torch.stack([sample_out.row, sample_out.col]), self.edge_index))
    self.assertTrue(tensor_equal_with_device(sample_out.edge, self.edge_ids))
  
  def test_cpu_weighted_sample_from_node_with_edge(self):
    g = Graph(self.csr_topo_with_weight, mode='CPU')
    stats = torch.zeros(7)
    sampler = NeighborSampler(
      g, [1], with_edge=True, with_weight=True, device=torch.device('cpu'))
    
    for _ in range(1000):
      sample_out = sampler.sample_from_nodes(self.input_seeds)
      edges = sample_out.edge
      stats.scatter_add_(0, edges, torch.ones(7))

    # with high probability holds
    self.assertTrue(stats[0] < 200 and stats[2] < 200 and stats[4] < 200)
    self.assertTrue(stats[1] > 800 and stats[3] > 800 and stats[5] > 800)
    self.assertEqual(stats[6], 1000)

  def test_cuda_sample_from_node_with_edge(self):
    g = Graph(self.csr_topo, 'CUDA', 0)
    sampler = NeighborSampler(
      g, self.num_neighbors, device=self.device, with_edge=True
    )
    sample_out = sampler.sample_from_nodes(self.input_seeds)
    self.assertTrue(tensor_equal_with_device(sample_out.node, self.cuda_nodes))
    self.assertTrue(tensor_equal_with_device(
      torch.stack([sample_out.row, sample_out.col]), self.cuda_edge_index))
    self.assertTrue(tensor_equal_with_device(sample_out.edge, self.cuda_edge_ids))

  def test_cpu_sample_from_edges(self):
    g = Graph(self.csr_topo, mode='CPU')
    bin_neg_sampling = NegativeSampling('binary')
    tri_neg_sampling = NegativeSampling('triplet')
    sampler = NeighborSampler(
      g, self.num_neighbors, device=torch.device('cpu'), with_neg=True,
    )
    bin_inputs = EdgeSamplerInput(self.edge_label_index[0],
                                  self.edge_label_index[1],
                                  neg_sampling=bin_neg_sampling)
    tri_inputs = EdgeSamplerInput(self.edge_label_index[0],
                                  self.edge_label_index[1],
                                  neg_sampling=tri_neg_sampling)
    sample_out = sampler.sample_from_edges(bin_inputs)
    out_index = sample_out.metadata['edge_label_index']
    pos_index = sample_out.node[out_index[:,:2]]
    neg_index = sample_out.node[out_index[:,2:]]
    baseline = torch.stack((self.edge_index[1], self.edge_index[0]), dim=0).T
    self.assertTrue(tensor_equal_with_device(pos_index, self.edge_label_index))
    self.assertFalse(
      torch.any(torch.sum((neg_index.T[0]==baseline).to(torch.int), dim=1) == 2)
    )

    sample_out = sampler.sample_from_edges(tri_inputs)
    pos_index = torch.stack(
      (sample_out.node[sample_out.metadata['src_index']],
       sample_out.node[sample_out.metadata['dst_pos_index']]
      ), dim=0)
    self.assertTrue(tensor_equal_with_device(pos_index, self.edge_label_index))


  def test_cuda_sample_from_edges(self):
    g = Graph(self.csr_topo, mode='CUDA')
    bin_neg_sampling = NegativeSampling('binary')
    tri_neg_sampling = NegativeSampling('triplet')
    sampler = NeighborSampler(
      g, self.num_neighbors, device=torch.device('cuda:0'), with_neg=True,
    )
    bin_inputs = EdgeSamplerInput(self.edge_label_index[0],
                                  self.edge_label_index[1],
                                  neg_sampling=bin_neg_sampling)
    tri_inputs = EdgeSamplerInput(self.edge_label_index[0],
                                  self.edge_label_index[1],
                                  neg_sampling=tri_neg_sampling)
    sample_out = sampler.sample_from_edges(bin_inputs)
    out_index = sample_out.metadata['edge_label_index']
    pos_index = sample_out.node[out_index[:,:2]]
    neg_index = sample_out.node[out_index[:,2:]]
    baseline = torch.stack((self.cuda_edge_index[1], self.cuda_edge_index[0]),
                            dim=0).T
    self.assertTrue(tensor_equal_with_device(pos_index,
                                             self.cuda_edge_label_index))
    self.assertFalse(
      torch.any(torch.sum((neg_index.T[0]==baseline).to(torch.int), dim=1) == 2)
    )

    sample_out = sampler.sample_from_edges(tri_inputs)
    pos_index = torch.stack(
      (sample_out.node[sample_out.metadata['src_index']],
       sample_out.node[sample_out.metadata['dst_pos_index']]
      ), dim=0)
    self.assertTrue(tensor_equal_with_device(pos_index,
                                             self.cuda_edge_label_index))

if __name__ == "__main__":
  unittest.main()