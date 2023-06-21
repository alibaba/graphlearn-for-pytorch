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
from graphlearn_torch.sampler import NeighborSampler
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
    indptr = torch.tensor([0, 2, 4, 6, 7], dtype=torch.int64)
    indices = torch.tensor([0, 1, 1, 3, 2, 4, 5], dtype=torch.int64)
    self.csr_topo = Topology(edge_index=(indptr, indices), input_layout='CSR')
    # input
    self.input_seeds1 = torch.tensor([0, 2, 1, 2, 4], dtype=torch.int64)
    # output
    self.nodes1 = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
    self.mapping1 = torch.tensor([0, 2, 1, 2, 3], dtype=torch.int64)
    self.rows1 = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
    self.cols1 = torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64)
    self.eids1 = torch.tensor([0, 1, 2, 4, 5], dtype=torch.int64)

    # input
    self.input_seeds2 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    # output
    self.nodes2 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    self.mapping2 = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    self.rows2 = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int64)
    self.cols2 = torch.tensor([0, 1, 1, 3, 2, 4, 5], dtype=torch.int64)
    self.eids2 = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)

  def test_cpu_node_subgraph(self):
    g = Graph(self.csr_topo, mode='CPU')
    sampler = NeighborSampler(g, device=torch.device('cpu'), with_edge=True)
    subgraph = sampler.subgraph(self.input_seeds1)
    self.assertTrue(tensor_equal_with_device(subgraph.node, self.nodes1))
    self.assertTrue(tensor_equal_with_device(subgraph.metadata, self.mapping1))
    self.assertTrue(tensor_equal_with_device(subgraph.row, self.cols1))
    self.assertTrue(tensor_equal_with_device(subgraph.col, self.rows1))
    self.assertTrue(tensor_equal_with_device(subgraph.edge, self.eids1))

  def test_cpu_khop_subgraph(self):
    g = Graph(self.csr_topo, mode='CPU')
    sampler = NeighborSampler(g, device=torch.device('cpu'),
                              num_neighbors=[-1, -1], with_edge=True)
    subgraph = sampler.subgraph(self.input_seeds2)
    self.assertTrue(tensor_equal_with_device(subgraph.node, self.nodes2))
    self.assertTrue(tensor_equal_with_device(subgraph.metadata, self.mapping2))
    self.assertTrue(tensor_equal_with_device(subgraph.row, self.cols2))
    self.assertTrue(tensor_equal_with_device(subgraph.col, self.rows2))
    self.assertTrue(tensor_equal_with_device(subgraph.edge, self.eids2))

  def test_cuda_node_subgraph(self):
    g = Graph(self.csr_topo, mode='CUDA')
    sampler = NeighborSampler(g, device=torch.device('cuda:0'), with_edge=True)
    subgraph = sampler.subgraph(self.input_seeds1)
    self.assertTrue(tensor_equal_with_device(subgraph.node, self.nodes1.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.metadata, self.mapping1.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.row, self.cols1.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.col, self.rows1.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.edge, self.eids1.to(0)))

  def test_cuda_khop_subgraph(self):
    g = Graph(self.csr_topo, mode='CUDA')
    sampler = NeighborSampler(g, device=torch.device('cuda:0'),
                              num_neighbors=[-1, -1], with_edge=True)
    subgraph = sampler.subgraph(self.input_seeds2)
    self.assertTrue(tensor_equal_with_device(subgraph.node, self.nodes2.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.metadata, self.mapping2.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.row, self.cols2.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.col, self.rows2.to(0)))
    self.assertTrue(tensor_equal_with_device(subgraph.edge, self.eids2.to(0)))

if __name__ == "__main__":
  unittest.main()