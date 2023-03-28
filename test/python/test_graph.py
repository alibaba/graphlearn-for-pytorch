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

from graphlearn_torch.data import CSRTopo, Graph


class GraphTestCase(unittest.TestCase):
  def setUp(self):
    self.indptr = torch.tensor([0, 2, 4, 7, 8], dtype=torch.int64)
    self.indices = torch.tensor([0, 1, 1, 3, 2, 3, 4, 5], dtype=torch.int64)
    self.edge_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)
    self.csr_topo = CSRTopo(
      edge_index=(self.indptr, self.indices),
      edge_ids=self.edge_ids,
      layout='CSR'
    )

  def test_csr_topo_with_coo(self):
    row = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3], dtype=torch.int64)
    col = torch.tensor([0, 1, 1, 3, 2, 3, 4, 5], dtype=torch.int64)
    edge_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64)

    csr_topo_from_coo = CSRTopo(
      edge_index=(row, col), edge_ids=edge_ids, layout='COO'
    )
    self.assertTrue(torch.equal(self.indptr, csr_topo_from_coo.indptr))
    self.assertTrue(torch.equal(self.indices, csr_topo_from_coo.indices))
    self.assertTrue(torch.equal(self.edge_ids, csr_topo_from_coo.edge_ids))

    row_from_csr, col_from_csr, edge_ids_from_csr = self.csr_topo.to_coo()
    self.assertTrue(torch.equal(row, row_from_csr))
    self.assertTrue(torch.equal(col, col_from_csr))
    self.assertTrue(torch.equal(edge_ids, edge_ids_from_csr))

  def test_csr_topo_with_csc(self):
    row = torch.tensor([0, 0, 1, 2, 1, 2, 2, 3], dtype=torch.int64)
    colptr = torch.tensor([0, 1, 3, 4, 6, 7, 8], dtype=torch.int64)
    edge_ids = torch.tensor([0, 1, 2, 4, 3, 5, 6, 7], dtype=torch.int64)

    csr_topo_from_csc = CSRTopo(
      edge_index=(row, colptr), edge_ids=edge_ids, layout='CSC'
    )
    self.assertTrue(torch.equal(self.indptr, csr_topo_from_csc.indptr))
    self.assertTrue(torch.equal(self.indices, csr_topo_from_csc.indices))
    self.assertTrue(torch.equal(self.edge_ids, csr_topo_from_csc.edge_ids))

    row_from_csr, colptr_from_csr, edge_ids_from_csr = self.csr_topo.to_csc()
    self.assertTrue(torch.equal(row, row_from_csr))
    self.assertTrue(torch.equal(colptr, colptr_from_csr))
    self.assertTrue(torch.equal(edge_ids, edge_ids_from_csr))

  def test_cpu_graph_init(self):
    g = Graph(self.csr_topo, mode='CPU')
    self.assertEqual(g.edge_count, self.indices.size(0))
    self.assertEqual(g.row_count, self.indptr.size(0) - 1)

  def test_cuda_graph_init(self):
    g = Graph(self.csr_topo, 'CUDA', 0)
    self.assertEqual(g.edge_count, self.indices.size(0))
    self.assertEqual(g.row_count, self.indptr.size(0) - 1)

  def test_pin_graph_init(self):
    g = Graph(self.csr_topo, 'ZERO_COPY', 0)
    self.assertEqual(g.edge_count, self.indices.size(0))
    self.assertEqual(g.row_count, self.indptr.size(0) - 1)


if __name__ == "__main__":
  unittest.main()