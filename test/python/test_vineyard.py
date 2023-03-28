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
import unittest

from graphlearn_torch.data import *


class LoadVineyardTest(unittest.TestCase):
  '''This test case is for the dataset
     https://github.com/GraphScope/gstest/tree/master/modern_graph
  '''

  @unittest.skip("Vineyard Env Needed")
  def setUp(self):
    sock = os.environ["socket"]
    fid = os.environ["fid"]
    self.indptr, self.indices, self.edge_ids = vineyard_to_csr(sock, fid, 0, 0, 1)
    self.vfeat = load_vertex_feature_from_vineyard(
      sock, fid, ['age', 'id'], 0, 'int64')
    self.efeat = load_edge_feature_from_vineyard(
      sock, fid, ["weight"], 0, 'float64')

  @unittest.skip("Vineyard Env Needed")
  def test_vineyard_csr(self):
    self.assertEqual(len(self.indptr), 5)
    self.assertEqual(len(self.indices), 2)
    self.assertEqual(sum(self.edge_ids), 1)
    self.assertEqual(len(self.vfeat), 4)
    self.assertEqual(len(self.vfeat[0]), 2)
    self.assertEqual(self.vfeat[2, 0], 32)
    self.assertEqual(len(self.efeat), 2)
    self.assertAlmostEqual(self.efeat[0].item(), 0.5)

if __name__ == "__main__":
  unittest.main()
