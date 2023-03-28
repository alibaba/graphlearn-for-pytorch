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

from graphlearn_torch.data import CSRTopo, DeviceGroup, Feature, sort_by_in_degree
from graphlearn_torch.utils import tensor_equal_with_device


class FeatureTestCase(unittest.TestCase):
  def setUp(self):
    tensor = torch.ones(128, 128, dtype=torch.float32)
    self.tensor = torch.cat([tensor, tensor*2, tensor*3], 0)

    rows = torch.cat([torch.arange(128*3),
                      torch.randint(128, (128*3,)),
                      torch.randint(128*2, (128*3,))])
    cols = torch.cat([torch.randint(128*3, (128*3,)),
                      torch.randint(128*3, (128*3,)),
                      torch.randint(128*3, (128*3,))])
    self.csr_topo = CSRTopo(edge_index=torch.stack([rows, cols]))
    self.input = torch.tensor([10, 20, 200, 210, 300, 310], dtype=torch.int64,
                              device= torch.device('cuda:0'))
    attr = torch.ones(2, 128, dtype=torch.float32,
                      device= torch.device('cuda:0'))
    self.res = torch.cat((attr, attr*2, attr*3), 0)

  def test_feature_without_degree_sort(self):
    device_group_list = [DeviceGroup(0, [0])]
    feature = Feature(
      feature_tensor=self.tensor.clone(), split_ratio=0.5,
      device_group_list=device_group_list, device=0)
    self.assertEqual(list(feature.shape), [128*3, 128])
    self.assertTrue(tensor_equal_with_device(feature[self.input], self.res))

  def test_feature_with_degree_sort(self):
    device_group_list = [DeviceGroup(0, [0, 1])]
    cpu_tensor, id2index = sort_by_in_degree(
      self.tensor.clone(), 0.5, self.csr_topo)
    feature = Feature(
      feature_tensor=cpu_tensor, id2index=id2index, split_ratio=0.5,
      device_group_list=device_group_list, device=0)
    self.assertEqual(list(feature.shape), [128*3, 128])
    self.assertTrue(tensor_equal_with_device(feature[self.input], self.res))

  def test_feature_with_degree_sort_pin(self):
    cpu_tensor, id2index = sort_by_in_degree(
      self.tensor.clone(), 0.0, self.csr_topo)
    feature = Feature(feature_tensor=cpu_tensor, id2index=id2index)
    self.assertEqual(list(feature.shape), [128*3, 128])
    self.assertTrue(tensor_equal_with_device(feature[self.input], self.res))

  def test_feature_with_degree_sort_cpu(self):
    cpu_tensor, id2index = sort_by_in_degree(
      self.tensor.clone(), 0.0, self.csr_topo)
    feature = Feature(feature_tensor=cpu_tensor, id2index=id2index,
      with_gpu=False)
    self.assertEqual(list(feature.shape), [128*3, 128])
    self.assertTrue(tensor_equal_with_device(feature[self.input], self.res.cpu()))

  def test_feature_with_degree_sort_gpu(self):
    device_group_list = [DeviceGroup(0, [0, 1])]
    cpu_tensor, id2index = sort_by_in_degree(
      self.tensor.clone(), 1.0, self.csr_topo)
    feature = Feature(
      feature_tensor=cpu_tensor, id2index=id2index, split_ratio=1.0,
      device_group_list=device_group_list, device=0)
    self.assertEqual(list(feature.shape), [128*3, 128])
    self.assertTrue(tensor_equal_with_device(feature[self.input], self.res))


if __name__ == "__main__":
  unittest.main()