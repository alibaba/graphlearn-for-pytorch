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

from graphlearn_torch.data import UnifiedTensor
from graphlearn_torch.utils import tensor_equal_with_device


class UnifiedTensorTestCase(unittest.TestCase):
  def test_unified_tensor_with_int_types(self):
    self._test_with_dtype(torch.uint8)
    self._test_with_dtype(torch.int8)
    self._test_with_dtype(torch.int16)
    self._test_with_dtype(torch.int32)
    self._test_with_dtype(torch.int64)

  def test_unified_tensor_with_float_types(self):
    self._test_with_dtype(torch.float16)
    self._test_with_dtype(torch.float32)
    self._test_with_dtype(torch.float64)
    self._test_with_dtype(torch.bfloat16)
    self._test_with_dtype(torch.complex64)
    self._test_with_dtype(torch.complex128)

  def _test_with_dtype(self, dtype: torch.dtype):
    cpu_tensor1 = torch.ones(128, 128, dtype=dtype)
    cpu_tensor2 = cpu_tensor1 * 2

    tensors = [cpu_tensor1, cpu_tensor2]
    tensor_devices = [0, -1]

    unified_tensor = UnifiedTensor(0, dtype)
    unified_tensor.init_from(tensors, tensor_devices)
    self.assertEqual(unified_tensor.shape, [128*2, 128])
    input = torch.tensor([10, 20, 200, 210], dtype=torch.int64,
                         device= torch.device('cuda:0'))
    feature = torch.ones(2, 128, dtype=dtype, device=torch.device('cuda:0'))
    expected_res = torch.cat((feature, feature*2), 0)
    res = unified_tensor[input]
    self.assertEqual(expected_res.dtype, res.dtype)
    self.assertTrue(tensor_equal_with_device(expected_res, res))


if __name__ == "__main__":
  unittest.main()