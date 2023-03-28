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

import torch
import graphlearn_torch as glt

import torch.multiprocessing as mp

def test_feature(rank, world_size, feature):
  torch.cuda.set_device(rank)
  assert list(feature.shape) == [128*3, 128]
  input = torch.tensor([10, 20, 200, 210, 300, 310], dtype=torch.int64,
                        device= torch.device(rank))
  attr = torch.ones(2, 128, dtype=torch.float32,
                    device=torch.device(rank))
  res = torch.cat((attr, attr*2, attr*3), 0)
  assert glt.utils.tensor_equal_with_device(feature[input], res)

if __name__ == "__main__":
  print('Use GPU 0 and GPU 1 for multiprocessing Feature test.')
  world_size = 2
  attr = torch.ones(128, 128, dtype=torch.float32)
  tensor = torch.cat([attr, attr*2, attr*3], 0)

  rows = torch.cat([torch.arange(128*3),
                    torch.randint(128, (128*3,)),
                    torch.randint(128*2, (128*3,))])
  cols = torch.cat([torch.randint(128*3, (128*3,)),
                    torch.randint(128*3, (128*3,)),
                    torch.randint(128*3, (128*3,))])
  csr_topo = glt.data.CSRTopo(edge_index=torch.stack([rows, cols]))

  device_group_list = [glt.data.DeviceGroup(0, [0]),
                       glt.data.DeviceGroup(1, [1])]
  #device_group_list = [glt.data.DeviceGroup(0, [0, 1])]
  split_ratio = 0.8 # [0, 1]
  cpu_tensor, id2index = glt.data.sort_by_in_degree(tensor, split_ratio, csr_topo)
  feature = glt.data.Feature(cpu_tensor, id2index, split_ratio, device_group_list, 0)
  mp.spawn(test_feature,
           args=(world_size, feature),
           nprocs=world_size,
           join=True)