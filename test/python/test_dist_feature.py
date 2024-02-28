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
import graphlearn_torch as glt


def run_dist_feature_test(world_size: int, rank: int, feature: glt.data.Feature,
                          partition_book: glt.data.PartitionBook, master_port: int):
  glt.distributed.init_worker_group(world_size, rank, 'dist-feature-test')
  glt.distributed.init_rpc(master_addr='localhost', master_port=master_port)

  partition2workers = glt.distributed.rpc_sync_data_partitions(world_size, rank)
  rpc_router = glt.distributed.RpcDataPartitionRouter(partition2workers)

  current_device = torch.device('cuda', rank % 2)

  dist_feature = glt.distributed.DistFeature(
    world_size, rank, feature, partition_book,
    local_only=False, rpc_router=rpc_router,
    device=current_device
  )

  input = torch.tensor(
    [10, 20, 260, 360, 200, 210, 420, 430],
    dtype=torch.int64,
    device=current_device
  )
  expected_features = torch.cat([
    torch.ones(2, 1024, dtype=torch.float32, device=current_device),
    torch.zeros(2, 1024, dtype=torch.float32, device=current_device),
    torch.ones(2, 1024, dtype=torch.float32, device=current_device)*2,
    torch.zeros(2, 1024, dtype=torch.float32, device=current_device)
  ])
  res = dist_feature[input]

  tc = unittest.TestCase()
  tc.assertTrue(glt.utils.tensor_equal_with_device(res, expected_features))

  glt.distributed.shutdown_rpc()


class DistFeatureTestCase(unittest.TestCase):
  def test_dist_feature_lookup(self):
    cpu_tensor0 = torch.cat([
      torch.ones(128, 1024, dtype=torch.float32),
      torch.ones(128, 1024, dtype=torch.float32)*2
    ])
    cpu_tensor1 = torch.cat([
      torch.zeros(128, 1024, dtype=torch.float32),
      torch.zeros(128, 1024, dtype=torch.float32)
    ])

    id2index = torch.arange(128 * 4)
    id2index[128*2:] -= 128*2

    partition_book = torch.cat([
      torch.zeros(128*2, dtype=torch.long),
      torch.ones(128*2, dtype=torch.long)
    ])
    partition_book.share_memory_()

    device_group_list = [
      glt.data.DeviceGroup(0, [0]),
      glt.data.DeviceGroup(1, [1])
    ]

    split_ratio = 0.8

    feature0 = glt.data.Feature(cpu_tensor0, id2index,
                                split_ratio, device_group_list)
    feature1 = glt.data.Feature(cpu_tensor1, id2index,
                                split_ratio, device_group_list)

    mp_context = torch.multiprocessing.get_context('spawn')
    port = glt.utils.get_free_port()
    w0 = mp_context.Process(
      target=run_dist_feature_test,
      args=(2, 0, feature0, partition_book, port)
    )
    w1 = mp_context.Process(
      target=run_dist_feature_test,
      args=(2, 1, feature1, partition_book, port)
    )
    w0.start()
    w1.start()
    w0.join()
    w1.join()


if __name__ == "__main__":
  unittest.main()
