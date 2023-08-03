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
import torch.multiprocessing as mp

from graphlearn_torch.channel import ShmChannel, QueueTimeoutError


def run_sender(_, channel):
  for i in range(1, 11):
    send_map = {}
    send_map['from_cpu'] = torch.ones([i, i], dtype=torch.float32)
    send_map['from_cuda'] = torch.arange(i, dtype=torch.int32,
                                         device=torch.device('cuda'))
    channel.send(send_map)
    print("[sender] message {} sent".format(i))


def run_receiver(_, channel):
  channel.pin_memory()
  print("[receiver] memory pinned!")
  tc = unittest.TestCase()
  for i in range(1, 11):
    received_map = channel.recv()
    print("[receiver] message {} received".format(i))
    tc.assertEqual(len(received_map), 2)
    tc.assertTrue('from_cpu' in received_map)
    tc.assertEqual(received_map['from_cpu'].device, torch.device('cpu'))
    tc.assertTrue(torch.equal(received_map['from_cpu'],
                              torch.ones([i, i], dtype=torch.float32)))
    tc.assertTrue('from_cuda' in received_map)
    tc.assertEqual(received_map['from_cuda'].device, torch.device('cpu'))
    tc.assertTrue(torch.equal(received_map['from_cuda'],
                              torch.arange(i, dtype=torch.int32)))
  try:
    channel.recv(10)
  except QueueTimeoutError as e:
    print('Expected Error', e)
  tc.assertTrue(channel.empty())


class SampleQueueCase(unittest.TestCase):
  def test_send_and_receive(self):
    channel = ShmChannel(capacity=5, shm_size=1024*1024)
    ctx1 = mp.spawn(run_sender, args=(channel,), nprocs=1, join=False)
    ctx2 = mp.spawn(run_receiver, args=(channel,), nprocs=1, join=False)
    ctx1.join()
    ctx2.join()


if __name__ == "__main__":
  unittest.main()
