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

import torch.multiprocessing as mp

from .base import SampleMessage, ChannelBase


class MpChannel(ChannelBase):
  r""" A simple multiprocessing channel using `torch.multiprocessing.Queue`.

  Args:
    The input arguments should be consistent with `torch.multiprocessing.Queue`.
  """
  def __init__(self, **kwargs):
    self._queue = mp.get_context('spawn').Queue(**kwargs)

  def send(self, msg: SampleMessage, **kwargs):
    self._queue.put(msg, **kwargs)

  def recv(self, **kwargs) -> SampleMessage:
    return self._queue.get(**kwargs)
