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

from typing import Union

from .. import py_graphlearn_torch as pywrap
from ..utils import parse_size

from .base import SampleMessage, ChannelBase


class ShmChannel(ChannelBase):
  r""" A communication channel for sample messages based on a shared-memory
    queue, which is implemented in the underlying c++ lib.

  Note that the underlying shared-memory buffer of this channel is pinnable,
    which will achieve better performance when the consumer needs to copy
    data from channel to gpu device.

  Args:
    capacity: The max bufferd number of sample messages in channel.
    shm_size: The allocated size (bytes) for underlying shared-memory.

  When the producer send sample message to the channel, it will be limited by
    both `capacity` and `shm_size`. E.g, if current number of buffered
    messages in channel reaches the `capacity` limit, or current used
    buffer memory reaches the `shm_size` limit, the current `send` operation
    will be blocked until some messages in channel are consumed and related
    resource are released.
  """
  def __init__(self,
               capacity: int=128,
               shm_size: Union[str, int]='256MB'):
    assert capacity > 0
    shm_size = parse_size(shm_size)
    self._queue = pywrap.SampleQueue(capacity, shm_size)

  def pin_memory(self):
    r""" Pin underlying shared-memory.
    """
    self._queue.pin_memory()

  def empty(self) -> bool:
    r""" Whether the queue is empty.
    """
    return self._queue.empty()

  def send(self, msg: SampleMessage, **kwargs):
    self._queue.send(msg)

  def recv(self, timeout_ms=None, **kwargs) -> SampleMessage:
    if timeout_ms is None:
      timeout_ms = 0
    return self._queue.receive(timeout_ms=timeout_ms)
