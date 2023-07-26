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

from abc import ABC, abstractmethod
from typing import Dict
import torch
from .. import py_graphlearn_torch as pywrap

QueueTimeoutError =  pywrap.QueueTimeoutError

# A `SampleMessage` contains all possible results from a sampler, including
# subgraph data, features and user defined metas.
SampleMessage = Dict[str, torch.Tensor]


class ChannelBase(ABC):
  r""" A base class that initializes a channel for sample messages and
    provides :meth:`send` and :meth:`recv` routines.
  """
  @abstractmethod
  def send(self, msg: SampleMessage, **kwargs):
    r""" Send a sample message into channel, the implemented channel should
      porcess this message data properly.

    Args:
      msg: The sample message to send.
    """

  @abstractmethod
  def recv(self, **kwargs) -> SampleMessage:
    r""" Recv a sample message from channel.
    """
