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

import threading
from typing import Optional

import torch


def get_available_device(device: Optional[torch.device] = None) -> torch.device:
  r""" Get an available device. If the input device is not ``None``, it will
  be returened directly. Otherwise an available device will be choosed (
  current cuda device will be preferred if available).
  """
  if device is not None:
    return torch.device(device)
  if torch.cuda.is_available():
    return torch.device('cuda', torch.cuda.current_device())
  return torch.device('cpu')


_cuda_device_assign_lock = threading.RLock()
_cuda_device_rank = 0

def assign_device():
  r""" Assign an device to use, the cuda device will be preferred if available.
  """
  if torch.cuda.is_available():
    global _cuda_device_rank
    with _cuda_device_assign_lock:
      device_rank = _cuda_device_rank
      _cuda_device_rank = (_cuda_device_rank + 1) % torch.cuda.device_count()
    return torch.device('cuda', device_rank)
  return torch.device('cpu')


def ensure_device(device: torch.device):
  r""" Make sure that current cuda kernel corresponds to the assigned device.
  """
  if (device.type == 'cuda' and
      device.index != torch.cuda.current_device()):
    torch.cuda.set_device(device)
