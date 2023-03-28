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

from typing import List

import torch

from .. import py_graphlearn_torch as pywrap


class UnifiedTensor(object):
  r""" Creates a CPU and GPUs unified Tensor for GPU direct access.
  For the tensor stored in the CPU memory, we use ZERO-COPY to provide
  efficient GPU access. For tensors stored in the GPU memory, p2p access
  between GPUs(such as NVLink) is required.

  Args:
    current_device (int): An integer to represent the GPU device where the
      underlying cuda operation kernel is launched.
    dtype (torch.dtype): The data type of the tensor elements.
  """
  def __init__(self, current_device: int, dtype: torch.dtype = torch.float32):
    self.current_device = current_device
    self.dtype = dtype
    self.unified_tensor = pywrap.UnifiedTensor(current_device, dtype)
    self.cpu_part = None # tensor stored in CPU memory.

  def __getitem__(self, ids):
    ids = ids.to(self.current_device)
    return self.unified_tensor[ids]

  def append_shared_tensor(self, shared_tensor):
    r""" Append from `SharedTensor`.

    Args:
      shared_tensor: A `pywrap.SharedTensor` object which means GPU tensor that
        can be shared with other GPUs.
    """
    self.unified_tensor.append_shared_tensor(shared_tensor)

  def append_cpu_tensor(self, cpu_tensor: torch.Tensor):
    r""" Append from CPU tensor.

    Args:
      cpu_tensor: A CPU torch.Tensor object which will be stored
        in pinned memory for ZERO-COPY.
    """
    self.unified_tensor.append_cpu_tensor(cpu_tensor)

  def init_from(self, tensors: List[torch.Tensor], tensor_devices: List[int]):
    r""" Initialize from CPU torch.Tensors.

    Args:
      tensors: CPU torch.Tensors indicating the tensors that need to be stored
        on different GPUs and CPU.
      tensor_devices: The indices of devices indicating the location of the
        tensor storage, -1 means on CPU and other > 0 value means on GPUs.
      Note that tensors and tensor_devices must correspond to each other.
    """
    self.unified_tensor.init_from(tensors, tensor_devices)

  @property
  def shape(self):
    return self.unified_tensor.shape()

  @property
  def device(self):
    return self.unified_tensor.device()

  @property
  def numel(self):
    return self.unified_tensor.numel()

  def size(self, dim):
    return self.unified_tensor.size(dim)

  def stride(self, dim):
    return self.unified_tensor.stride(dim)

  def share_ipc(self):
    r""" Shares ipc handles.

    Returns:
      A list of cuda ipcs and cpu part tensor.
    """
    shared_tensors = self.unified_tensor.share_cuda_ipc()
    cuda_ipc_list = [item.share_cuda_ipc() for item in shared_tensors]
    return cuda_ipc_list, self.cpu_part

  def from_ipc_handle(self, cuda_ipc_list, cpu_part):
    r""" Builds from ipc handle.

    Args:
      cuda_ipc_list: A list of CUDA ipcs, in the same order as tensors_devices.
      cpu_part: A CPU torch.Tensor.
    """
    for ipc in cuda_ipc_list:
      shared_tensor = pywrap.SharedTensor()
      shared_tensor.from_cuda_ipc(ipc)
      self.unified_tensor.append_shared_tensor(shared_tensor)
    if cpu_part is not None and cpu_part.numel() > 0:
      self.cpu_part = cpu_part
      self.unified_tensor.append_cpu_tensor(cpu_part)

  @classmethod
  def new_from_ipc(cls, ipc_handles, current_device: int, dtype: torch.dtype):
    r""" Creates `UnifiedTensor` from ipc handles.

    Args:
      ipc_handles: ipc handles consists of CUDA ipcs and cpu part torch.Tensor.
      current_device (int): An integer to represent the GPU device where the
        underlying cuda operation kernel is launched.
      dtype (torch.dtype): The data type of the tensor elements.

    Returns:
      A `UnifiedTensor` instance.
    """
    cuda_ipc_list, cpu_part = ipc_handles
    unified_tensor = cls(current_device, dtype)
    unified_tensor.from_ipc_handle(cuda_ipc_list, cpu_part)
    return unified_tensor
