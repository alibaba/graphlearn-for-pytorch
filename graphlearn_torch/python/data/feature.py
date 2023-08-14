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
from multiprocessing.reduction import ForkingPickler
from typing import List, Optional

import torch

from ..typing import TensorDataType
from ..utils import convert_to_tensor, share_memory

from .unified_tensor import UnifiedTensor


scope_lock = threading.Lock()


class DeviceGroup(object):
  r""" A group of GPUs with peer-to-peer access (NVLinks) to each other.

  Args:
    group_id: An integer representing the rank of the device group.
    device_list: A list of devices that can be accessed p2p.
  """
  def __init__(self, group_id: int, device_list: List[torch.device]):
    self.group_id = group_id
    self.device_list = device_list

  @property
  def size(self):
    return len(self.device_list)


class Feature(object):
  r""" A class for feature storage and lookup with hardware topology awareness
  and high performance.

  According to ``split_ratio``, ``Feature`` splits the feature data into the
  GPU part and the CPU part(ZERO-COPY), and the GPU part is replicated between
  all device groups in the input device group list. Each GPU can p2p access data
  on other GPUs in the same ``DeviceGroup`` it belongs to, and can access data
  on CPU part with zero copy.

  Args:
    feature_tensor (torch.Tensor or numpy.ndarray): A CPU tensor of the raw
      feature data.
    id2index (torch.Tensor, optional):: A tensor mapping the node id to the
      index in the raw cpu feature tensor. If the feature data in the input
      ``feature_tensor`` are not consecutive and ordered by node ids, this
      parameter should be provided. (Default: ``None``).
    split_ratio (float): The proportion of feature data allocated to the GPU,
      between 0 and 1.  (Default: ``0.0``).
    device_group_list (List[DeviceGroup], optional): A list of device groups
      used for feature lookups, the GPU part of feature data will be replicated
      on each device group in this list during the initialization. GPUs with
      peer-to-peer access to each other should be set in the same device group
      properly. Note that this parameter will be ignored if the ``split_ratio``
      set to zero. If set to ``None``, the GPU part will be replicated on all
      available GPUs got by ``torch.cuda.device_count()``, and each GPU device
      is an independent group. (Default: ``None``).
    device (int, optional): The target cuda device rank to perform feature
      lookups with the GPU part on the current ``Feature`` instance.
      The value of ``torch.cuda.current_device()`` will be used if set to
      ``None``(Default: ``None``).
    with_gpu (bool): A Boolean value indicating whether the ``Feature`` uses
      ``UnifiedTensor``. If True, it means ``Feature`` consists of
      ``UnifiedTensor``, otherwise ``Feature`` is PyTorch CPU Tensor and
      ``split_ratio``, ``device_group_list`` and ``device`` will be invalid.
      (Default: ``True``).
    dtype (torch.dtype): The data type of feature elements.
      (Default: ``torch.float32``).

  Example:
    >>> feat_tensor, id2index = sort_by_in_degree(feat_tensor, topo)
    >>> # suppose you have 8 GPUs.
    >>> # if there is no NVLink.
    >>> device_groups = [DeviceGroup(i, [i]) for i in range(8)]
    >>> # if there are NVLinks between GPU0-3 and GPU4-7.
    >>> device_groups = [DeviceGroup(0, [0,1,2,3]), DeviceGroup(1, [4,5,6,7])]
    >>> # Split the cpu feature tensor, of which the GPU part accounts for 60%.
    >>> # Launch the GPU kernel on device 0 for this ``Feature`` instance.
    >>> feature = Feature(feat_tensor, id2index, 0.6, device_groups, 0)
    >>> out = feature[input]

  TODO(baole): Support to automatically find suitable GPU groups. For now,
    you can use ``nvidia-smi topo -m`` to find the right groups.
  """
  def __init__(self,
               feature_tensor: TensorDataType,
               id2index: Optional[torch.Tensor] = None,
               split_ratio: float = 0.0,
               device_group_list: Optional[List[DeviceGroup]] = None,
               device: Optional[int] = None,
               with_gpu: Optional[bool] = True,
               dtype: torch.dtype = torch.float32):
    self.feature_tensor = convert_to_tensor(feature_tensor, dtype)
    self.id2index = convert_to_tensor(id2index, dtype=torch.int64)
    self.split_ratio = float(split_ratio)
    self.device_group_list = device_group_list
    self.device = device
    self.with_gpu = with_gpu
    self.dtype = dtype

    self._device2group = {}
    self._unified_tensors = {}
    self._cuda_id2index = None
    self._ipc_handle = None
    self._cuda_ipc_handle_dict = None

    if self.with_gpu:
      if self.feature_tensor is not None:
        self.feature_tensor = share_memory(self.feature_tensor.cpu())
      if self.device_group_list is None:
        self.device_group_list = [
          DeviceGroup(i, [i]) for i in range(torch.cuda.device_count())]

      self._device2group = {}
      group_size = self.device_group_list[0].size
      for dg in self.device_group_list:
        assert group_size == dg.size
        for d in dg.device_list:
          self._device2group[d] = dg.group_id
      if self.feature_tensor is not None:
        self._split_and_init()


  def __getitem__(self, ids: torch.Tensor):
    r""" Perform feature lookups with GPU part and CPU part.
    """
    if not self.with_gpu:
      return self.cpu_get(ids)
    self.lazy_init_with_ipc_handle()
    ids = ids.to(self.device)
    if self.id2index is not None:
      if self._cuda_id2index is None:
        self._cuda_id2index = self.id2index.to(self.device)
      ids = self._cuda_id2index[ids]
    group_id = self._device2group[self.device]
    unified_tensor = self._unified_tensors[group_id]
    return unified_tensor[ids]

  def cpu_get(self, ids: torch.Tensor):
    r""" Perform feature lookups only with CPU feature tensor.
    """
    self.lazy_init_with_ipc_handle()
    ids = ids.to('cpu')
    if self.id2index is not None:
      ids = self.id2index[ids]
    return self.feature_tensor[ids]

  def _check_and_set_device(self):
    if self.device is None:
      self.device = torch.cuda.current_device()
    else:
      self.device = int(self.device)
      assert (
        self.device >= 0 and self.device < torch.cuda.device_count()
      ), f"'{self.__class__.__name__}': invalid device rank {self.device}"

  def _split(self, feature_tensor: torch.Tensor):
    device_part_size = int(feature_tensor.shape[0] * self.split_ratio)
    return feature_tensor[:device_part_size], feature_tensor[device_part_size:]

  def _split_and_init(self):
    r""" Split cpu feature tensor and initialize GPU part and CPU part.
    """
    self._check_and_set_device()
    device_part, cpu_part = self._split(self.feature_tensor)

    if device_part.shape[0] > 0: # GPU part
      for group in self.device_group_list:
        block_size = device_part.shape[0] // group.size
        unified_tensor = UnifiedTensor(group.device_list[0], self.dtype)
        tensors, tensor_devices = [], []
        cur_pos = 0
        for idx, device in enumerate(group.device_list):
          if idx == group.size - 1:
            tensors.append(device_part[cur_pos:])
          else:
            tensors.append(device_part[cur_pos:cur_pos + block_size])
            cur_pos += block_size
          tensor_devices.append(device)
        unified_tensor.init_from(tensors, tensor_devices)
        self._unified_tensors[group.group_id] = unified_tensor

    if cpu_part.numel() > 0: # CPU part
      group_id = self._device2group[self.device]
      unified_tensor = self._unified_tensors.get(group_id, None)
      if unified_tensor is None:
        unified_tensor = UnifiedTensor(group_id, self.dtype)
      unified_tensor.append_cpu_tensor(cpu_part)
      self._unified_tensors[group_id] = unified_tensor

  def share_ipc(self):
    r""" Create ipc handle for multiprocessing.
    """
    if self._ipc_handle is not None:
      return self._ipc_handle

    if self.id2index is not None:
      self.id2index = self.id2index.cpu()
      self.id2index.share_memory_()

    if self._cuda_ipc_handle_dict is None:
      self._cuda_ipc_handle_dict = {}
      for group_id, tensor in self._unified_tensors.items():
        self._cuda_ipc_handle_dict[group_id] = tensor.share_ipc()[0]

    return (
      self.feature_tensor,
      self.id2index,
      self.split_ratio,
      self.device_group_list,
      self._cuda_ipc_handle_dict,
      self.with_gpu,
      self.dtype
    )

  @classmethod
  def from_ipc_handle(cls, ipc_handle):
    _, _, split_ratio, device_group_list, _, with_gpu, dtype = ipc_handle
    feature = cls(None, None, split_ratio, device_group_list,
                  with_gpu=with_gpu, dtype=dtype)
    feature._ipc_handle = ipc_handle
    return feature

  def lazy_init_with_ipc_handle(self):
    with scope_lock:
      if self._ipc_handle is None:
        return
      self.feature_tensor, self.id2index, _, _, self._cuda_ipc_handle_dict, _, _ \
        = self._ipc_handle
      if not self.with_gpu:
        self._ipc_handle = None
        return
      self._check_and_set_device()
      _, cpu_part = self._split(self.feature_tensor)
      group_id = self._device2group[self.device]
      self._unified_tensors[group_id] = UnifiedTensor.new_from_ipc(
        ipc_handles=(self._cuda_ipc_handle_dict.get(group_id, []), cpu_part),
        current_device=self.device,
        dtype=self.dtype
      )
      self._ipc_handle = None

  @property
  def shape(self):
    self.lazy_init_with_ipc_handle()
    return self.feature_tensor.shape

  def size(self, dim):
    self.lazy_init_with_ipc_handle()
    return self.feature_tensor.size(dim)


## Pickling Registration

def rebuild_feature(ipc_handle):
  feature = Feature.from_ipc_handle(ipc_handle)
  return feature

def reduce_feature(feature: Feature):
  ipc_handle = feature.share_ipc()
  return (rebuild_feature, (ipc_handle, ))

ForkingPickler.register(Feature, reduce_feature)
