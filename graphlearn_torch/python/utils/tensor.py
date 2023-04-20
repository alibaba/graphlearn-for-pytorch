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

from typing import Any, List, Union

import numpy
import torch


def tensor_equal_with_device(lhs: torch.Tensor, rhs: torch.Tensor):
  r""" Check whether the data and device of two tensors are same.
  """
  if lhs.device == rhs.device:
    return torch.equal(lhs, rhs)
  return False


def id2idx(ids: Union[List[int], torch.Tensor]):
  r""" Get tensor of mapping from id to its original index.
  """
  if not isinstance(ids, torch.Tensor):
    ids = torch.tensor(ids, dtype=torch.int64)
  ids = ids.to(torch.int64)
  max_id = torch.max(ids).item()
  id2idx = torch.zeros(max_id + 1, dtype=torch.int64, device=ids.device)
  id2idx[ids] = torch.arange(ids.size(0), dtype=torch.int64, device=ids.device)
  return id2idx


def convert_to_tensor(data: Any, dtype: torch.dtype = None):
  r""" Convert the input data to a tensor based type.
  """
  if isinstance(data, dict):
    new_data = {}
    for k, v in data.items():
      new_data[k] = convert_to_tensor(v, dtype)
    return new_data
  if isinstance(data, list):
    new_data = []
    for v in data:
      new_data.append(convert_to_tensor(v, dtype))
    return new_data
  if isinstance(data, tuple):
    return tuple(convert_to_tensor(list(data), dtype))
  if isinstance(data, torch.Tensor):
    return data.type(dtype) if dtype is not None else data
  if isinstance(data, numpy.ndarray):
    return (
      torch.from_numpy(data).type(dtype) if dtype is not None
      else torch.from_numpy(data)
    )
  return data


def apply_to_all_tensor(data: Any, tensor_method, *args, **kwargs):
  r""" Apply the specified method to all tensors contained by the
  input data recursively.
  """
  if isinstance(data, dict):
    new_data = {}
    for k, v in data.items():
      new_data[k] = apply_to_all_tensor(v, tensor_method, *args, **kwargs)
    return new_data
  if isinstance(data, list):
    new_data = []
    for v in data:
      new_data.append(apply_to_all_tensor(v, tensor_method, *args, **kwargs))
    return new_data
  if isinstance(data, tuple):
    return tuple(apply_to_all_tensor(list(data), tensor_method, *args, **kwargs))
  if isinstance(data, torch.Tensor):
    return tensor_method(data, *args, **kwargs)
  return data


def share_memory(data: Any):
  r""" Share memory for all tensors contained by the input data.
  """
  return apply_to_all_tensor(data, torch.Tensor.share_memory_)


def squeeze(data: Any):
  r""" Squeeze all tensors contained by the input data.
  """
  return apply_to_all_tensor(data, torch.Tensor.squeeze)
