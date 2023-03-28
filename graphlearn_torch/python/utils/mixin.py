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

class CastMixin:
  r""" This class is same as PyG's :class:`~torch_geometric.utils.CastMixin`:
  https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mixin.py
  """
  @classmethod
  def cast(cls, *args, **kwargs):
    if len(args) == 1 and len(kwargs) == 0:
      elem = args[0]
      if elem is None:
        return None
      if isinstance(elem, CastMixin):
        return elem
      if isinstance(elem, (tuple, list)):
        return cls(*elem)
      if isinstance(elem, dict):
        return cls(**elem)
    return cls(*args, **kwargs)
