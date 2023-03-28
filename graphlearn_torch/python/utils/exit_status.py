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

import atexit


python_exit_status = False
r"""Whether Python is shutting down. This flag is guaranteed to be set before
the Python core library resources are freed, but Python may already be exiting
for some time when this is set.

Hook to set this flag is `_set_python_exit_flag`, and is same as used in
Pytorch's dataloader:
https://github.com/pytorch/pytorch/blob/f1a6f32b72b7c2b73277f89bbf7e7459a400d80a/torch/utils/data/_utils/__init__.py
"""

def _set_python_exit_flag():
  global python_exit_status
  python_exit_status = True

atexit.register(_set_python_exit_flag)
