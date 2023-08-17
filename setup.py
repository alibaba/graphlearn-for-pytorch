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
# =============================================================================

import os
import sys
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup

# This version string should be updated when releasing a new version.
_VERSION = '0.2.1'

RELEASE = os.getenv("RELEASE", "FALSE")
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
WITH_VINEYARD = os.getenv('WITH_VINEYARD', 'OFF')
WITH_CUDA = os.getenv('WITH_CUDA', 'ON')
BUILD_TESTS = os.getenv('BUILD_TESTS', 'ON')
DEBUG = os.getenv('DEBUG', 'OFF')

sys.path.append(os.path.join(ROOT_PATH, 'graphlearn_torch', 'python', 'utils'))
from build import ext_module

setup(
  name='graphlearn-torch',
  version=_VERSION,
  author='GLT Team',
  description='Graph Learning for PyTorch (GraphLearn-for-PyTorch)',
  url="https://github.com/alibaba/graphlearn-for-pytorch",
  python_requires='>=3.6',
  requires=['torch'],
  cmdclass={'build_ext': BuildExtension},
  ext_package='graphlearn_torch',
  ext_modules=[
    ext_module(
      name='py_graphlearn_torch',
      root_path=ROOT_PATH,
      with_cuda=WITH_CUDA == "ON",
      with_vineyard=WITH_VINEYARD == "ON",
      release=RELEASE == "TRUE",
      build_tests=BUILD_TESTS == "ON",
      debug=DEBUG == "ON"
    )
  ],
  package_dir={'graphlearn_torch': 'graphlearn_torch/python'},
  packages=[
    'graphlearn_torch', 'graphlearn_torch.channel', 'graphlearn_torch.data',
    'graphlearn_torch.distributed', 'graphlearn_torch.loader',
    'graphlearn_torch.partition', 'graphlearn_torch.sampler',
    'graphlearn_torch.utils'
  ]
)