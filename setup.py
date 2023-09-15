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
import torch
import subprocess
import re

# This version string should be updated when releasing a new version.
_VERSION = '0.2.1'

RELEASE = os.getenv("RELEASE", "FALSE")
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
WITH_VINEYARD = os.getenv('WITH_VINEYARD', 'OFF')
WITH_CUDA = os.getenv('WITH_CUDA', 'ON')

sys.path.append(os.path.join(ROOT_PATH, 'graphlearn_torch', 'python', 'utils'))
from build import glt_ext_module, glt_v6d_ext_module

GLT_V6D_EXT_NAME = "py_graphlearn_torch_vineyard"
GLT_EXT_NAME = "py_graphlearn_torch"

def get_gcc_use_cxx_abi():
    output = subprocess.run("cmake .", capture_output=True, text=True, shell=True)
    print('output', str(output))
    match = re.search(r"GCC_USE_CXX11_ABI: (\d)", str(output))
    if match:
        return match.group(1)
    else:
        return None
  
GCC_USE_CXX11_ABI = get_gcc_use_cxx_abi()

class CustomizedBuildExtension(BuildExtension):
  def _add_gnu_cpp_abi_flag(self, extension):
    gcc_use_cxx_abi = GCC_USE_CXX11_ABI if extension.name == GLT_V6D_EXT_NAME else str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    print('GCC_USE_CXX11_ABI for {}: {}', extension.name, gcc_use_cxx_abi)
    self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + gcc_use_cxx_abi) 
        

ext_modules = [
  glt_ext_module(
    name=GLT_EXT_NAME,
    root_path=ROOT_PATH,
    with_cuda=WITH_CUDA == "ON",
    release=RELEASE == "TRUE"
  )
]

if WITH_VINEYARD == "ON":
  ext_modules.append(
    glt_v6d_ext_module(
      name=GLT_V6D_EXT_NAME,
      root_path=ROOT_PATH,      
    ),
  )

setup(
  name='graphlearn-torch',
  version=_VERSION,
  author='GLT Team',
  description='Graph Learning for PyTorch (GraphLearn-for-PyTorch)',
  url="https://github.com/alibaba/graphlearn-for-pytorch",
  python_requires='>=3.6',
  requires=['torch'],
  cmdclass={'build_ext': CustomizedBuildExtension},
  ext_package='graphlearn_torch',
  ext_modules=ext_modules,
  package_dir={'graphlearn_torch': 'graphlearn_torch/python'},
  packages=[
    'graphlearn_torch', 'graphlearn_torch.channel', 'graphlearn_torch.data',
    'graphlearn_torch.distributed', 'graphlearn_torch.loader',
    'graphlearn_torch.partition', 'graphlearn_torch.sampler',
    'graphlearn_torch.utils'
  ]
)
