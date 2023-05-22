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

import glob
import os

from setuptools import setup, find_packages
from torch.utils import cpp_extension

# This version string should be updated when releasing a new version.
_VERSION = '0.2.0'

RELEASE = os.getenv("RELEASE", "FALSE")
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
WITH_VINEYARD = os.getenv('WITH_VINEYARD', 'OFF')
WITH_CUDA = os.getenv('WITH_CUDA', 'ON')

extensions = []
include_dirs=[]
library_dirs=[]
libraries=[]
extra_cxx_flags=[]
extra_link_args=[]
define_macros=[]
undef_macros=[]

include_dirs.append(ROOT_PATH)
include_dirs.append('/usr/local/cuda' + '/include')
library_dirs.append('/usr/local/cuda' + 'lib64')
if WITH_VINEYARD == 'ON':
  include_dirs.append('/usr/local/include')
  include_dirs.append('/usr/local/include' + '/glog')
  include_dirs.append('/usr/local/include' + '/gflags')
  include_dirs.append('/usr/local/include' + '/vineyard')
  include_dirs.append('/usr/local/include' + '/vineyard/contrib')
  include_dirs.append('/home/pai/include')

  library_dirs.append('/usr/local/lib')
  library_dirs.append('/home/pai/lib')

  libraries.append('pthread')
  libraries.append('mpi')
  libraries.append('glog')
  libraries.append('vineyard_basic')
  libraries.append('vineyard_client')
  libraries.append('vineyard_graph')
  libraries.append('vineyard_io')


extra_cxx_flags.append('-D_GLIBCXX_USE_CXX11_ABI=0')
extra_cxx_flags.append('-std=gnu++14')

sources = ['graphlearn_torch/python/py_export.cc']
sources += glob.glob('graphlearn_torch/csrc/**/**.cc', recursive=True)

if WITH_CUDA == 'ON':
  sources += glob.glob('graphlearn_torch/csrc/**/**.cu', recursive=True)

if WITH_VINEYARD == 'ON':
  define_macros.append(('WITH_VINEYARD', 'ON'))
else:
  undef_macros.append(('WITH_VINEYARD'))

if WITH_CUDA == 'ON':
  define_macros.append(('WITH_CUDA', 'ON'))
else:
  undef_macros.append(('WITH_CUDA'))

if RELEASE == 'TRUE':
  nvcc_flags = ['-O3', '--expt-extended-lambda', '-lnuma',
                '-arch=sm_50',
                '-gencode=arch=compute_50,code=sm_50',
                '-gencode=arch=compute_52,code=sm_52',
                '-gencode=arch=compute_60,code=sm_60',
                '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70',
                '-gencode=arch=compute_75,code=sm_75',
                '-gencode=arch=compute_75,code=compute_75']
else:
  nvcc_flags = ['-O3', '--expt-extended-lambda', '-lnuma']
extensions.append(cpp_extension.CppExtension(
  'py_graphlearn_torch',
  sources,
  extra_link_args=extra_link_args,
  include_dirs=include_dirs,
  library_dirs=library_dirs,
  libraries = libraries,
  extra_compile_args={
    'cxx': extra_cxx_flags,
    'nvcc': nvcc_flags,
  },
  define_macros=define_macros,
  undef_macros=undef_macros,
))


setup(
  name='graphlearn-torch',
  version=_VERSION,
  author='Baole Ai',
  description='Graph Learning for PyTorch (GraphLearn-for-PyTorch)',
  url="https://github.com/alibaba/graphlearn-for-pytorch",
  python_requires='>=3.6',
  requires=['torch'],
  cmdclass={'build_ext': cpp_extension.BuildExtension},
  ext_package='graphlearn_torch',
  ext_modules=extensions,
  package_dir={'graphlearn_torch' : 'graphlearn_torch/python'},
  packages=[
    'graphlearn_torch',
    'graphlearn_torch.channel',
    'graphlearn_torch.data',
    'graphlearn_torch.distributed',
    'graphlearn_torch.loader',
    'graphlearn_torch.partition',
    'graphlearn_torch.sampler',
    'graphlearn_torch.utils'
  ]
)
