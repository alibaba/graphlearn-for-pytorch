import os
import platform
import shutil
import subprocess

from torch.utils.cpp_extension import CppExtension

def glt_v6d_ext_module(
  name: str,
  root_path: str,
):
  include_dirs = []
  library_dirs = []
  libraries = []
  extra_cxx_flags = []

  include_dirs.append(root_path)
  # We assume that glt_v6d is built in graphscope environment

  if 'GRAPHSCOPE_HOME' in os.environ:
    include_dirs.append(os.environ['GRAPHSCOPE_HOME'] + '/include' + '/vineyard')
    include_dirs.append(os.environ['GRAPHSCOPE_HOME'] + '/include' + '/vineyard/contrib')
    include_dirs.append(os.environ['GRAPHSCOPE_HOME'] + '/include')
    library_dirs.append(os.environ['GRAPHSCOPE_HOME'] + '/lib')

  if platform.system() == 'Darwin':
    homebrew_prefix = subprocess.check_output([shutil.which('brew'), '--prefix']) \
                                .decode('utf-8', errors='ignore').strip()
    include_dirs.append(os.path.join(homebrew_prefix, 'include'))
    include_dirs.append(os.path.join(homebrew_prefix, 'include', 'vineyard'))
    include_dirs.append(os.path.join(homebrew_prefix, 'include', 'vineyard', 'contrib'))
    library_dirs.append(os.path.join(homebrew_prefix, 'lib'))

  if platform.system() == "Linux":
    include_dirs.append('/usr/lib/x86_64-linux-gnu/openmpi/include')
    library_dirs.append('/usr/local/lib')

  libraries.append('pthread')
  libraries.append('mpi')
  libraries.append('glog')
  libraries.append('vineyard_basic')
  libraries.append('vineyard_client')
  libraries.append('vineyard_graph')
  libraries.append('vineyard_io')

  extra_cxx_flags.append('-std=c++17')

  sources = [os.path.join(root_path, 'graphlearn_torch/python/py_export_v6d.cc')]

  import glob
  sources += glob.glob(
    os.path.join(root_path, 'graphlearn_torch/v6d/**.cc'), recursive=True
  )
  extra_link_args = []
  define_macros = [('WITH_VINEYARD', 'ON')]
  undef_macros = []
  return CppExtension(
    name,
    sources,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={
      'cxx': extra_cxx_flags,
    },
    define_macros=define_macros,
    undef_macros=undef_macros,
  )

def glt_ext_module(
  name: str,
  root_path: str,
  with_cuda: bool = False,
  release: bool = False
):
  include_dirs = []
  library_dirs = []
  libraries = []
  extra_cxx_flags = []
  extra_link_args = []
  define_macros = []
  undef_macros = []

  include_dirs.append(root_path)
  if with_cuda:
    include_dirs.append('/usr/local/cuda' + '/include')
    library_dirs.append('/usr/local/cuda' + 'lib64')

  extra_cxx_flags.append('-std=c++17')

  sources = [os.path.join(root_path, 'graphlearn_torch/python/py_export_glt.cc')]

  import glob
  sources += glob.glob(
    os.path.join(root_path, 'graphlearn_torch/csrc/**/**.cc'), recursive=True
  )

  if with_cuda:
    sources += glob.glob(
      os.path.join(root_path, 'graphlearn_torch/csrc/**/**.cu'), recursive=True
    )

  if with_cuda:
    define_macros.append(('WITH_CUDA', 'ON'))
  else:
    undef_macros.append(('WITH_CUDA'))

  if release:
    nvcc_flags = [
      '-O3', '--expt-extended-lambda', '-lnuma', '-arch=sm_50',
      '-gencode=arch=compute_50,code=sm_50',
      '-gencode=arch=compute_52,code=sm_52',
      '-gencode=arch=compute_60,code=sm_60',
      '-gencode=arch=compute_61,code=sm_61',
      '-gencode=arch=compute_70,code=sm_70',
      '-gencode=arch=compute_75,code=sm_75',
      '-gencode=arch=compute_75,code=compute_75'
    ]
  else:
    nvcc_flags = ['-O3', '--expt-extended-lambda', '-lnuma']
  return CppExtension(
    name,
    sources,
    extra_link_args=extra_link_args,
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args={
      'cxx': extra_cxx_flags,
      'nvcc': nvcc_flags,
    },
    define_macros=define_macros,
    undef_macros=undef_macros,
  )
