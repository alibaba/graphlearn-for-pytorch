import os
import glob
import sys
import site
import sysconfig
import subprocess

from torch.utils.cpp_extension import CppExtension


def ext_module(
  name: str,
  root_path: str,
  with_cuda: bool = False,
  with_vineyard: bool = True,
  release: bool = False
):
  PYTHON_PKG_PATH = site.getsitepackages()[0]
  PYTHON_INCLUDE_PATH = sysconfig.get_paths()["include"]
  
  python_version = sys.version_info[:2]
  python_lib_name = f"python{python_version[0]}.{python_version[1]}" # e.g. python3.8

  include_dirs = []
  library_dirs = []
  libraries = [python_lib_name]
  extra_cxx_flags = []
  extra_link_args = []
  define_macros = []
  undef_macros = []
  
  # add third_party grpc libraries
  for file_path in glob.glob(f"{root_path}/third_party/grpc/build/lib/*.a"):
    libraries.append(file_path.split('/')[-1][3:-2])
  libraries.extend(['absl_time', 'absl_int128', 'absl_throw_delegate', 'absl_time_zone', 'upb', 'address_sorting', 'ssl', 'crypto', 'cares'])
  
  # generate proto files
  cmd = [
    sys.executable,
    os.path.join(
        root_path,
        "graphlearn_torch",
        "csrc",
        "rpc",
        "proto",
        "proto_generator.py",
    ),
    os.path.join(root_path, "graphlearn_torch", "csrc", "rpc", "generated"),
  ]
  subprocess.check_call(
      cmd,
      env=os.environ.copy(),
  )

  library_dirs.append(root_path + '/third_party/grpc/build/lib')
  library_dirs.append(PYTHON_PKG_PATH + '/torch/lib/')
  
  include_dirs.append(root_path)
  include_dirs.append(root_path + '/third_party/grpc/build/include')
  include_dirs.append(PYTHON_PKG_PATH + '/torch/include')
  include_dirs.append(PYTHON_PKG_PATH + '/torch/include/torch/csrc/api/include/')
  include_dirs.append(PYTHON_INCLUDE_PATH)
  
  if with_cuda:
    include_dirs.append('/usr/local/cuda' + '/include')
    library_dirs.append('/usr/local/cuda' + 'lib64')
  if with_vineyard:
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

  sources = [os.path.join(root_path, 'graphlearn_torch/python/py_export.cc')]

  sources += glob.glob(
    os.path.join(root_path, 'graphlearn_torch/csrc/**/**.cc'), recursive=True
  )

  if with_cuda:
    sources += glob.glob(
      os.path.join(root_path, 'graphlearn_torch/csrc/**/**.cu'), recursive=True
    )

  if with_vineyard:
    define_macros.append(('WITH_VINEYARD', 'ON'))
  else:
    undef_macros.append(('WITH_VINEYARD'))

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
