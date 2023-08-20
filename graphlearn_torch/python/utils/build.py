import os
import glob
import sys
import site
import sysconfig
import subprocess
import multiprocessing

from torch.utils.cpp_extension import CppExtension

def ext_module(
  name: str,
  root_path: str,
  with_cuda: bool = False,
  with_vineyard: bool = True,
  release: bool = False,
  build_tests: bool = False,
  debug: bool = False
):
  # call cmake to build the library
  cmake_call = ['cmake', '-DCMAKE_CXX_FLAGS=-fPIC']
  
  cmake_call.append('-DWITH_CUDA=ON' if with_cuda else '-DWITH_CUDA=OFF')
  cmake_call.append('-DWITH_VINEYARD=ON' if with_vineyard else '-DWITH_VINEYARD=OFF')
  cmake_call.append('-DBUILD_TESTS=ON' if build_tests else '-DBUILD_TESTS=OFF')
  cmake_call.append('-DDEBUG=ON' if debug else '-DDEBUG=OFF')
  if with_cuda:
    cmake_call.append('-DCMAKE_CUDA_FLAGS=-Xcompiler=-fPIC')
  cmake_call.append(root_path)
  
  subprocess.check_call(  
    cmake_call,
    env=os.environ.copy(),
  )
  subprocess.check_call(
    ["cmake", "--build", root_path, "-j", str(multiprocessing.cpu_count())],
    env=os.environ.copy(),
  )
  print("cmake build done...")
  
  PYTHON_PKG_PATH = site.getsitepackages()[0]
  PYTHON_INCLUDE_PATH = sysconfig.get_paths()["include"]
  python_lib = PYTHON_INCLUDE_PATH.split('/')[-1]

  include_dirs = []
  library_dirs = ['built/lib']
  libraries = ['graphlearn_torch']
  extra_cxx_flags = []
  extra_link_args = ["-Wl,-rpath=$ORIGIN/built/lib"]
  define_macros = []
  undef_macros = []
  
  library_dirs.append(root_path + '/third_party/grpc/build/lib')
  library_dirs.append(PYTHON_PKG_PATH + '/torch/lib/')
  library_dirs.append('/usr/local/cuda' + 'lib64')
  
  include_dirs.append(root_path)
  include_dirs.append('/usr/local/cuda' + '/include')
  include_dirs.append(root_path + '/third_party/grpc/build/include')
  include_dirs.append(PYTHON_PKG_PATH + '/torch/include')
  include_dirs.append(PYTHON_PKG_PATH + '/torch/include/torch/csrc/api/include/')
  include_dirs.append(PYTHON_INCLUDE_PATH)
  

  extra_cxx_flags.append('-D_GLIBCXX_USE_CXX11_ABI=0')
  extra_cxx_flags.append('-std=gnu++14')
  extra_cxx_flags.append('-fPIC')

  sources = [os.path.join(root_path, 'graphlearn_torch/python/py_export.cc')]

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
    runtime_library_dirs=['built/lib']
  )
