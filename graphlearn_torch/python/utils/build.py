import os

from torch.utils.cpp_extension import CppExtension


def ext_module(
  name: str,
  root_path: str,
  with_cuda: bool = False,
  with_vineyard: bool = True,
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

  import glob
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
