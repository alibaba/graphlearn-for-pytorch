# Installation

### Requirements
- cuda
- python>=3.6
- torch(PyTorch)
- torch_geometric, torch_scatter, torch_sparse. Please refer to [PyG](https://github.com/pyg-team/pytorch_geometric) for installation.
### Pip Wheels

```
# glibc>=2.14, torch>=1.13
pip install graphlearn-torch
```

### Build from source

#### Install Dependencies
```shell
git submodule update --init
./install_dependencies.sh
```

#### Python
1. Build
``` shell
python setup.py bdist_wheel
pip install dist/*
```
2. UT
``` shell
./scripts/run_python_ut.sh
```

#### C++
If you need to test C++ operations, you can only build the C++ part.

1. Build
``` shell
cmake .
make -j
```
2. UT
``` shell
./scripts/run_cpp_ut.sh
```
