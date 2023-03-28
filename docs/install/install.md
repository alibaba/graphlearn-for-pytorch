# Installation

### Requirements
- Python>=3.6
- PyTorch
- [PyG](https://github.com/pyg-team/pytorch_geometric)
### Pip Wheels

```
pip install graphlearn-torch
```

### Build from source

#### C++
1. Build
``` shell
git submodule update --init
sh install_dependencies.sh
cmake .
make -j
```
2. UT
``` shell
sh test_cpp_ut.sh
```

#### Python
1. Build
``` shell
python setup.py bdist_wheel
pip install dist/*
```
2. UT
``` shell
sh test_python_ut.sh
```
