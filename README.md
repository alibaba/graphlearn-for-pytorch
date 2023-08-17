[![GLT-pypi](https://img.shields.io/pypi/v/graphlearn-torch.svg)](https://pypi.org/project/graphlearn-torch/)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://graphlearn-torch.readthedocs.io/en/latest/)
[![GLT CI](https://github.com/alibaba/graphlearn-for-pytorch/workflows/GLT%20CI/badge.svg)](https://github.com/alibaba/graphlearn-for-pytorch/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/graphlearn-for-pytorch/blob/main/LICENSE)

**GraphLearn-for-PyTorch(GLT)** is a graph learning library for PyTorch that makes
distributed GNN training and inference easy and efficient. It leverages the
power of GPUs to accelerate graph sampling and utilizes UVA to reduce the
conversion and copying of features of vertices and edges. For large-scale graphs,
it supports distributed training on multiple GPUs or multiple machines through
fast distributed sampling and feature lookup. Additionally, it provides flexible
deployment for distributed training to meet different requirements.


- [Highlighted Features](#highlighted-features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Pip Wheels](#pip-wheels)
  - [Build from source](#build-from-source)
    - [Install Dependencies](#install-dependencies)
    - [Python](#python)
    - [C++](#c)
- [Quick Tour](#quick-tour)
  - [Accelarating PyG model training on a single GPU.](#accelarating-pyg-model-training-on-a-single-gpu)
  - [Distributed training](#distributed-training)
- [License](#license)

## Highlighted Features
* **GPU acceleration**

  GLT provides both CPU-based and GPU-based graph operators such
  as neighbor sampling, negative sampling, and feature lookup. For GPU training,
  GPU-based graph operations accelerate the computation and reduce data movement
  by a considerable amount.

* **Scalable and efficient distributed training**

  For distributed training, we implement multi-processing asynchronous sampling,
  pin memory buffer, hot feature cache, and use fast networking
  technologies (PyTorch RPC with RDMA support) to speed up distributed sampling
  and reduce communication. As a result, GLT can achieve high
  scalability and support graphs with billions of edges.

* **Easy-to-use API**

  Most of the APIs of GLT are compatible with PyG/PyTorch,
  so you can only need to modify a few lines of PyG's code to get the
  acceleration of the program. For GLT specific APIs,
  they are compatible with PyTorch and there is complete documentation
  and usage examples available.

* **Large-scale real-world GNN models**

  We focus on real-world scenarios and provide distributed GNN training examples
  on large-scale graphs. Since GLT is compatible with PyG,
  you can use almost any PyG's model as the base model. We will also continue to
  provide models with practical effects in industrial scenarios.

* **Easy to extend**

  GLT directly uses PyTorch C++ Tensors and is easy to extend just
  like PyTorch. There are no extra restrictions for CPU or CUDA based graph
  operators, and adding a new one is straightforward. For distributed
  operations, you can write a new one in Python using PyTorch RPC.

* **Flexible deployment**

  Graph Engine(Graph operators) and PyTorch engine(PyTorch nn modules) can be
  deployed either co-located or separated on different machines. This
  flexibility enables you to deploy GLT to your own environment
  or embed it in your project easily.


## Architecture Overview
<p align="center">
  <img width="60%" src=docs/figures/arch.png />
</p>


The main goal of GLT is to leverage hardware resources like GPU/NVLink/RDMA and
characteristics of GNN models to accelerate end-to-end GNN training in both
the single-machine and distributed environments.

In the case of multi-GPU training, graph sampling and CPU-GPU data transferring
could easily become the major performance bottleneck. To speed up graph sampling
and feature lookup, GLT implements the Unified Tensor Storage to unify the
memory management of CPU and GPU. Based on this storage, GLT supports
both CPU-based and GPU-based graph operators such as neighbor sampling,
negative sampling, feature lookup, subgraph sampling etc.
To alleviate the CPU-GPU data transferring overheads incurred by feature collection,
GLT supports caching features of hot vertices in GPU memory,
and accessing the remaining feature data (stored in pinned memory) via UVA.
We further utilize the high-speed NVLink between GPUs expand the capacity of
GPU cache.

As for distributed training, to prevent remote data access from blocking
the progress of model training, GLT implements an efficient RPC framework on top
of PyTorch RPC and adopts asynchronous graph sampling and feature lookup operations
to hide the network latency and boost the end-to-end training throughput.

To lower the learning curve for PyG users,
the APIs of key abstractions in GLT, such as dataset and dataloader,
are designed to be compatible with PyG. Thus PyG users can
take full advantage of GLT's acceleration capabilities by only modifying
very few lines of code.

For model training, GLT supports different models to fit different scales of
real-world graphs. It allows users to collocate model training  and graph
sampling (including feature lookup) in the same process, or separate them into
different processes or even different machines.
We provide two example to illustrate the training process on
small graphs: [single GPU training example](examples/train_sage_ogbn_products.py)
and [multi-GPU training example](examples/multi_gpu/). For large-scale graphs,
GLT separates sampling and training processes for
asynchronous and parallel acceleration, and supports deployment of sampling
and training processes on the same or different machines. Examples of
distributed training can be found in [distributed examples](examples/distributed/).

## Installation

### Requirements
- cuda
- python>=3.6
- torch(PyTorch)
- torch_geometric, torch_scatter, torch_sparse. Please refer to [PyG](https://github.com/pyg-team/pytorch_geometric) for installation.
- grpcio, grpcio-tools (e.g. pip install grpcio grpcio-tools)
### Pip Wheels

```
# glibc>=2.14, torch>=1.13
pip install graphlearn-torch
```

### Build from source

#### Install Dependencies
```shell
git submodule update --init
sh install_dependencies.sh
```

#### Python
1. Build
``` shell
python setup.py bdist_wheel
pip install dist/*
```

Build in CPU-mode 
``` shell
WITH_CUDA=OFF python setup.py bdist_wheel
pip install dist/*
```

2. UT
``` shell
sh scripts/run_python_ut.sh
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
sh scripts/run_cpp_ut.sh
```
## Quick Tour

### Accelarating PyG model training on a single GPU.

Let's take PyG's [GraphSAGE on OGBN-Products](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_sage.py)
as an example, you only need to replace PyG's `torch_geometric.loader.NeighborSampler`
by the [`graphlearn_torch.loader.NeighborLoader`](graphlearn_torch.loader.NeighborLoader)
to benefit from the the acceleration of model training using GLT.

```python
import torch
import graphlearn_torch as glt
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset

# PyG's original code preparing the ogbn-products dataset
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
data = dataset[0]

# Enable GLT acceleration on PyG requires only replacing
# PyG's NeighborSampler with the following code.
glt_dataset = glt.data.Dataset()
glt_dataset.build(edge_index=data.edge_index,
                  feature_data=data.x,
                  sort_func=glt.data.sort_by_in_degree,
                  split_ratio=0.2,
                  label=data.y,
                  device=0)
train_loader = glt.loader.NeighborLoader(glt_dataset,
                                         [15, 10, 5],
                                         split_idx['train'],
                                         batch_size=1024,
                                         shuffle=True,
                                         drop_last=True,
                                         as_pyg_v1=True)
```

The complete example can be found in [`examples/train_sage_ogbn_products.py`](examples/train_sage_ogbn_products.py).

<details>

While building the `glt_dataset`, the GPU where the graph sampling operations
are performed is specified by parameter `device`. By default, the graph topology are stored
in pinned memory for ZERO-COPY access. Users can also choose to stored the graph
topology in GPU by specifying `graph_mode='CUDA` in [`graphlearn_torch.data.Dataset.build`](graphlearn_torch.data.Dataset.build).
The `split_ratio` determines the fraction of feature data to be cached in GPU.
By default, GLT sorts the vertices in descending order according to vertex indegree
and selects vetices with higher indegree for feature caching. The default sort
function used as the input parameter for
[`graphlearn_torch.data.Dataset.build`](graphlearn_torch.data.Dataset.build) is
[`graphlearn_torch.data.reorder.sort_by_in_degree`](graphlearn_torch.data.reorder.sort_by_in_degree). Users can also customize their own sort functions with compatible APIs.
</details>

### Distributed training

For PyTorch DDP distributed training, there are usually several steps as follows:

First, load the graph and feature from partitions.
```python
import torch
import os.path as osp
import graphlearn_torch as glt

# load from partitions and create distributed dataset.
# Partitions are generated by following script:
# `python partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2`

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data', 'products')
glt_dataset = glt.distributed.DistDataset()
glt_dataset.load(
  num_partitions=2,
  partition_idx=int(os.environ['RANK']),
  graph_dir=osp.join(root, 'ogbn-products-graph-partitions'),
  feature_dir=osp.join(root, 'ogbn-products-feature-partitions'),
  label_file=osp.join(root, 'ogbn-products-label', 'label.pt') # whole label
)
train_idx = torch.load(osp.join(root, 'ogbn-products-train-partitions',
                                'partition' + str(os.environ['RANK']) + '.pt'))
```

Second, create distributed neighbor loader based on the dataset above.
```python
# distributed neighbor loader
train_loader = glt.distributed.DistNeighborLoader(
  data=glt_dataset,
  num_neighbors=[15, 10, 5],
  input_nodes=train_idx,
  batch_size=batch_size,
  drop_last=True,
  collect_features=True,
  to_device=torch.device(rank % torch.cuda.device_count()),
  worker_options=glt.distributed.MpDistSamplingWorkerOptions(
    num_workers=nsampling_proc_per_train,
    worker_devices=[torch.device('cuda', (i + rank) % torch.cuda.device_count())
                    for i in range(nsampling_proc_per_train)],
    worker_concurrency=4,
    master_addr='localhost',
    master_port=12345, # different from port in pytorch training.
    channel_size='2GB',
    pin_memory=True
  )
)
```

Finally, define DDP model and run.
```python
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import GraphSAGE

# DDP model
model = GraphSAGE(
  in_channels=num_features,
  hidden_channels=256,
  num_layers=3,
  out_channels=num_classes,
).to(rank)
model = DistributedDataParallel(model, device_ids=[rank])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# training.
for epoch in range(0, epochs):
  model.train()
  for batch in train_loader:
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index)[:batch.batch_size].log_softmax(dim=-1)
    loss = F.nll_loss(out, batch.y[:batch.batch_size])
    loss.backward()
    optimizer.step()
  dist.barrier()
```

The training scripts for 2 nodes each with 2 GPUs are as follows:
```shell
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use_env --nnodes=2 --node_rank=0 --master_addr=xxx dist_train_sage_supervised.py

# node 1:
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --use_env --nnodes=2 --node_rank=1 --master_addr=xxx dist_train_sage_supervised.py
```

Full code can be found in [distributed training example](examples/distributed/dist_train_sage_supervised.py).

## License
[Apache License 2.0](LICENSE)