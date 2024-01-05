# RGNN examples on IGBH

- RGCN Paper: https://arxiv.org/abs/1703.06103
- IGBH Dataset: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/tree/main

We implement RGNN models(RGAT and RSAGE) based on the RGCN paper, where SAGEConv and RGATConv are utilized to enable the training of heterogeneous graphs.
We use the IGBH dataset to provide both single-node and multi-node distributed training, utilizing GPU sampling and training as well as CPU-only sampling and training examples.


## 0. Reference test environment and dependency packages
docker image: pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install --no-index  torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install graphlearn-torch
```
Please refer to the [documentation](../../README.md#installation) for installation and build details.

## 1. Dataset Preparation
If you want download the IGBH-large or IGBH600M, i.e. `dataset_size` is 'large' or 'full',
please use the following script:
```
bash download_igbh_large.sh
# bash download_igbh_full.sh
```

For the `tiny`, `small` or `medium` dataset, the download procedure is included
in the training script below. Note that in `dataset.py`, we have converted the graph i
nto an undirected graph.

## 2. Single node training:
```
python train_rgnn.py --model='rgat' --dataset_size='tiny' --num_classes=19
```
The script uses a single GPU, please add `--cpu_mode` if you want to use CPU only. 
To save the memory costs while training large datasets, add `--use_fp16` to load
feature data in FP16 format. Option `--pin_feature` decides if the feature data will be
pinned in host memory, which enables zero-copy feature access from GPU but will 
incur extra memory costs.

To train the model using multiple GPUs and FP16 format wihtout pinning the feature:
```
CUDA_VISIBLE_DEVICES=0,1 python train_rgnn_multi_gpu.py --model='rgat' --dataset_size='tiny' --num_classes=19 --use_fp16
```

Note that the original graph is in `COO` format, the above scripts will transform
the graph from `COO` to `CSC` or `CSR` according to the edge direction of sampling. 
This process could be time consuming. We provide a script to convert the graph layout
from `COO` to `CSC/CSR` and persist the feature in FP16 format:

```
python compress_graph.py --dataset_size='tiny' --layout='CSC' --use_fp16
```

Once the conversion is completed, train the model:

```
CUDA_VISIBLE_DEVICES=0,1 python train_rgnn_multi_gpu.py --model='rgat' --dataset_size='tiny' --num_classes=19 --use_fp16 --layout='CSC'
```

Note that, when the sampling edge direction is `in`, the layout should be `CSC`. 
When the sampling edge direction is `out`, the layout should be `CSR`.


## 3. Distributed (multi nodes) examples

We use 2 nodes as an example.

### 3.1 Data partitioning

To partition the dataset (including both the topology and feature):
```
python partition.py --dataset_size='tiny' --num_partitions=2 --num_classes=19
```

GLT also supports two-stage partitioning, which splits the process of topology 
partitioning and feature partitioning. After the topology partitioning is executed,
the feature partitioning process can be conducted in each training node in parallel 
to speedup the partitioning process.

The topology partitioning is conducted by setting  `--with_feature=0`:
```
python partition.py --dataset_size='tiny' --num_partitions=2 --num_classes=19 --with_feature=0
```

By default the layout of partitioned graph is in the `COO` format, `CSC` and `CSR` are also
supported by setting `--layout` for `partition.py`.


The feature partitioning in conducted in each training node:
```
# node 0 which holds partition 0:
python build_partition_feature.py --dataset_size='tiny' --in_memory=0 --partition_idx=0

# node 1 which holds partition 1:
python build_partition_feature.py --dataset_size='tiny' --in_memory=0 --partition_idx=1
```
Building partition feature with `--use_fp16` will convert the data type of feature
from FP32 into FP16.

### 3.2 Example of distributed training
2 nodes each with 2 GPUs
```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --num_training_procs=2 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --num_training_procs=2 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19
```
The script uses GPU default, please add `--cpu_mode` if you want to use CPU only.

To seperate the GPU used by sampling and training processes, please add `--split_training_sampling` and set `--num_training_procs` as half of the number of devices:

```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --num_training_procs=1 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19 --split_training_sampling

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --num_training_procs=1 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19 --split_training_sampling
```
The script uses one GPU for training and another GPU for sampling in each node. 

Note:
- The `num_partitions` and `num_nodes` must be the same.
- You should change master_addr to the ip of node#0.
