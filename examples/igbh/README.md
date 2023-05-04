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
Please refer to the [documentation](../../README.md#installation) for installation details.

## 1. Dataset Preparation
If you want download the IGBH-large or IGBH600M, i.e. `dataset_size` is 'large' or 'full',
please use the following script:
```
bash download_igbh_large.sh
# bash download_igbh_full.sh
```

For the `tiny`, `small` or `medium` dataset, the download procedure is included
in the training script below.

Note that in `dataset.py`, we have converted the graph into an undirected graph.

## 2. Single node and single GPU training:
```
python train_rgnn.py --dataset_size='tiny' --model='gat' --data_sze='tiny'--num_classes=19
```
The script uses GPU default, please add `--cpu_mode` if you want to use CPU only.

## 3. Distributed (multi nodes) examples

We use 2 nodes as an example.
### 3.1 Data partitioning
```
python partition.py --dataset_size='tiny' --num_partitions=2 --num_classes=19
```

### 3.2 Example of distributed training
2 nodes each with 2 GPUs
```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --num_training_procs=2 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --num_training_procs=2 --master_addr=localhost --model='rgat' --dataset_size='tiny' --num_classes=19
```
The script uses GPU default, please add `--cpu_mode` if you want to use CPU only.

Note:
- The `num_partitions` and `num_nodes` must be the same.
- You should change master_addr to the ip of node#0.
