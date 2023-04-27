# RGNN examples on IGBH.

- RGCN Paper: https://arxiv.org/abs/1703.06103
- IGBH Dataset: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/tree/main

We implement RGNN models(RGAT and RSAGE) based on the RGCN paper, where SAGEConv and RGATConv are utilized to enable the training of heterogeneous graphs.
We use the IGBH dataset to provide both single-node and multi-node distributed training, utilizing GPU sampling and training as well as CPU-only sampling and training examples.

## 1. Single node and single GPU training:
```
python train_rgnn.py --dataset_size='tiny'
```
add `--cpu_mode` if use CPU only.

## 2. Distributed (multi nodes) examples.

We use 2 nodes as an example.
### 2.1 Prepare data.
```
python partition.py --dataset_size='tiny' --num_partitions=2
```

### 2.2 Example of distributed training
2 nodes each with 2 GPUs
```
# node 0:
CUDA_VISIBLE_DEVICES=0,1 python dist_train_rgnn.py --num_nodes=2 --node_rank=0 --master_addr=localhost

# node 1:
CUDA_VISIBLE_DEVICES=2,3 python dist_train_rgnn.py --num_nodes=2 --node_rank=1 --master_addr=localhost
```
add `--cpu_mode` if use CPU only.

Note: you should change master_addr to the ip of node#0.
