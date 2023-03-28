# GNN examples

1. Multi-processing Feature usage.

```
python feature_mp.py
```

2. Single GPU basic GraphSAGE on OGBN-Products similar to PyG.
This example is implemeted using `NeighborSampler`.

```
python train_sage_ogbn_products.py
```

3. Multiple GPUs GraphSAGE on OGBN-papers100M.
```
python multi_gpu/train_sage_ogbn_papers100m.py
```

4. Heterogeneous GraphSage examples on OGBN-MAG compatible with PyG.

```
# single GPU
python hetero/train_hgt_mag.py
# multi-GPUs
python hetero/train_hgt_mag_mp.py
```

5. Training on PAI.

see pai/README.md

6. Distributed (multi nodes) examples.
see distributed/README.md