## Run on PAI with MaxCompute(ODPS) table as input.


1. Prepare files
```
tar -zcvf train_products_sage.tar.gz *.py requirements.txt
```


2. Input
```
  ogbn_products_node(id bigint, feature string)
  ogbn_products_edge(src_id bigint, dst_id bigint)
  ogbn_products_train(id bigint, label bigint)
```
You can first generate text files use
```
python data_preprocess.py
```

and then use tunnel to upload text files to ODPS Tables
```
create table if not exists ogbn_products_node(id bigint, feature string);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_node ogbn_products_node -fd "\t";
create table if not exists ogbn_products_edge(src_id bigint, dst_id bigint);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_edge ogbn_products_edge -fd "\t";
create table if not exists ogbn_products_train(id bigint, label bigint);
tunnel u /{your_path}/graphlearn-for-pytorch/data/products/ogbn_products_train ogbn_products_train -fd "\t";
```


3. PAI command
- single GPU

```
pai -name pytorch112z -Dscript='file:///{your_path}/graphlearn-for-pytorch/examples/pai/train_products_sage.tar.gz' -DentryFile='train_products_sage.py' -Dtables="odps://{your_project}/tables/ogbn_products_node,odps://{your_project}/tables/ogbn_products_edge,odps://{your_project}/tables/ogbn_products_train" -Dcluster="{\"worker\":{\"gpu\":100}}" -DworkerCount=1 -DuserDefinedParameters='--split_ratio=0.2';
```

- 4 GPUs example

```
pai -name pytorch112z -Dscript='file:///{your_path}/graphlearn-for-pytorch/examples/pai/train_products_sage.tar.gz' -DentryFile='train_products_sage_mp.py' -Dtables="odps://{your_project}/tables/ogbn_products_node,odps://{your_project}/tables/ogbn_products_edge,odps://{your_project}/tables/ogbn_products_train" -Dcluster="{\"worker\":{\"gpu\":400}}" -DworkerCount=1 -DuserDefinedParameters='--split_ratio=0.8';
```

Note if there is no NVLink between your 4 GPUs, please remove `device_group_list` argument in glt.data.TableDataset.