#!/bin/bash

#https://github.com/IllinoisGraphBenchmark/IGB-Datasets/blob/main/igb/download_igbh600m.sh
echo "IGBH600M download starting"
cd ../../data/
mkdir -p igbh/full/processed
cd igbh/full/processed

# paper
mkdir paper
cd paper
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy
cd ..

# paper__cites__paper
mkdir paper__cites__paper
cd paper__cites__paper
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__cites__paper/edge_index.npy
cd ..

# author
mkdir author
cd author
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author/author_id_index_mapping.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author/node_feat.npy
cd ..

# conference
mkdir conference
cd conference
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/conference/conference_id_index_mapping.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/conference/node_feat.npy
cd ..

# institute
mkdir institute
cd institute
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/institute/institute_id_index_mapping.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/institute/node_feat.npy
cd ..

# journal
mkdir journal
cd journal
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/journal/journal_id_index_mapping.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/journal/node_feat.npy
cd ..

# fos
mkdir fos
cd fos
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/fos/fos_id_index_mapping.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/fos/node_feat.npy
cd ..

# author__affiliated_to__institute
mkdir author__affiliated_to__institute
cd author__affiliated_to__institute
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/author__affiliated_to__institute/edge_index.npy
cd ..

# paper__published__journal
mkdir paper__published__journal
cd paper__published__journal
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__published__journal/edge_index.npy
cd ..

# paper__topic__fos
mkdir paper__topic__fos
cd paper__topic__fos
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__topic__fos/edge_index.npy
cd ..

# paper__venue__conference
mkdir paper__venue__conference
cd paper__venue__conference
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__venue__conference/edge_index.npy
cd ..

# paper__written_by__author
mkdir paper__written_by__author
cd paper__written_by__author
wget https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper__written_by__author/edge_index.npy
cd ..

echo "IGBH-IGBH download complete"
