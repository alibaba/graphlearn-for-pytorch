ARG GLT_CUDA_VERSION=12.1
ARG GLT_TORCH_VERSION=2.3.0

FROM nvidia/cuda:${GLT_CUDA_VERSION}.0-devel-ubuntu22.04

RUN apt update && apt install python3-pip vim git cmake -y
RUN ln -s python3 /usr/bin/python

ARG GLT_TORCH_VERSION GLT_CUDA_VERSION

RUN CUDA_WHEEL_VERSION=$(echo "$GLT_CUDA_VERSION" | sed 's/\.//g') \
    && pip install torch==${GLT_TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_WHEEL_VERSION} \
    && pip install torch_geometric \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${GLT_TORCH_VERSION}+cu${CUDA_WHEEL_VERSION}.html
    
