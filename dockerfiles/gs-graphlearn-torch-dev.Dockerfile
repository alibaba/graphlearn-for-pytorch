ARG REGISTRY=registry.cn-hongkong.aliyuncs.com
ARG GLT_TORCH_VERSION=2.3.0
ARG GS_VERSION=latest

FROM $REGISTRY/graphscope/graphscope-dev:$GS_VERSION

ARG GLT_TORCH_VERSION

RUN pip install torch==${GLT_TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
    && pip install torch_geometric \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${GLT_TORCH_VERSION}+cpu.html
    
