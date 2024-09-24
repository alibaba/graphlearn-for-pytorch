ARG GLT_TORCH_VERSION=2.3.0
ARG CUDA_WHEEL_VERSION=121

FROM pytorch/manylinux-cuda$CUDA_WHEEL_VERSION

ARG GLT_TORCH_VERSION CUDA_WHEEL_VERSION
ENV ABIS="cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311"

RUN set -ex; \
    for abi in $ABIS; do \
        PYBIN=/opt/python/${abi}/bin; \
        ${PYBIN}/pip install ninja scipy; \
        ${PYBIN}/pip install torch==${GLT_TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_WHEEL_VERSION}; \
        ${PYBIN}/pip install torch_geometric; \
        ${PYBIN}/pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
            -f https://data.pyg.org/whl/torch-${GLT_TORCH_VERSION}+cu${CUDA_WHEEL_VERSION}.html; \
        ${PYBIN}/pip install auditwheel; \
    done