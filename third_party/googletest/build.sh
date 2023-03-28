#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
code_src=${script_dir}/googletest
install_prefix=${script_dir}/build

export CXXFLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0"

cd "${code_src}" && mkdir -p tmp                                 && \
cd tmp && cmake -DCMAKE_INSTALL_PREFIX="${install_prefix}" ..    && \
make -j && mkdir -p "${install_prefix}" && make install && cd .. && \
rm -rf tmp

unset CXXFLAGS