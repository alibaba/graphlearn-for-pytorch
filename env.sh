#!/bin/bash

set -eo pipefail

script_dir=$(dirname "$(realpath "$0")")
SYS_LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64
export LD_LIBRARY_PATH=$SYS_LD_LIBRARY_PATH:${script_dir}/built/lib:$LD_LIBRARY_PATH