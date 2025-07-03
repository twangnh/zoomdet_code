#!/usr/bin/env bash

for config_file in ./configs/faster_rcnn/Visdrone_grid_search/*; do
  # 提取文件名
    file_name=$(basename $config_file)

    # 检查文件名是否以 "faster-rcnn_r101" 开头
    if [[ $file_name == faster-rcnn_r101* ]]; then
        CONFIG=$config_file
        GPUS=4
        NNODES=${NNODES:-1}
        NODE_RANK=${NODE_RANK:-0}
        PORT=${PORT:-29500}
        MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            $(dirname "$0")/train.py \
            $CONFIG \
            --cfg-options randomness.seed=2024 \
              randomness.diff_rank_seed=True \
            --launcher pytorch ${@:3} \
            --work-dir /root/autodl-tmp/0410_diff_hingemag_$file_name
    else
        echo "Skipping config file: $config_file"
    fi
done
