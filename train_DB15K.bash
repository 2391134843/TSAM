#!/bin/bash

# Specify the GPU device to use
export CUDA_VISIBLE_DEVICES=0

# Run the Python script and pass in the corresponding parameters
python3 Train.py \
    --data DB15K \
    --num_epoch 5000 \
    --hidden_dim 1024 \
    --lr 4e-4 \
    --dim 256 \
    --max_vis_token 16 \
    --max_txt_token 24 \
    --num_head 4 \
    --emb_dropout 0.9 \
    --vis_dropout 0.4 \
    --txt_dropout 0.1 \
    --num_layer_dec 2 
    