#!/bin/bash

# Define weight loss values
weight_losses=(0.01 0.1 0.5 1.0 4.0)

# Loop through each weight loss value
for weight_loss in "${weight_losses[@]}"
do
    # Run the baseline script with specified parameters
    echo "Running baseline with weight loss $weight_loss, no global sampling"
    CUDA_VISIBLE_DEVICES=3 python run_ComGA.py \
        --dataset amazon \
        --model GCN \
        --epoch 300 \
        --strategy uncertainty \
        --file_io 1 \
        --lr 0.001 \
        --hidden 64 \
        --weight_t_loss "$weight_loss" \
        --test_percents 10percent \
        --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ \
        >> ./logs/loss_tloss_${weight_loss}.txt 2>&1
done &