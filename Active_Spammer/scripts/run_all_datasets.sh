#!/bin/bash

# Define weight loss values
weight_losses=("5percent" "10percent" "30percent" "50percent")

# Loop through each weight loss value
for weight_loss in "${weight_losses[@]}"
do
    # Run the baseline script with specified parameters
    echo "Running baseline with dataset $weight_loss, no global sampling"
    CUDA_VISIBLE_DEVICES=3 nohup python run_ComGA.py \
        --dataset amazon \
        --model GCN \
        --epoch 300 \
        --strategy uncertainty \
        --file_io 1 \
        --lr 0.001 \
        --hidden 64 \
        --test_percents "$weight_loss" \
        --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ \
        >> ./logs/loss_${weight_loss}.txt 2>&1 &
done
