echo "\n####### running time is $(date) #######\n" >> ./logs/Active_Spammer.txt

# Define test percentages
test_percents=("50percent" "30percent" "10percent" "5percent")
# test_percents=("50percent" "10percent" "5percent")

# Loop through each test percentage
for test_percent in "${test_percents[@]}"
do
    # Run the baseline script with specified parameters
    echo "Running baseline for $test_percent, no global sampling"
    CUDA_VISIBLE_DEVICES=5 nohup python run_baselines.py --dataset amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --test_percents $test_percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ >> ./logs/Active_Spammer_update.txt 2>&1 &&
    
    echo "Running baseline for $test_percent, with global sampling"
    CUDA_VISIBLE_DEVICES=5 nohup python run_baselines.py --dataset amazon --model GCN_update --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --test_percents $test_percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global >> ./logs/Active_Spammer_update.txt 2>&1
done &

# Optional: Use nohup to run the script in the background and log output