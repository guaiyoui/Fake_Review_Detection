



CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --test_percents 30percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy largest_degrees --file_io 1 --lr 0.01 --hidden 16 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy coreset_greedy --file_io 1 --lr 0.01 --hidden 16 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ --sample_global

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

CUDA_VISIBLE_DEVICES=4 python run_ComGA.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.001 --hidden 64 --test_percents 5percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/



CUDA_VISIBLE_DEVICES=4 python clustering_al.py --dataset amazon --epoch 300 --strategy LSCALE --file_io 1 --reweight 0 --lr 0.001 --hidden 64  --feature cat --adaptive 1 --test_percents 10percent --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/

# CUDA_VISIBLE_DEVICES=0 python LSCALE.py --dataset $i --epoch 300 --strategy LSCALE --file_io 1 --reweight 0 --hidden 100 --feature cat --adaptive 1 --weight_decay 0.000005

