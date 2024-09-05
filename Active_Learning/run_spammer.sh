nohup python run_baselines.py --dataset spammer --model GCN --epoch 300 --strategy random --file_io 1 --lr 0.01 --hidden 16 >> ./logs/spammer_v1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/AmazonData/ >> ./logs/amazon_v1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python run_baselines.py --dataset yelp --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/ >> ./logs/yelp_v1.txt 2>&1 &

python LSCALE.py --dataset amazon --model GCN --epoch 300 --strategy LSCALE --file_io 1 --lr 0.01 --reweight 0 --hidden 100 --feature cat --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/AmazonData/

python LSCALE.py --dataset amazon --model GCN --epoch 300 --strategy LSCALE --file_io 1 --lr 0.01 --reweight 0 --hidden 100 --feature cat --data_path ../Yelp_Kaggle/Data/


CUDA_VISIBLE_DEVICES=4 nohup python run_baselines.py --dataset yelp --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Yelp_Kaggle/Data/ >> ./logs/yelp_v2.txt 2>&1 &

scp -o 'ProxyJump z5380302@cse.unsw.edu.au' -r jianweiw@icml1.cse.unsw.edu.au:/data1/jianweiw/LLM/Fake_review_detection/Fake_Review_Detection/Active_Learning/ ./

scp -o 'ProxyJump z5380302@cse.unsw.edu.au' -r jianweiw@icml1.cse.unsw.edu.au:/data1/jianweiw/LLM/Fake_review_detection/Fake_Review_Detection/Active_Learning/ ./code

scp -o 'ProxyJump z5380302@cse.unsw.edu.au' -r jianweiw@icml1.cse.unsw.edu.au:/data1/jianweiw/LLM/Fake_review_detection/Fake_Review_Detection/Yelp_Kaggle/YelpData/ ./code

scp -r ./code/ jianweiw@202.120.5.12:/data/jianweiw/review_utd/ 

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --dataset yelp --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Yelp_Kaggle/Data/


CUDA_VISIBLE_DEVICES=0 nohup python run_baselines.py --dataset yelp --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../YelpData/ >> ./logs/yelp_v3.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python run_baselines.py --dataset yelp --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../YelpData/ >> ./logs/yelp_v4.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python run_baselines.py --dataset amazon --model GCN --epoch 300 --strategy uncertainty --file_io 1 --lr 0.01 --hidden 16 --data_path ../Spammer-ISR-Initial-Exp/ISR-spammer-detection/AmazonData/ >> ./logs/amazon_v1.txt 2>&1 &