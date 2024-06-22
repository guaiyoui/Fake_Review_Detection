Online Review Spammers Detection: An Implementation with Amazon Dataset

Code and data for paper under review at Information Systems Research (ISR-2022-047): "Are Neighbors Alike? A Semi-supervised Probabilistic Ensemble for Online Review Spammers Detection.” 

Repo Structure
1. Data/ contains the Amazon data set used in the paper:
   1) Amazon_metadata_original.txt is the original online review data file;
   2) UserLabel.txt indicates the mapping relation between UID in Meta-data and numerical UID, associated binary labels where 1 indicates spammer , 0 indicates legitimate and -10 indicates unused users;
   3) ProductMap.txt indicates the mapping relation between ASIN in Meta-data and numerical PID;
   4) UserFeature.txt stores ten individual behavioral features of each user, as illustrated in Table 1 of our paper;
   5) J01Network.txt stores the reviewer network constructed by default settings, where the network is stored by the edge list format;
   6) Training_Testing/ contains generated users for training and testing with 5%, 10%, 30% and 50% labeled data;
2. Code/ contains the implementations in C++ with Python for our model:
   1) Main.cpp is the C++ source file of our implementation, and it can be complied by:
      gcc -I/miniforge3/include -lpython3.9 -L/miniforge3/lib -lstdc++ main.cpp
      g++ -I/miniforge3/include -lpython3.9 -L/miniforge3/lib -lstdc++ main.cpp
      g++ -I/data1/jianweiw/env_forge/include -lpython3.10 -L/data1/jianweiw/env_forge/lib -lstdc++ main.cpp
      g++ -I/data1/jianweiw/env_forge/include main.cpp -L/data1/jianweiw/env_forge/lib -lpython3.10 -lstdc++
   2)python2c.py is the Python implementation of logistic regression and it is invoked by C++ code, in order to accelerate computations on large-scale data. 
3. Rank-Metrics/ contains evaluation modules including AP, AUC and Fmeasure metrics. Note that the Matlab codes are revised based on SpEAGLE implementations downloaded from http://shebuti.com/collective-opinion-spam-detection/