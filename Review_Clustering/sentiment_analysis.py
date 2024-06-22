import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from sklearn.cluster import KMeans
import numpy as np
import argparse

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--cluster_num', type=int, default=100)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    # Read Data
    comment_data = pd.read_csv('comment_data.csv', encoding='gbk', sep='\t', header=None, index_col=None)
    
    # 消除缺失数据 NaN为缺失数据
    comment_data = comment_data.dropna().values.tolist() 
    print(comment_data[0:10])