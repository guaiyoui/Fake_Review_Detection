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
    
    # load the pre-trained language model to get the review embedding
    model_name = "bert-base-chinese"  # 可以替换为其他预训练模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    new_list = []
    for sentence in comment_data:
        # 去除换行符、缩进
        sentence = str(sentence[0])
        sentence = sentence.replace('\n', '')
        sentence = sentence.replace('\t', '')
        sentence = sentence.replace(' ', '')
        new_list.append(sentence)

    results = pd.DataFrame({'Review': new_list[0:10]})
    
    results.to_csv("comment_query.csv", index=False)

    # print(new_list[0:10])
    # print(len(new_list))
    # list_sen_embedding = []
    # i=0
    # for one in new_list:
    #     # 使用tokenizer对句子进行编码
    #     inputs = tokenizer(one, return_tensors="pt", padding=True, truncation=True)
    
    #     # 获取句子向量
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    
    #     # 获取句子向量（CLS token对应的向量）
    #     sentence_vector = outputs.last_hidden_state[:, 0, :]
    #     list_sen_embedding.append(sentence_vector)
    
    # # 转换为numpy array
    # sentence_vectors = np.concatenate(list_sen_embedding, axis=0)
    
    # print(sentence_vectors.shape)
    
    # # 使用K均值聚类进行聚类
    # kmeans = KMeans(n_clusters=100)
    # kmeans.fit(sentence_vectors)
    # # 获取聚类结果
    # cluster_labels = kmeans.labels_
    # # 打印聚类结果
    # print(kmeans.labels_)

    # results = pd.DataFrame({'Review': new_list, 'Clustering_id': cluster_labels})
    
    # results.to_csv("./results/clustering_results.csv", index=False)
    
    
     
