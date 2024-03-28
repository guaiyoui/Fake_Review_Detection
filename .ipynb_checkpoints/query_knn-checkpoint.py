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
    parser.add_argument('--top_k', type=int, default=20)

    return parser.parse_args()

def top_similar_vectors(vector1, vectors2, top_k=10):
    similarities = np.dot(vectors2, vector1) / (np.linalg.norm(vectors2, axis=1) * np.linalg.norm(vector1))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices

if __name__ == "__main__":

    args = parse_args()

    # load the pre-trained language model to get the review embedding
    model_name = "bert-base-chinese"  # 可以替换为其他预训练模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Read Data
    comment_data = pd.read_csv('comment_data.csv', encoding='gbk', sep='\t', header=None, index_col=None)
    
    # 消除缺失数据 NaN为缺失数据
    comment_data = comment_data.dropna().values.tolist() 

    new_list = []
    for sentence in comment_data:
        # 去除换行符、缩进
        sentence = str(sentence[0])
        sentence = sentence.replace('\n', '')
        sentence = sentence.replace('\t', '')
        sentence = sentence.replace(' ', '')
        new_list.append(sentence)
    
    print(new_list[0:10])
    print(len(new_list))
    list_sen_embedding = []
    i=0
    for one in new_list:
        # 使用tokenizer对句子进行编码
        inputs = tokenizer(one, return_tensors="pt", padding=True, truncation=True)
    
        # 获取句子向量
        with torch.no_grad():
            outputs = model(**inputs)
    
        # 获取句子向量（CLS token对应的向量）
        sentence_vector = outputs.last_hidden_state[:, 0, :]
        list_sen_embedding.append(sentence_vector)
    
    # 转换为numpy array
    sentence_vectors = np.concatenate(list_sen_embedding, axis=0)
    print(sentence_vectors.shape)


    # Read Query Data
    comment_query = pd.read_csv('comment_query.csv', sep='\t', index_col=None)
    comment_query = comment_query.dropna().values.tolist() 

    new_list_query = []
    for sentence in comment_query:
        sentence = str(sentence[0])
        sentence = sentence.replace('\n', '')
        sentence = sentence.replace('\t', '')
        sentence = sentence.replace(' ', '')
        new_list_query.append(sentence)
    
    print(new_list_query[0:10])
    print(len(new_list_query))
    query_sen_embedding = []
    i=0
    for one in new_list_query:
        inputs = tokenizer(one, return_tensors="pt", padding=True, truncation=True)
    
        with torch.no_grad():
            outputs = model(**inputs)
    
        sentence_vector = outputs.last_hidden_state[:, 0, :]
        query_sen_embedding.append(sentence_vector)
    
    query_vectors = np.concatenate(query_sen_embedding, axis=0)
    print(query_vectors.shape)

    top_indices_list = []
    for query in query_vectors:
        top_indices = top_similar_vectors(query, sentence_vectors, top_k=args.top_k)
        top_indices_list.append(top_indices)
    
    # print(top_indices_list)

    query_review_result, knn_result = [], []
    for i in range(query_vectors.shape[0]):
        for j in range(args.top_k):
            query_review_result.append(comment_query[i])
            knn_result.append(comment_data[top_indices_list[i][j]])

    results = pd.DataFrame({'Query': query_review_result, 'KNN Review': knn_result})
    
    results.to_csv("./results/query_knn_results.csv", index=False)

    
    
     
