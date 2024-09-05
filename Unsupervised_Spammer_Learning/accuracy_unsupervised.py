import torch
import argparse
import numpy as np
# from kmeans_pytorch import kmeans
from numpy import *
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree
import torch_geometric.data as data
import pickle
import torch.nn.functional as F
import pandas as pd
from sklearn.cluster import KMeans

def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()
    # main parameters
    parser.add_argument('--dataset', type=str, default='spammer', help='dataset name')
    parser.add_argument('--EmbeddingPath', type=str, default='./pretrain_result/', help='embedding path')
    parser.add_argument('--seed', type=int, default=0, help='the seed of model.')
    parser.add_argument('--num_cluster', type=int, default=2, help='the number of communities.')
    parser.add_argument('--max_hop', type=int, default=10, help='the number of communities.')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ===== attribute-based clustering =====
    embedding_tensor = torch.from_numpy(np.load(args.EmbeddingPath + args.dataset + '.npy'))

    num_cluster=args.num_cluster

    print(embedding_tensor, embedding_tensor.shape)

    clusters = [[] for i in range(num_cluster)]
    if num_cluster > 1:

        embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)
        # cluster_ids_x, cluster_centers = kmeans(X=embedding_tensor, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=1e-4, max_iter=100)

        kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=10)
        cluster_ids_x = kmeans.fit_predict(embedding_tensor.cpu().numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to('cuda:0')

        for i in range(cluster_ids_x.shape[0]):
            clusters[cluster_ids_x[i]].append(i)
    
    
    # ===== evaluate the quality =====

    # labels_data = np.loadtxt("./data_graph/spammer_label.txt", delimiter=' ', dtype=int)
    # labels = torch.from_numpy(labels_data[:, 2])

    labels_data = pd.read_csv("./data_graph/spammer_label.txt", sep=' ', usecols=[1, 2], header=None)
    labels_data = labels_data.to_numpy()
    labels = torch.from_numpy(labels_data[:, 1])

    # print(clusters, labels)

    for cluster_id in range(num_cluster):
        tp, fn, fp, tn = 0, 0, 0, 0
        pred = clusters[cluster_id]
        for i in range(labels.shape[0]):
            if labels[i] == -10:
                continue
            if i in pred and labels[i]==1:
                tp += 1
            elif i not in pred and labels[i]==1:
                fn += 1
            elif i in pred and labels[i]==0:
                fp += 1
            elif i not in pred and labels[i]==0:
                tn += 1
            else:
                raise ValueError("the category number is incorrect")
        
        print(tp, ": Spammer to Spammer")
        print(fn, ": Spammer to Normal")
        print(fp, ": Normal to Spammer")
        print(tn, ": Normal to Normal")

        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f = 2*recall*precision /(recall + precision)

        print("RECALL = ", recall)
        print("PRECISION = ", precision)
        print("F-MEASURE = ", f)