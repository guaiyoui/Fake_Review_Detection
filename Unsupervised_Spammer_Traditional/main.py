import torch
import numpy as np
import argparse
import networkx as nx
from community import community_louvain
import random
from sklearn.cluster import SpectralClustering
from networkx.algorithms.community import modularity_max
import matplotlib.pyplot as plt
import pandas as pd

# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cora',
                        help='Choose from {pubmed}')
    parser.add_argument('--device', type=int, default=1, 
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--readout', type=str, default="mean")
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='the value the balance the loss.')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--group_epoch_gap', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates',  type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=50, 
                        help='Patience for early stopping')
    
    # model saving
    parser.add_argument('--save_path', type=str, default='./model/',
                        help='The path for the model to save')
    parser.add_argument('--model_name', type=str, default='cora',
                        help='The name for the model to save')
    parser.add_argument('--embedding_path', type=str, default='./pretrain_result/',
                        help='The path for the embedding to save')

    return parser.parse_args()

def compute_metric(pred, labels):
    tp, fn, fp, tn = 0, 0, 0, 0
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
        
    # print(tp, ": Spammer to Spammer")
    # print(fn, ": Spammer to Normal")
    # print(fp, ": Normal to Spammer")
    # print(tn, ": Normal to Normal")

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    f = 2*recall*precision /(recall + precision)

    # print("RECALL = ", recall)
    # print("PRECISION = ", precision)
    # print("F-MEASURE = ", f)
    return f, recall, precision


def plot(x_data, y_data, figure_name):
    # 1: x_data 和 y_data 是长度相等的 list
    # 2: figure_name 是 figure 要存的相对位置
    # 3: 需要标出最小和最后的值

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data)

    # 找出最小值和最后的值
    min_value = min(y_data)
    max_value = max(y_data)
    min_index = y_data.index(min_value)
    max_index = y_data.index(max_value)

    # 标注最小值（左上角）
    plt.annotate(f'Min: {min_value}', 
                 xy=(x_data[min_index], min_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='blue', shrink=0.05))

    # 标注最后的值（右上角）
    plt.annotate(f'Max: {max_value}', 
                 xy=(x_data[max_index], max_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction',
                 ha='right',
                 arrowprops=dict(facecolor='red', shrink=0.05))
    
    # 设置标题和轴标签
    plt.title(figure_name)
    plt.xlabel('Kcore Value')
    plt.ylabel('F1-scores')

    # 保存图表
    plt.savefig("./figures/"+figure_name+".png")
    plt.close()


if __name__ == "__main__":
    
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    graph_data = np.loadtxt("../Unsupervised_Spammer_Learning/data_graph/spammer_edge_index.txt", delimiter=' ', dtype=int)
    features = np.loadtxt("../Unsupervised_Spammer_Learning/data_graph/spammer_feature.txt", delimiter='\t')
    # labels_data = np.loadtxt("../Unsupervised_Spammer_Learning/data_graph/spammer_label.txt", delimiter=' ', dtype=int)
    # labels = torch.from_numpy(labels_data[:, 2])
    labels_data = pd.read_csv("../Unsupervised_Spammer_Learning/data_graph/spammer_label.txt", sep=' ', usecols=[1, 2], header=None)
    labels_data = labels_data.to_numpy()
    labels = torch.from_numpy(labels_data[:, 1])

    # 调整节点索引
    graph_data[:,0] = graph_data[:,0] - 1
    graph_data[:,1] = graph_data[:,1] - 1

    # 创建NetworkX图
    G = nx.Graph()
    G.add_edges_from(graph_data)

    # 1. Modularity-based community detection
    partition = community_louvain.best_partition(G)
    modularity = community_louvain.modularity(partition, G)

    print("Modularity-based communities:")
    print(f"Number of communities: {len(set(partition.values()))}")
    print(f"Modularity: {modularity}")

    print("===== 1. The modularity-based community detection. =====")
    # Modularity-based community detection
    communities = modularity_max.greedy_modularity_communities(G)
    # Take the largest two communities
    top_communities = sorted(communities, key=len, reverse=True)[:2]
    f1, recall, precision = compute_metric(top_communities[0], labels)
    print("RECALL = ", recall)
    print("PRECISION = ", precision)
    print("F-MEASURE = ", f1)

    f1, recall, precision = compute_metric(top_communities[1], labels)
    print("RECALL = ", recall)
    print("PRECISION = ", precision)
    print("F-MEASURE = ", f1)

    # 2. K-core-based community detection
    k_core = nx.k_core(G)
    k_core_communities = list(nx.connected_components(k_core))

    print("===== 2. The k-core-based community detection. =====")
    # Calculate k-core values for each node
    core_numbers = nx.core_number(G)

    communities_aff = {}
    # Print k-core values
    print("K-core values for each node:")
    for node, kcore in core_numbers.items():
        # print(f"Node {node}: k-core value {kcore}")
        if kcore not in communities_aff.keys():
            communities_aff[kcore] = []
        communities_aff[kcore].append(node)
    
    k_value = []
    f1_score = []
    recall_score = []
    precision_score = []
    for k in range(1, max(communities_aff.keys())+1):
        print(f"the k value is {k}")
        detected_community = []
        for i in range(k, max(communities_aff.keys())+1):
            if i in communities_aff.keys():
                detected_community.extend(communities_aff[i])
        f1, recall, precision = compute_metric(detected_community, labels)
        k_value.append(k)
        f1_score.append(f1)
        recall_score.append(recall)
        precision_score.append(precision)
    plot(k_value, f1_score, "kcore_method")
    plot(k_value, recall_score, "recall_score_method")
    plot(k_value, precision_score, "precision_score_method")

    # 3. 指定数量的社区检测（基于谱聚类）
    # 指定想要的社区数量
    n_communities = 2  # 你可以根据需要修改这个数字

    nodes = list(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    sc = SpectralClustering(n_clusters=n_communities, affinity='precomputed', n_init=100, assign_labels='discretize')
    partition = sc.fit_predict(adj_matrix)

    print("===== 3. Spectral clustering communities. =====")
    print(f"Number of communities: {n_communities}")

    # 计算modularity
    def calculate_modularity(G, partition):
        m = G.number_of_edges()
        degrees = dict(G.degree())
        Q = 0
        for community in set(partition.values()):
            community_nodes = [node for node in G.nodes() if partition[node] == community]
            community_subgraph = G.subgraph(community_nodes)
            L_c = community_subgraph.number_of_edges()
            D_c = sum(degrees[node] for node in community_nodes)
            Q += (L_c / m) - (D_c / (2 * m)) ** 2
        return Q

    # 创建节点到社区的映射
    partition_dict = {node: community for node, community in zip(nodes, partition)}

    modularity = calculate_modularity(G, partition_dict)
    print(f"Modularity: {modularity}")
    for community in range(2):
        community_nodes = [node for node in G.nodes() if partition_dict[node] == community]
        f1, recall, precision = compute_metric(community_nodes, labels)
        print("RECALL = ", recall)
        print("PRECISION = ", precision)
        print("F-MEASURE = ", f1)
    
    
