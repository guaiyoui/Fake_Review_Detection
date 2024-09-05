import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation_multi_steps, early_stopping, remove_nodes_from_walks, sgc_precompute, \
    get_classes_statistic, load_reddit_data, load_Amazon, load_coauthor
from models import get_model
from metrics import accuracy, f1, f1_my
import pickle as pkl
from args import get_citation_args
from time import perf_counter
from sampling_methods import *
import os
import datetime
import json
import pandas as pd
from community import community_louvain
from torch_geometric.utils import get_laplacian
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
# from torch_geometric.transforms import RandomNodeSampling

# Arguments
args = get_citation_args()



def loss_function_laplacian_regularization(output, train_labels, edge_index):
    loss_cls = F.cross_entropy(output, train_labels)
    
    # 计算稀疏格式的图拉普拉斯矩阵
    lap_sp = get_laplacian(edge_index, normalization='sym')[0]
    lap_sp = sp.FloatTensor(lap_sp)
    
    # 使用稀疏矩阵乘法计算损失
    loss_lap = torch.sum((lap_sp @ output.T) ** 2)
    
    return loss_cls + 0.1 * loss_lap

def loss_function_consistency_regularization(model, x, edge_index, train_labels, selected_nodes):
    # 计算原始输入的预测结果
    output = model(x, edge_index)
    output_select = output[selected_nodes, :]
    loss_cls = F.cross_entropy(output_select, train_labels)
    
    # Generate adversarial samples by adding random noise
    adv_x = x + 0.1 * torch.randn_like(x)

    # Compute the adversarial output
    adv_output = model(adv_x, edge_index)

    # Compute the consistency loss
    loss_cons = F.kl_div(adv_output, output.detach(), reduction='batchmean')

    return loss_cls + 0.1 * loss_cons

def loss_function_subgraph_regularization(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()
    # node_membership = node_membership.T
    
    # 计算同一子图内节点预测差异的平方和
    loss_subgraph = torch.sum(torch.matmul(node_membership.T, torch.matmul(adj, node_membership)))
    
    return loss_cls + 0.1 * loss_subgraph

def loss_function_subgraph_regularization_v1(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()

    # 计算邻域内节点预测相似性
    # 计算邻域内节点预测的相似度矩阵
    sim_matrix = torch.matmul(output, output.T)  # 计算预测值的相似度矩阵
    neighborhood_sim = torch.sum(sim_matrix * adj)  # 在邻接矩阵中加权相似度矩阵
    
    # 最小化相似性
    loss_similarity = torch.mean(neighborhood_sim)
    
    return loss_cls + 0.1 * loss_similarity

def loss_function_local_consistency(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    # 分类损失
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    # 计算邻接矩阵
    adj = edge_index.coalesce()
    adj = adj.to_dense()
    
    # 获取节点嵌入
    node_membership = model.get_node_embedding()
    
    # 计算每个节点及其邻域的预测差异
    loss_local_consistency = 0
    for node in selected_nodes:
        neighbors = torch.nonzero(adj[node, :]).squeeze()  # 获取邻域节点
        if len(neighbors) > 0:
            node_output = output[node, :]
            neighbors_output = output[neighbors, :]
            # 计算预测差异
            loss_local_consistency += torch.sum((node_output - neighbors_output) ** 2)
    
    return loss_cls + 0.1 * loss_local_consistency


def train_GCN(model, adj, selected_nodes, val_nodes,
             features, train_labels, val_labels,
             epochs=args.epochs, weight_decay=args.weight_decay,
             lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    best_acc_val = 0
    should_stop = False
    stopping_step = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        output = output[selected_nodes, :]
        # print(f'output.size(): {output.size()}')

        loss_train = F.cross_entropy(output, train_labels)
        # loss_train = loss_function_laplacian_regularization(output, train_labels, adj)
        # loss_train = loss_function_consistency_regularization(model, features, adj, train_labels, selected_nodes)
        # loss_train = loss_function_subgraph_regularization(model, features, adj, train_labels, selected_nodes)
        # loss_train = loss_function_local_consistency(model, features, adj, train_labels, selected_nodes)

        loss_train.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output = model(features, adj)
                output = output[val_nodes, :]
                acc_val = accuracy(output, val_labels)
                best_acc_val, stopping_step, should_stop = early_stopping(acc_val, best_acc_val, stopping_step,
                                                                          flag_step=10)
                if acc_val == best_acc_val:
                    # save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, f'checkpoint_{args.strategy}_{args.dataset}.pt')
                if should_stop:
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, acc_val, best_acc_val))
                    # load best model
                    checkpoint = torch.load(f'checkpoint_{args.strategy}_{args.dataset}.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        output = output[val_nodes, :]
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        print('acc_val: {}'.format(acc_val))
        f1_val, recall_val, precision_val = f1_my(output, val_labels)
        print('f1_val_my: {}'.format(f1_val))
    return model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val

def test_GCN(model, adj, test_mask, features, test_labels):
    model.eval()
    output = model(features, adj)
    output = output[test_mask, :]
    acc_test = accuracy(output, test_labels)
    micro_test, macro_test = f1(output, test_labels)

    f1_test, recall_test, precision_test = f1_my(output, test_labels)

    return acc_test, micro_test, macro_test, f1_test, recall_test, precision_test


def print_time_ratio(name, time1, train_time):
    print("{}: {:.4f}s, ratio: {}".format(name, time1, time1 / train_time))


def save_time_result(file_name, *args):
    # args is the names of the time
    save_dict = {}
    save_list = []
    for arg in args:
        save_list.append(arg)

    for x in save_list:
        save_dict[x] = eval(x)
    # print(save_dict)
    import pickle
    with open(file_name, 'wb') as f:
        pickle.dump(save_dict, f)


def ensure_nonrepeat(idx_train, selected_nodes):
    for node in idx_train:
        if node in selected_nodes:
            raise Exception(
                'In this iteration, the node {} need to be labelled is already in selected_nodes'.format(node))
    return

def augment_feature(feature, nx_G):
    print("===== 1. The modularity-based feature augmentation. =====")
    partition = community_louvain.best_partition(nx_G)
    modularity = community_louvain.modularity(partition, nx_G)
    print(f"the modularity of community is {modularity}")
    # 创建一个字典存储每个社区的modularity值
    node_modularity = {}
    for community in set(partition.values()):
            # 取出该社区的节点
        nodes_in_community = [node for node, comm in partition.items() if comm == community]
        # 计算该社区在整体中的modularity贡献
        subgraph = nx_G.subgraph(nodes_in_community)
        # print(subgraph)
        community_partition = {node: community for node in nodes_in_community}
        community_modularity = community_louvain.modularity({**partition, **community_partition}, nx_G)
        # 分配给该社区中的每个节点
        for node in nodes_in_community:
            node_modularity[node] = community_modularity
    
    augmented_mod_feat = []
    for i in range(feature.shape[0]):
        if i in node_modularity:
            augmented_mod_feat.append(node_modularity[i])
        else:
            augmented_mod_feat.append(0)
    # kcore based 

    augmented_core_feat = []
    print("===== 2. The k-core-based feature augmentation. =====")
    # Calculate k-core values for each node
    core_numbers = nx.core_number(nx_G)
    for i in range(feature.shape[0]):
        if i in core_numbers:
            augmented_core_feat.append(core_numbers[i])
        else:
            augmented_core_feat.append(0)
    
    # print(augmented_core_feat)
    result = np.column_stack((feature, np.array(augmented_mod_feat), np.array(augmented_core_feat)))

    return result
    

class run_wrapper():
    def __init__(self, dataset, normalization, cuda):
        if dataset in ['spammer', 'amazon', 'yelp']:

            self.graph = None
            # graph_data = np.loadtxt("../Unsupervised_Spammer_Learning/data_graph/spammer_edge_index.txt", delimiter=' ', dtype=int)
            print("start loading J01Network")
            graph_data = np.loadtxt(args.data_path+"J01Network.txt", delimiter=' ', dtype=int)
            graph_data[:,0] = graph_data[:,0] - 1
            graph_data[:,1] = graph_data[:,1] - 1
            self.nx_G = nx.Graph()
            self.nx_G.add_edges_from(graph_data)
            
            print("start constructing adj")
            edge_tensor = torch.from_numpy(graph_data).long()
            indices = edge_tensor.t().contiguous()
            num_edges = edge_tensor.shape[0]
            values = torch.ones(num_edges)
            num_nodes = edge_tensor.max().item() + 1
            adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
            adj = adj.coalesce()
            adj = adj.to('cuda:0')
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            row_sum[row_sum == 0] = 1  # 避免除以零
            values_normalized = 1.0 / row_sum[adj.indices()[0]]
            adj_normalized = torch.sparse_coo_tensor(adj.indices(), values_normalized, adj.size())
            self.adj = adj_normalized

            print("start loading features")
            
            features = np.loadtxt(args.data_path+"UserFeature.txt", delimiter='\t')
            # features = augment_feature(features, self.nx_G)
            self.features = torch.from_numpy(features).float().cuda()

            print("start loading labels")
            labels_data = pd.read_csv(args.data_path+"UserLabel.txt", sep=' ', usecols=[1, 2])
            labels_data = labels_data.to_numpy()
            self.labels = torch.from_numpy(labels_data[:, 1]).cuda()
            

            training_data = np.loadtxt(args.data_path+"Training_Testing/5percent/train_4.csv", delimiter=' ', dtype=int)
            testing_data = np.loadtxt(args.data_path+"Training_Testing/5percent/test_4.csv", delimiter=' ', dtype=int)
    
            self.idx_test = torch.from_numpy(testing_data[:,0] - 1).cuda()

            self.idx_non_test = (training_data[:,0]-1).tolist() 

        self.dataset = dataset
        print(f'self.labels: {self.labels, self.labels.shape}')
        print(f'self.adj: {self.adj}')
        print(f'self.feature: {self.features, self.features.shape}')
        print(f'self.idx_test is {len(self.idx_test)}, self.idx_non_test is {len(self.idx_non_test)}')
        print('finished loading dataset')
        self.raw_features = self.features
        if args.model == "SGC":
            self.features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
            print("{:.4f}s".format(precompute_time))
            if args.strategy == 'featprop':
                self.dis_features = self.features
        else:
            if args.strategy == 'featprop':
                self.dis_features, precompute_time = sgc_precompute(self.features, self.adj, args.degree)
                # torch.save(self.dis_features.data, 'visualization/featprop_feat.pt')
                # input('wait')


    def run(self, strategy, num_labeled_list=[10, 15, 20, 25, 30, 35, 40, 50], max_budget=160, seed=1):
        set_seed(seed, args.cuda)
        max_budget = num_labeled_list[-1]
        if strategy in ['ppr', 'pagerank', 'pr_ppr', 'mixed', 'mixed_random', 'unified']:
            print('strategy is ppr or pagerank')
            # nx_G = nx.from_dict_of_lists(self.graph)
            nx_G = self.nx_G
            PR_scores = nx.pagerank(nx_G, alpha=0.85)
            # print('PR_scores: ', PR_scores)
            nx_nodes = nx.nodes(nx_G)
            original_weights = {}
            for node in nx_nodes:
                original_weights[node] = 0.

        idx_non_test = self.idx_non_test.copy()
        print('len(idx_non_test) is {}'.format(len(idx_non_test)))
        # Select validation nodes.
        # num_val = 500
        num_val = 10
        idx_val = np.random.choice(idx_non_test, num_val, replace=False)
        idx_non_test = list(set(idx_non_test) - set(idx_val))

        # initially select some nodes.
        L = 5
        selected_nodes = np.random.choice(idx_non_test, L, replace=False)
        idx_non_test = list(set(idx_non_test) - set(selected_nodes))

        model = get_model(args.model, self.features.size(1), 2, args.hidden, args.dropout,
                          args.cuda)

        budget = 20
        steps = 6
        pool = idx_non_test
        print('len(idx_non_test): {}'.format(len(idx_non_test)))
        np.random.seed() # cancel the fixed seed
        if args.model == 'GCN':
            args.lr = 0.01
            model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                                self.labels[selected_nodes],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)
        print('-------------initial results------------')
        print('micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
        # Active learning
        print('strategy: ', strategy)
        cur_num = 0
        val_results = {'acc': [], 'micro': [], 'macro': [], 'f1': [], "recall":[], "precision":[]}
        test_results = {'acc': [], 'micro': [], 'macro': [], 'f1': [], "recall":[], "precision":[]}

        uncertainty_results = {}
        if strategy == 'rw':
            self.walks = remove_nodes_from_walks(self.walks, selected_nodes)
        if strategy == 'unified':
            nodes = nx.nodes(nx_G)
            uncertainty_score = get_uncertainty_score(model, self.features, nodes)
            init_weights = {n: float(uncertainty_score[n]) for n in nodes}
            for node in selected_nodes:
                init_weights[node] = 0
            uncertainty_results[5] = {'selected_nodes': selected_nodes.tolist(), 'uncertainty_scores': init_weights}


        time_AL = 0
        for i in range(len(num_labeled_list)):
            if num_labeled_list[i] > max_budget:
                break
            budget = num_labeled_list[i] - cur_num
            cur_num = num_labeled_list[i]
            t1 = perf_counter()
            if strategy == 'random':
                idx_train = query_random(budget, pool)
            elif strategy == 'uncertainty':
                if args.model == 'GCN':
                    idx_train = query_uncertainty_GCN(model, self.adj, self.features, budget, pool)
                else:
                    idx_train = query_uncertainty(model, self.features, budget, pool)
            elif strategy == 'largest_degrees':
                if args.dataset not in ['cora', 'citeseer', 'pubmed']:
                    idx_train = query_largest_degree(self.graph, budget, pool)
                else:
                    idx_train = query_largest_degree(nx.from_dict_of_lists(self.graph), budget, pool)
            elif strategy == 'coreset_greedy':
                idx_train = qeury_coreset_greedy(self.features, list(selected_nodes), budget, pool)
            elif strategy == 'featprop':
                idx_train = query_featprop(self.dis_features, budget, pool)
            elif strategy == 'pagerank':
                idx_train = query_pr(PR_scores, budget, pool)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))

            time_AL += perf_counter() - t1
            assert len(idx_train) == budget
            ensure_nonrepeat(idx_train, selected_nodes)
            selected_nodes = np.append(selected_nodes, idx_train)
            pool = list(set(pool) - set(idx_train))
            if args.model == 'GCN':
                model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                             self.labels[selected_nodes],
                                                                             self.labels[idx_val],
                                                                             args.epochs, args.weight_decay, args.lr,
                                                                             args.dropout)
            print(f"the number of labels is {num_labeled_list[i]}")
            if args.model == 'GCN':
                acc_test, micro_test, macro_test, f1_test, recall_test, precision_test = test_GCN(model, self.adj, self.idx_test, self.features,
                                                                   self.labels[self.idx_test])
            

            print('f1_val_my: {}'.format(f1_val))
            print('f1_test_my: {}'.format(f1_test))

            acc_val = round(acc_val, 4)
            acc_test = round(acc_test, 4)
            micro_val = round(micro_val, 4)
            micro_test = round(micro_test, 4)
            macro_val = round(macro_val, 4)
            macro_test = round(macro_test, 4)
            f1_val = round(f1_val, 4)
            f1_test = round(f1_test, 4)
            recall_val = round(recall_val, 4)
            recall_test = round(recall_test, 4)
            precision_val = round(precision_val)
            precision_test = round(precision_test)

            val_results['acc'].append(acc_val)
            val_results['micro'].append(micro_val)
            val_results['macro'].append(macro_val)
            val_results['f1'].append(f1_val)
            val_results['recall'].append(recall_val)
            val_results['precision'].append(precision_val)
            test_results['acc'].append(acc_test)
            test_results['micro'].append(micro_test)
            test_results['macro'].append(macro_test)
            test_results['f1'].append(f1_test)
            test_results['recall'].append(recall_test)
            test_results['precision'].append(precision_test)

        print('AL Time: {}s'.format(time_AL))
        return val_results, test_results, get_classes_statistic(self.labels[selected_nodes].cpu().numpy()), time_AL



if __name__ == '__main__':

    if args.dataset == 'spammer':
        num_labeled_list = [i for i in range(10,151,10)]
    elif args.dataset == 'amazon':
        num_labeled_list = [i for i in range(10,201,10)]
    elif args.dataset == 'yelp':
        num_labeled_list = [10, 20, 30, 40] + [i for i in range(50,1001,50)]
    num_interval = len(num_labeled_list)

    val_results = {'micro': [[] for _ in range(num_interval)],
                   'macro': [[] for _ in range(num_interval)],
                   'acc': [[] for _ in range(num_interval)],
                   'f1': [[] for _ in range(num_interval)],
                   'recall': [[] for _ in range(num_interval)],
                   'precision': [[] for _ in range(num_interval)]}

    test_results = {'micro': [[] for _ in range(num_interval)],
                    'macro': [[] for _ in range(num_interval)],
                    'acc': [[] for _ in range(num_interval)],
                    'f1': [[] for _ in range(num_interval)],
                    'recall': [[] for _ in range(num_interval)],
                    'precision': [[] for _ in range(num_interval)]}
    if args.file_io:
        input_file = 'random_seed_10.txt'
        with open(input_file, 'r') as f:
            seeds = f.readline()
        seeds = list(map(int, seeds.split(' ')))
    else:
        seeds = [52, 574, 641, 934, 12]
        # seeds = [574]

    # seeds = seeds * 10 # 10 runs
    seeds = seeds * 1 # 2 runs
    seed_idx_map = {i: idx for idx, i in enumerate(seeds)}
    num_run = len(seeds)
    wrapper = run_wrapper(args.dataset, args.normalization, args.cuda)

    total_AL_time = 0
    for i in range(len(seeds)):
        print('current seed is {}'.format(seeds[i]))
        val_dict, test_dict, classes_dict, cur_AL_time = wrapper.run(args.strategy, num_labeled_list=num_labeled_list,
                                                                     seed=seeds[i])

        for metric in ['micro', 'macro', 'acc', 'f1', 'recall', 'precision']:
            for j in range(len(val_dict[metric])):
                val_results[metric][j].append(val_dict[metric][j])
                test_results[metric][j].append(test_dict[metric][j])

        total_AL_time += cur_AL_time

    val_avg_results = {'micro': [0. for _ in range(num_interval)],
                       'macro': [0. for _ in range(num_interval)],
                       'acc': [0. for _ in range(num_interval)],
                       'f1': [0. for _ in range(num_interval)],
                       'recall': [0. for _ in range(num_interval)],
                       'precision': [0. for _ in range(num_interval)]}
    test_avg_results = {'micro': [0. for _ in range(num_interval)],
                    'macro': [0. for _ in range(num_interval)],
                    'acc': [0. for _ in range(num_interval)],
                    'f1': [0. for _ in range(num_interval)],
                    'recall': [0. for _ in range(num_interval)],
                    'precision': [0. for _ in range(num_interval)]}
    val_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)],
                        'acc': [0. for _ in range(num_interval)],
                        'f1': [0. for _ in range(num_interval)],
                        'recall': [0. for _ in range(num_interval)],
                        'precision': [0. for _ in range(num_interval)]}
    test_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)],
                        'acc': [0. for _ in range(num_interval)],
                        'f1': [0. for _ in range(num_interval)],
                        'recall': [0. for _ in range(num_interval)],
                        'precision': [0. for _ in range(num_interval)]}
    for metric in ['micro', 'macro', 'acc', 'f1', 'recall', 'precision']:
        for j in range(len(val_results[metric])):
            val_avg_results[metric][j] = np.mean(val_results[metric][j])
            test_avg_results[metric][j] = np.mean(test_results[metric][j])
            val_std_results[metric][j] = np.std(val_results[metric][j])
            test_std_results[metric][j] = np.std(test_results[metric][j])


    if args.model == 'GCN':
        dir_path = os.path.join('./10splits_10runs_results', args.dataset)
    else:
        dir_path = os.path.join('./results', args.dataset)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, '{}.txt'.format(args.strategy))
    with open(file_path, 'a') as f:
        f.write('---------datetime: %s-----------\n' % datetime.datetime.now())
        f.write(f'Budget list: {num_labeled_list}\n')
        f.write(f'learning rate: {args.lr}, epoch: {args.epochs}, weight decay: {args.weight_decay}, hidden: {args.hidden}\n')
        f.write(f'50runs using seed.txt\n')
        for metric in ['micro', 'macro', 'acc', 'f1', 'recall', 'precision']:
            f.write("Test_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_avg_results[metric])))
            f.write("Test_{}_std {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_std_results[metric])))

        f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
    
