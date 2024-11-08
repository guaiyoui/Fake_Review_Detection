import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import early_stopping, remove_nodes_from_walks, sgc_precompute, \
    get_classes_statistic
from models import get_model, DGI
from metrics import accuracy, f1, f1_isr
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
import matplotlib.pyplot as plt
import psgd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Arguments
args = get_citation_args()
def plot(index, data, figure_name):

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(index, data)

    # 找出最小值、最大值和最后的值
    min_value = min(data)
    max_value = max(data)
    last_value = data[-1]
    min_index = data.index(min_value)
    max_index = data.index(max_value)
    last_index = len(data) - 1

    # 标注最小值（左上角）
    plt.annotate(f'Min: {min_value}', 
                 xy=(min_index, min_value), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction')

    # 标注最大值（中间上方）
    plt.annotate(f'Max: {max_value}', 
                 xy=(max_index, max_value), 
                 xytext=(0.5, 0.95), 
                 textcoords='axes fraction',
                 ha='center')

    # 标注最后的值（右上角）
    plt.annotate(f'Last: {last_value}', 
                 xy=(last_index, last_value), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right')
    
    # 设置标题和轴标签
    plt.title(figure_name)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 保存图表
    plt.savefig("./figures/"+figure_name+".png")
    plt.close()

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


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
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
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                output = model(val_features)
                acc_val = accuracy(output, val_labels)
                best_acc_val, stopping_step, should_stop = early_stopping(acc_val, best_acc_val, stopping_step,
                                                                          flag_step=10)
                if acc_val == best_acc_val:
                    # save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, 'checkpoint_inc12.pt')
                if should_stop:
                    print('epoch: {}, acc_val: {}, best_acc_val: {}'.format(epoch, acc_val, best_acc_val))
                    # load best model
                    checkpoint = torch.load('checkpoint_inc12.pt')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        print('acc_val: {}'.format(acc_val))
        f1_val, recall_val, precision_val = f1_isr(output, val_labels)
        print('f1_val_my: {}'.format(f1_val))
    return model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val


def test_regression(model, all_test_features, all_test_labels, test_features, test_labels):
    model.eval()
    output_test_all = model(all_test_features)

    micro_test_all, macro_test_all = f1(output_test_all, all_test_labels)
    f1_test_all, recall_test_all, precision_test_all = f1_isr(output_test_all, all_test_labels)

    output_in_test = model(test_features)
    micro_test, macro_test = f1(output_in_test, test_labels)
    f1_test, recall_test, precision_test = f1_isr(output_in_test, test_labels)

    print(f'macro_test_all: {macro_test_all}, f1_test_all: {f1_test_all}, macro_test: {macro_test}, f1_test: {f1_test}')

    return macro_test_all, f1_test_all, macro_test, f1_test



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

    gamma = 2.0
    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index=adj)
        output = output[selected_nodes, :]
        # print(f'output.size(): {output.size()}')

        # loss_train = F.cross_entropy(output, train_labels)
        loss_train = F.nll_loss(output, train_labels)
        # loss_train = loss_function_laplacian_regularization(output, train_labels, adj)
        # loss_train = loss_function_consistency_regularization(model, features, adj, train_labels, selected_nodes)
        loss_train = loss_function_subgraph_regularization(model, features, adj, train_labels, selected_nodes)
        # loss_train = loss_function_local_consistency(model, features, adj, train_labels, selected_nodes)

        # loss_train.backward()
        loss_train.backward(retain_graph=True)
        optimizer.step()
        

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        output = output[val_nodes, :]
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val = f1(output, val_labels)
        # print('macro_val: {}'.format(macro_val))
        f1_val, recall_val, precision_val = f1_isr(output, val_labels)
        # print('f1_val_isr: {}'.format(f1_val))
    return model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val

def test_GCN(model, adj, features, test_mask, test_labels, all_test_idx, all_test_labels, save_name=None, dataset_name=None, sample_global=False):
    model.eval()
    output_all = model(features, adj)

    output_test_all = output_all[all_test_idx, :]
    output_test_all_preds = output_test_all.max(1)[1]
    if sample_global:
        path = "./spammer_results/"+dataset_name+"/"+save_name+"_all_sample_global.txt"
    else:
        path = "./spammer_results/"+dataset_name+"/"+save_name+"_all.txt"

    with open(path, 'w') as file:
        for i, pred in zip(all_test_idx, output_test_all_preds):
            file.write(f'{i} {pred}\n')

    output_in_test = output_all[test_mask, :]
    output_in_test_preds = output_in_test.max(1)[1]
    if sample_global:
        path = "./spammer_results/"+dataset_name+"/"+save_name+"_sample_global.txt"
    else:
        path = "./spammer_results/"+dataset_name+"/"+save_name+".txt"
    
    with open(path, 'w') as file:
        for i, pred in zip(test_mask, output_in_test_preds):
            file.write(f'{i} {pred}\n')
    
    micro_test_all, macro_test_all = f1(output_test_all, all_test_labels)
    f1_test_all, recall_test_all, precision_test_all = f1_isr(output_test_all, all_test_labels)

    micro_test, macro_test = f1(output_in_test, test_labels)
    f1_test, recall_test, precision_test = f1_isr(output_in_test, test_labels)

    print(f'macro_test_all: {macro_test_all}, f1_test_all: {f1_test_all}, macro_test: {macro_test}, f1_test: {f1_test}')

    return macro_test_all, f1_test_all, macro_test, f1_test


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
            
            # 获取最大的节点编号
            max_node = max(graph_data.max(), graph_data.max())  # 确保编号从0到max_node

            # 手动添加孤立节点，确保包含所有节点
            for node in range(max_node + 1):
                if node not in self.nx_G:
                    self.nx_G.add_node(node)

            self.graph = self.nx_G

            print("start constructing adj")
            edge_tensor = torch.from_numpy(graph_data).long()
            indices = edge_tensor.t().contiguous()
            num_edges = edge_tensor.shape[0]
            values = torch.ones(num_edges)
            num_nodes = edge_tensor.max().item() + 1
            adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
            adj = adj.coalesce()
            # adj = adj.to('cuda:0')
            adj = adj.cuda()
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()
            row_sum[row_sum == 0] = 1  # 避免除以零
            values_normalized = 1.0 / row_sum[adj.indices()[0]]
            adj_normalized = torch.sparse_coo_tensor(adj.indices(), values_normalized, adj.size())
            self.adj = adj_normalized
            print(self.adj)

            print("start loading features")
            
            features = np.loadtxt(args.data_path+"UserFeature.txt", delimiter='\t')
            # features = augment_feature(features, self.nx_G)
            self.features = torch.from_numpy(features).float().cuda()

            print("start loading labels")
            labels_data = pd.read_csv(args.data_path+"UserLabel.txt", sep=' ', usecols=[1, 2])
            labels_data = labels_data.to_numpy()
            self.labels = torch.from_numpy(labels_data[:, 1]).cuda()
            

            training_data = np.loadtxt(args.data_path+"Training_Testing/"+args.test_percents+"/train_4.csv", delimiter=' ', dtype=int)
            testing_data = np.loadtxt(args.data_path+"Training_Testing/"+args.test_percents+"/test_4.csv", delimiter=' ', dtype=int)
    
            self.idx_test = torch.from_numpy(testing_data[:,0] - 1).cuda()

            self.idx_non_test = (training_data[:,0]-1).tolist() 

            self.idx_test_ori = torch.from_numpy(testing_data[:,0] - 1).cuda()

        self.dataset = dataset
        print(f'self.labels: {self.labels, self.labels.shape}')
        print(f'self.adj: {self.adj}')
        print(f'self.feature: {self.features, self.features.shape}')
        print(f'self.idx_test is {len(self.idx_test)}, self.idx_non_test is {len(self.idx_non_test)}')
        print('finished loading dataset')
        self.raw_features = self.features
        self.nb_nodes = self.features.size(0)
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

        batch_size = 1
        dgi_lr = 0.001
        dgi_weight_decay = 0.0
        dgi_epoch = 1000
        best_loss = 1e9
        best_iter = 0
        cnt_wait = 0
        patience = 20
        b_xent = torch.nn.BCEWithLogitsLoss()
        ft_size = self.raw_features.size(1)
        nb_nodes = self.raw_features.size(0)
        features = self.raw_features[np.newaxis]

        DGI_model = DGI(ft_size, 128, 'prelu')
        for name, param in DGI_model.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        opt = optim.Adam(DGI_model.parameters(), lr=dgi_lr,
                               weight_decay=dgi_weight_decay)
        DGI_model.train()
        print('Training unsupervised model.....')
        for i in range(dgi_epoch):
            opt.zero_grad()

            perm_idx = np.random.permutation(self.nb_nodes)
            shuf_fts = features[:, perm_idx, :]


            lbl_1 = torch.ones(batch_size, nb_nodes)
            lbl_2 = torch.zeros(batch_size, nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1)
            if torch.cuda.is_available():
                DGI_model.cuda()
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            logits = DGI_model(features, shuf_fts, self.adj, True, None, None, None)

            loss = b_xent(logits, lbl)


            if loss.item() < best_loss:
                best_loss = loss.item()
                best_iter = i
                cnt_wait = 0
                torch.save(DGI_model.state_dict(), 'best_dgi_inc11.pkl')
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early Stopping')
                break

            loss.backward()
            opt.step()

        print(f'Finished training unsupervised model, Loading {best_iter}th epoch')

        DGI_model.load_state_dict(torch.load('best_dgi_inc11.pkl'))
        self.features, _ = DGI_model.embed(features, self.adj, True, None)
        self.features = torch.squeeze(self.features, 0)

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

        model = get_model('distance_based', self.features.size(1), self.labels.max().item() + 1, args.hidden, 
                          args.dropout, args.cuda)
        # model.reset_parameters()
        budget = 20
        steps = 6
        pool = idx_non_test
        print('len(idx_non_test): {}'.format(len(idx_non_test)))
        np.random.seed() # cancel the fixed seed
        
        if args.sample_global:
            all_test_idx = list(set(self.idx_test).union(set(pool)))
            pool = list(set(self.idx_test.cpu().numpy().tolist()).union(set(pool)))
            test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))
        else:
            all_test_idx = list(set(self.idx_test).union(set(pool)))
            test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))

        
        model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_regression(model, self.features[selected_nodes], self.labels[selected_nodes], self.features[idx_val], self.labels[idx_val], args.epochs, args.weight_decay, args.lr, args.dropout)

        print('-------------initial results------------')
        print('micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
        # Active learning
        print('strategy: ', strategy)
        cur_num = 0
        val_results = {'acc': [], 'micro': [], 'macro': [], 'f1': [], "recall":[], "precision":[]}
        test_results = {'macro_test_all': [], 'f1_test_all': [], 'macro_test': [], 'f1_test': []}


        time_AL = 0
        fixed_medoids = []
        for i in range(len(num_labeled_list)):
            budget = num_labeled_list[i]
            u_features = model.new_features(self.features)
            if args.feature == 'cat':
                if args.adaptive == 1:
                    alpha = 0.99 ** num_labeled_list[i]
                    beta = 1 - alpha
                    print(f'alpha: {alpha}, beta: {beta}')
                    dis_features = torch.cat((alpha * F.normalize(self.features, p=1, dim=1), beta * F.normalize(u_features, p=1, dim=1)), dim=1)
                else:
                    dis_features = torch.cat((F.normalize(self.features, dim=1), F.normalize(u_features, dim=1)), dim=1)
            else:
                dis_features = u_features
            t1 = perf_counter()
            if strategy == 'LSCALE':
                idx_train, original_medoids = query_ours_increment(dis_features, model, budget, fixed_medoids, pool, reweight_flag=args.reweight)
                # idx_train, original_medoids = query_ours(dis_features, model, budget, pool, reweight_flag=args.reweight)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))

            time_AL += perf_counter() - t1
            #print(f'selected_nodes: {selected_nodes}')
            #print(f'idx_train: {idx_train}')
            ensure_nonrepeat(idx_train, selected_nodes)
            selected_nodes = np.append(selected_nodes, idx_train)
            fixed_medoids.extend(original_medoids)
            #print(f'fixed_medoids: {fixed_medoids}')
            assert len(fixed_medoids) == budget

            # pool = list(set(pool) - set(idx_train) - set(fixed_medoids))

            if args.sample_global:
                print("============sample global=======")
                all_test_idx = list(set(pool))
                test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()).intersection(set(pool)))
                print(len(test_idx_in_test))
                print(len(all_test_idx))
            else:
                print("============sample only in training=======")
                test_idx_in_test = list(set(self.idx_test.cpu().numpy().tolist()))
                all_test_idx = list(set(pool).union(test_idx_in_test)- set(selected_nodes))
                print(len(test_idx_in_test))
                print(len(all_test_idx))
            

            model, acc_val, micro_val, macro_val, train_time, f1_val, recall_val, precision_val = train_regression(model, self.features[selected_nodes],
                                                                                self.labels[selected_nodes],
                                                                                self.features[idx_val],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)
            print(f"the number of labels is {num_labeled_list[i]}")
            if args.model == 'GCN':
                macro_test_all, f1_test_all, macro_test, f1_test = test_GCN(model, self.adj, self.features, test_idx_in_test, self.labels[test_idx_in_test], all_test_idx, self.labels[all_test_idx], save_name=args.test_percents, dataset_name=args.dataset, sample_global=args.sample_global)
            
            macro_test_all, f1_test_all, macro_test, f1_test = test_regression(model, self.features[all_test_idx], self.labels[all_test_idx], self.features[test_idx_in_test], self.labels[test_idx_in_test])

            print('f1_val_isr: {}'.format(f1_val))
            print('f1_test_isr: {}'.format(f1_test))

            macro_test = round(macro_test, 4)
            f1_test = round(f1_test, 4)
            macro_test_all = round(macro_test_all, 4)
            f1_test_all = round(f1_test_all, 4)

            test_results['macro_test_all'].append(macro_test_all)
            test_results['f1_test_all'].append(f1_test_all)
            test_results['macro_test'].append(macro_test)
            test_results['f1_test'].append(f1_test)

        print('AL Time: {}s'.format(time_AL))
        return val_results, test_results, get_classes_statistic(self.labels[selected_nodes].cpu().numpy()), time_AL



if __name__ == '__main__':

    if args.dataset == 'spammer':
        num_labeled_list = [i for i in range(10,151,10)]
    elif args.dataset == 'amazon':
        if args.test_percents in ['50percent', '30percent', '10percent']:
            num_labeled_list = [i for i in range(10,721,10)]
        else:
            num_labeled_list = [i for i in range(10,401,10)]
    elif args.dataset == 'yelp':
        num_labeled_list = [10, 20, 30, 40] + [i for i in range(50,1001,50)]
    num_interval = len(num_labeled_list)

    val_results = {'micro': [[] for _ in range(num_interval)],
                   'macro': [[] for _ in range(num_interval)],
                   'acc': [[] for _ in range(num_interval)],
                   'f1': [[] for _ in range(num_interval)],
                   'recall': [[] for _ in range(num_interval)],
                   'precision': [[] for _ in range(num_interval)]}

    test_results = {'macro_test_all': [[] for _ in range(num_interval)],
                    'f1_test_all': [[] for _ in range(num_interval)],
                    'macro_test': [[] for _ in range(num_interval)],
                    'f1_test': [[] for _ in range(num_interval)]}
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
        
        for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
            for j in range(len(test_dict[metric])):
                test_results[metric][j].append(test_dict[metric][j])

        total_AL_time += cur_AL_time

    test_avg_results = {'macro_test_all': [0. for _ in range(num_interval)],
                    'f1_test_all': [0. for _ in range(num_interval)],
                    'macro_test': [0. for _ in range(num_interval)],
                    'f1_test': [0. for _ in range(num_interval)]}

    for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
        for j in range(len(test_results[metric])):
            test_avg_results[metric][j] = np.mean(test_results[metric][j])

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
        for metric in ['macro_test_all', 'f1_test_all', 'macro_test', 'f1_test']:
            f.write("Test_{}_macro {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_results[metric][0])))
        

        f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))
    
    if args.sample_global:
        plot(num_labeled_list, test_avg_results['macro_test_all'], args.test_percents+args.save_name+"macro_test_all_global")
        plot(num_labeled_list, test_avg_results['f1_test_all'], args.test_percents+args.save_name+"f1_test_all_global")
        plot(num_labeled_list, test_avg_results['macro_test'], args.test_percents+args.save_name+"macro_test_global")
        plot(num_labeled_list, test_avg_results['f1_test'], args.test_percents+args.save_name+"f1_test_global")
    else:
        plot(num_labeled_list, test_avg_results['macro_test_all'], args.test_percents+args.save_name+"macro_test_all")
        plot(num_labeled_list, test_avg_results['f1_test_all'], args.test_percents+args.save_name+"f1_test_all")
        plot(num_labeled_list, test_avg_results['macro_test'], args.test_percents+args.save_name+"macro_test")
        plot(num_labeled_list, test_avg_results['f1_test'], args.test_percents+args.save_name+"f1_test")
