import utils
# import dgl
import torch
import scipy.sparse as sp
import numpy as np 

# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
# from dgl.data import  AmazonCoBuyPhotoDataset,CoauthorCSDataset,CoauthorPhysicsDataset


def edge_index_to_sparse_coo(edge_index):
    row = edge_index[0].long()
    col = edge_index[1].long()

    # 构建稀疏矩阵的形状
    num_nodes = torch.max(edge_index) + 1
    size = (num_nodes.item(), num_nodes.item())
    # size = (18333, 18333)
    print(edge_index, num_nodes)
    # 构建稀疏矩阵
    values = torch.ones_like(row)
    edge_index_sparse = torch.sparse_coo_tensor(torch.stack([row, col]), values, size)

    return edge_index_sparse


def get_dataset(dataset, pe_dim):
    # if dataset in {"pubmed", "photo", "cs", "cora", "physics","citeseer"}:
    #     if dataset in {"photo", "cs"}:
    #         file_path = "dataset/"+dataset+"_dgl.pt"
    #     else:
    #         file_path = "dataset/"+dataset+"_pyg.pt"
    #     # file_path = "dataset/"+dataset+".pt"
    #     data_list = torch.load(file_path)
        
    #     adj = data_list[0]
        
    #     features = data_list[1]
        
    #     if dataset == "pubmed":
    #         graph = PubmedGraphDataset()[0]
    #     elif dataset == "photo":
    #         graph = AmazonCoBuyPhotoDataset()[0]
    #     elif dataset == "cs":
    #         graph = CoauthorCSDataset()[0]
    #     elif dataset == "physics":
    #         graph = CoauthorPhysicsDataset()[0]
    #     elif dataset == "cora":
    #         graph = CoraGraphDataset()[0]
    #     elif dataset == "citeseer":
    #         graph = CiteseerGraphDataset()[0]

    #     # print(graph)
    #     graph = dgl.to_bidirected(graph)
        
    #     lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
     
    #     features = torch.cat((features, lpe), dim=1)
    
    
    # elif dataset in {"texas", "cornell", "wisconsin", "dblp", "reddit"}:
    #     file_path = "dataset/"+dataset+"_pyg.pt"

    #     data_list = torch.load(file_path)
       
    #     adj = data_list[0]
        
    #     features = data_list[1]
        
    #     adj_scipy = utils.torch_adj_to_scipy(adj)
    #     graph = dgl.from_scipy(adj_scipy)
    #     lpe = utils.laplacian_positional_encoding(graph, pe_dim) 
    #     features = torch.cat((features, lpe), dim=1)
    
    if dataset in {"spammer"}:
        graph_data = np.loadtxt("./data_graph/spammer_edge_index.txt", delimiter=' ', dtype=int)
        features = np.loadtxt("./data_graph/spammer_feature.txt", delimiter='\t')
        graph_data[:,0] = graph_data[:,0]-1
        graph_data[:,1] = graph_data[:,1]-1

        print(graph_data)
        edges = torch.from_numpy(graph_data).T
        features = torch.from_numpy(features).float()

        edge0 = edges[0].tolist() 
        edge0 = [int(i) for i in edge0]
        edge1 = edges[1].tolist() 
        edge1 = [int(i) for i in edge1]
        edge0_double = edge0+edge1
        edge1_double = edge1+edge0
        edges = torch.Tensor([edge0_double, edge1_double]).type(torch.int)
        # print(edges)
        adj = edge_index_to_sparse_coo(edges)

    print(type(adj), type(features))
    
    return adj.cpu().type(torch.LongTensor), features.long()




