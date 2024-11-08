import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, MixHopConv, GINConv, MLP, GATConv, LayerNorm, GraphNorm
from torch.nn import BatchNorm1d
# from layer import GCNConv
from layers import AvgReadout, Discriminator, GCN
from torch.nn.parameter import Parameter

class distance_based(nn.Module):
    """
    distance_based classifier.
    The input feature should be DGI features.
    """
    def __init__(self, nfeat, nembed, nclass):
        super(distance_based, self).__init__()
        self.nfeat = nfeat
        self.nembed = nembed
        self.nclass = nclass
        self.W = nn.Linear(nfeat, nembed)
        self.class_embed = nn.Embedding(nclass, nembed)

    def forward(self, x):
        u = self.W(x)
        num_nodes = u.size(0)
        u = u.view(num_nodes, -1, self.nembed)
        class_embed = self.class_embed.weight.view(-1, self.nclass, self.nembed)
        distances = torch.norm(u - class_embed, dim=-1)
        return distances

    def new_features(self, x):
        u = self.W(x)
        return u.detach()


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        # self.fc = nn.Linear(n_in, n_h)
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.act = nn.PReLU()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, x_1, x_2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(x_1, adj, sparse)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        h_2 = self.gcn(x_2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        # h_1 = self.sigm(self.fc(seq))
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print(f'in_feat: {in_features}, out_feat: {out_features}')
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = self.W(input)
        output = torch.spmm(adj, support)
        return output

class GCN_Classifier(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class GINConv(nn.Module):
    """
    A Graph Isomorphism Network Layer (GIN)
    """
    def __init__(self, in_features, out_features, eps=0, train_eps=False):
        super(GINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        print(f'in_feat: {in_features}, out_feat: {out_features}')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if not train_eps:
            self.eps.requires_grad = False
        
        self.init()

    def init(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # GIN aggregation
        support = (1 + self.eps) * x + torch.spmm(adj, x)
        output = self.mlp(support)
        return output

class GIN_Classifier(nn.Module):
    """
    A Two-layer GIN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GIN_Classifier, self).__init__()

        self.gin1 = GINConv(nfeat, nhid)
        self.gin2 = GINConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gin1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gin2(x, adj)
        return x


class GCN_adv(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN_adv, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        self.node_embedding = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_node_embedding(self):
        return self.node_embedding

import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=2, hidden_dim=32, eps=0, learn_eps=False):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            eps=eps,
            train_eps=learn_eps
        ))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                eps=eps,
                train_eps=learn_eps
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.jump = nn.Linear(num_layers * hidden_dim, num_classes)

    def forward(self, x, edge_index):
        xs = []
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            xs.append(x)
            self.node_embedding = x

        x = torch.cat(xs, dim=1)
        x = self.jump(x)

        return F.log_softmax(x, dim=-1)
    def get_node_embedding(self):
        return self.node_embedding

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, 64, heads=4)
        self.norm1 = LayerNorm(64)
        self.conv2 = GATConv(64 * 4, 32, heads=4)
        self.norm2 = LayerNorm(32)
        self.conv3 = GCNConv(32 * 4, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        self.node_embedding = x
        # x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_node_embedding(self):
        return self.node_embedding
    
class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True, normalize=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True, normalize=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.crd = CRD(num_features, 32, 0.5)
        self.cls = CLS(32, num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, x, edge_index):
        x = self.crd(x, edge_index)
        x = self.cls(x, edge_index)
        return x

class GIN_adv(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=2, hidden_dim=32, eps=0, learn_eps=False):
        super(GIN_adv, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.jump = nn.Linear(num_layers * hidden_dim, num_classes)

    def forward(self, x, edge_index):
        xs = []
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            xs.append(x)
            self.node_embedding = x

        x = torch.cat(xs, dim=1)
        x = self.jump(x)

        return F.log_softmax(x, dim=-1)
    def get_node_embedding(self):
        return self.node_embedding

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class tGCN(torch.nn.Module):
    def __init__(self, num_features, num_nodes, num_classes, num_layers=2, hidden_dim=32, n_clusters=10):
        super(tGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.autoencoder = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(num_features, hidden_dim))
        self.autoencoder.append(nn.Linear(num_nodes, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.autoencoder.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers-1):
            self.autoencoder.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.autoencoder.append(nn.Linear(hidden_dim, num_nodes))

        self.jump = nn.Linear(num_layers * hidden_dim, num_classes)

        self.v = 1
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, b, x, edge_index):
        xs = []
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            self.node_embedding = x
            xs.append(x)

            b = self.autoencoder[i](b)
            x = x + b
            self.latent_encoding = b

        for i in range(len(self.autoencoder)-len(self.convs)):
            b = self.autoencoder[i+len(self.convs)](b)

        x = torch.cat(xs, dim=1)
        x = self.jump(x)


        return F.log_softmax(x, dim=-1), b

    def compute_reconstruction_loss(self, modularity_matrix, b):
        loss = torch.norm(b - modularity_matrix, p='fro') ** 2
        return loss
    
    def compute_kl_loss(self):
        # 将 Z 转换为对数概率分布（log_softmax），符合KL散度的计算需求
        Z_log_prob = F.log_softmax(self.latent_encoding, dim=-1)
        
        # 将 H 转换为概率分布（softmax）
        H_prob = F.softmax(self.node_embedding, dim=-1)
        
        # 计算 KL 散度
        kl_div = F.kl_div(Z_log_prob, H_prob, reduction='batchmean')
        return kl_div

    def compute_t_loss(self):

        q = 1.0 / (1.0 + torch.sum(torch.pow(self.latent_encoding.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        p = target_distribution(q)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        return kl_loss

    def get_node_embedding(self):
        return self.node_embedding


def get_model(model_opt, nfeat, nsample, nclass, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        # model = GCN_Classifier(nfeat=nfeat,
        #             nhid=nhid,
        #             nclass=nclass,
        #             dropout=dropout)

        # model = GCN_adv(num_features=nfeat,
        #             num_classes=nclass)

        # model = GIN(num_features=nfeat,
        #             num_classes=nclass)
        
        # current best
        # model = GIN_adv(num_features=nfeat,
        #             num_classes=nclass)
        
        model = tGCN(num_features=nfeat,
                    num_classes=nclass,
                    num_nodes=nsample)

        # model = Net(num_features=nfeat,
        #             num_classes=nclass)

        # model = GAT(num_features=nfeat,
        #             num_classes=nclass)
    elif model_opt == 'GCN_update':
        model = GIN_adv(num_features=nfeat,
                    num_classes=nclass)

    elif model_opt == 'distance_based':
        model = distance_based(nfeat=nfeat, nembed=nhid, nclass=nclass)

    if cuda: model.cuda()
    return model
