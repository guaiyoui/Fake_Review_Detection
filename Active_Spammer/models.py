import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, MixHopConv, GINConv, MLP, GATConv, LayerNorm, GraphNorm
from torch.nn import BatchNorm1d
# from layer import GCNConv

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


def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        # model = GCN_Classifier(nfeat=nfeat,
        #             nhid=nhid,
        #             nclass=nclass,
        #             dropout=dropout)

        # model = GCN_adv(num_features=nfeat,
        #             num_classes=nclass)

        # model = GIN(num_features=nfeat,
        #             num_classes=nclass)
        
        model = GIN_adv(num_features=nfeat,
                    num_classes=nclass)
        
        # model = Net(num_features=nfeat,
        #             num_classes=nclass)

        # model = GAT(num_features=nfeat,
        #             num_classes=nclass)

    if cuda: model.cuda()
    return model
