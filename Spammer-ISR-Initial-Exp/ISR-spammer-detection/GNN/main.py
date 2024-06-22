import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    training_data = np.loadtxt("../Data/Training_Testing/5percent/train_4.csv", delimiter=' ', dtype=int)
    testing_data = np.loadtxt("../Data/Training_Testing/5percent/test_4.csv", delimiter=' ', dtype=int)
    graph_data = np.loadtxt("../Data/J01Network.txt", delimiter=' ', dtype=int)
    feature = np.loadtxt("../Data/UserFeature.txt", delimiter='\t')

    # 把index转化成从0开始的
    training_data[:,0] = training_data[:,0]-1
    testing_data[:,0] = testing_data[:,0]-1
    graph_data[:,0] = graph_data[:,0]-1
    graph_data[:,1] = graph_data[:,1]-1

    y = torch.full((feature.shape[0],), -1, dtype=torch.long)  # 初始化所有标签为-1
    y[training_data[:, 0]] = torch.from_numpy(training_data[:, 1])
    y[testing_data[:, 0]] = torch.from_numpy(testing_data[:, 1])
    

    train_mask = torch.zeros(feature.shape[0], dtype=torch.bool)
    train_mask[training_data[:, 0]] = True

    test_mask = torch.zeros(feature.shape[0], dtype=torch.bool)
    test_mask[testing_data[:, 0]] = True

    edge_index = torch.from_numpy(graph_data)
    data = Data(x=torch.from_numpy(feature).float(), edge_index=edge_index.t().contiguous().long(), y=y, train_mask=train_mask, test_mask=test_mask)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = GCN(feature.shape[1], num_classes=2).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    
    # 评估模型
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(testing_data.shape[0]):
        if pred[testing_data[i, 0]]==1 and y[testing_data[i, 0]]==1:
            tp += 1
        elif pred[testing_data[i, 0]]==0 and y[testing_data[i, 0]]==1:
            fn += 1
        elif pred[testing_data[i, 0]]==1 and y[testing_data[i, 0]]==0:
            tp += 1
        elif pred[testing_data[i, 0]]==0 and y[testing_data[i, 0]]==0:
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







