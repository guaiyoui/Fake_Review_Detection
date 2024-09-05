from sklearn.metrics import f1_score
import torch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # return correct / len(labels)
    result = correct / len(labels)
    return result.cpu().detach().numpy().item()

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def f1_my(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    print(preds.shape, labels.shape)
    # print(output, preds, labels)
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(preds.shape[0]):
        # print(preds[i], labels[i])
        if preds[i]==1 and labels[i]==1:
            tp += 1
        elif preds[i]==0 and labels[i]==1:
            fn += 1
        elif preds[i]==1 and labels[i]==0:
            fp += 1
        elif preds[i]==0 and labels[i]==0:
            tn += 1
        else:
            raise ValueError("the category number is incorrect")
    
    if tp+fn == 0:
        recall = tp/(tp+fn+0.0001)
    else:
        recall = tp/(tp+fn)

    if tp+fp == 0:
        precision = tp/(tp+fp+0.0001)
    else:
        precision = tp/(tp+fp)
    if recall+precision == 0:
        return 2*recall*precision /(recall + precision + 0.0001), recall, precision
    else:
        return 2*recall*precision /(recall + precision), recall, precision

def f1_my_micro(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    
    tp, fp, fn = 0, 0, 0
    for pred, label in zip(preds, labels):
        if pred == label:
            tp += 1
        else:
            fp += 1
            fn += 1
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1, recall, precision 

output = torch.Tensor([[0.1, 0.9],
                  [0.3, 0.7],
                  [0.6, 0.4],
                  [0.8, 0.2]])

labels = torch.Tensor([0, 0, 1, 0])
print(f1(output, labels))
print(f1_my(output, labels))
print(f1_my_micro(output, labels))
