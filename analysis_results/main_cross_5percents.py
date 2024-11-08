import pandas as pd

from sklearn.metrics import f1_score
import torch
from sklearn.metrics import classification_report


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # return correct / len(labels)
    result = correct / len(labels)
    return result.cpu().detach().numpy().item()

def f1(output, labels):
    # preds = output.max(1)[1]
    preds = output.values
    labels = labels.values
    # preds = preds.cpu().detach().numpy()
    # labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def f1_isr(output, labels):
    # preds = output.max(1)[1]
    # preds = preds.cpu().detach().numpy()
    # labels = labels.cpu().detach().numpy()
    preds = output.values
    labels = labels.values
    # print(f"preds.shape: {preds.shape}, labels.shape: {labels.shape}")
    tp, fn, fp, tn = 0, 0, 0, 0
    for i in range(preds.shape[0]):
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


# path_result_our_all = '../Active_Spammer/spammer_results/amazon/30percent_all_sample_global.txt'
# path_result_our_all = '../Active_Spammer/spammer_results/amazon/30percent_sample_global.txt'
# path_result_our_all = '../Active_Spammer/spammer_results/amazon/30percent_all.txt'
# path_result_our_all = '../Active_Spammer/spammer_results/amazon/30percent.txt'
for percent in ["50percent", "30percent", "10percent", "5percent"]:
    print("#####################\n\n\n {}".format(percent) + "#################\n\n\n")
    path_result_isr = f'../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Code/spammer_results/prediction_{percent}s.txt'
    # path_result_isr = f'../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Code/spammer_results/prediction_5cpercents.txt'
    for path_result_our_all in [f'../Active_Spammer/spammer_results/amazon/5percent_all_sample_global.txt']:
        print("============================================")
        print(path_result_isr)
        result_isr = pd.read_csv(path_result_isr, sep=' ',names=['user_no','isr_pred'])
        result_our30_all = pd.read_csv(path_result_our_all, sep=' ',names=['user_no','our30_all_pred'])
        result_our30_all['user_no'] += 1

        label_ori = pd.read_csv("../Spammer-ISR-Initial-Exp/ISR-spammer-detection/Data/UserLabel.txt", sep=' ')

        result_isr.drop_duplicates(subset=['user_no'], keep='first', inplace=True)
        # print(result_isr)
        result_our30_all.drop_duplicates(subset=['user_no'], keep='first', inplace=True)
        # print(result_our30_all)

        label_isr = label_ori.merge(result_isr, on='user_no', how='left')
        label_isr.dropna(inplace=True)
        label_our = label_ori.merge(result_our30_all, on='user_no', how='left')
        label_our.dropna(inplace=True)
        print("\n xxx The result of separate part of test sets (case 2.2): ")
        # print('ISR')
        # print(classification_report(label_isr['label'], label_isr['isr_pred']))
        # print('Our30')
        # print(classification_report(label_our['label'], label_our['our30_all_pred']))

        print(f"ISR results by F1-one (isr paper used): {f1_isr(label_isr['isr_pred'], label_isr['label'])[0]}, sample num: {len(label_isr['isr_pred'])}")
        print(f"Our results by F1-one (isr paper used): {f1_isr(label_our['our30_all_pred'], label_our['label'])[0]}, len: {len(label_our['our30_all_pred'])}")
        print(f"ISR results by macro-F1: {f1(label_isr['isr_pred'], label_isr['label'])[1]}, sample num: {len(label_isr['isr_pred'])}")
        print(f"Our results by macro-F1: {f1(label_our['our30_all_pred'], label_our['label'])[1]}, sample num: {len(label_our['our30_all_pred'])}")

        label = label_ori.merge(result_isr, on='user_no', how='left')
        label = label.merge(result_our30_all, on='user_no', how='left')
        label.dropna(inplace=True)


        print("\n xxx The result of intersection part of test sets (case 2.1): ")
        # #report the prediction performance
        # print('ISR')
        # print(classification_report(label['label'], label['isr_pred']))
        # print('Our30')
        # print(classification_report(label['label'], label['our30_all_pred']))

        print(f"ISR results by F1-one (isr paper used): {f1_isr(label['isr_pred'], label['label'])[0]}, sample num: {len(label['isr_pred'])}")
        print(f"Our results by F1-one (isr paper used): {f1_isr(label['our30_all_pred'], label['label'])[0]}, sample num: {len(label['our30_all_pred'])}")
        print(f"ISR results by macro-F1: {f1(label['isr_pred'], label['label'])[1]}, sample num: {len(label['isr_pred'])}")
        print(f"Our results by macro-F1: {f1(label['our30_all_pred'], label['label'])[1]}, sample num: {len(label['our30_all_pred'])}")
    
