import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os
import datetime
import json
import pandas as pd
from torch_geometric.utils import get_laplacian
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def plot(index, data1, index2, data2, index3, data3, figure_name):

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(index, data1, label='Our Method')
    plt.plot(index2, data2, label='ISR Method')
    plt.plot(index3, data3, label='Our Method', color='red')

    # 找出每组数据的最小值、最大值和最后的值
    min_value1 = min(data1)
    max_value1 = max(data1)
    last_value1 = data1[-1]
    min_index1 = data1.index(min_value1)
    max_index1 = data1.index(max_value1)
    last_index1 = len(data1) - 1

    min_value2 = min(data2)
    max_value2 = max(data2)
    last_value2 = data2[-1]
    min_index2 = data2.index(min_value2)
    max_index2 = data2.index(max_value2)
    last_index2 = len(data2) - 1

    # 标注数据1的最小值（左上角）
    plt.annotate(f'Data 1 Min: {min_value1}', 
                 xy=(min_index1, min_value1), 
                 xytext=(0.05, 0.95), 
                 textcoords='axes fraction')

    # 标注数据1的最大值（中间上方）
    plt.annotate(f'Data 1 Max: {max_value1}', 
                 xy=(max_index1, max_value1), 
                 xytext=(0.5, 0.95), 
                 textcoords='axes fraction',
                 ha='center')

    # 标注数据1的最后的值（右上角）
    plt.annotate(f'Data 1 Last: {last_value1}', 
                 xy=(last_index1, last_value1), 
                 xytext=(0.95, 0.95), 
                 textcoords='axes fraction',
                 ha='right')

    # 标注数据2的最小值（左下角）
    plt.annotate(f'Data 2 Min: {min_value2}', 
                 xy=(min_index2, min_value2), 
                 xytext=(0.05, 0.05), 
                 textcoords='axes fraction')

    # 标注数据2的最大值（中间下方）
    plt.annotate(f'Data 2 Max: {max_value2}', 
                 xy=(max_index2, max_value2), 
                 xytext=(0.5, 0.05), 
                 textcoords='axes fraction',
                 ha='center')

    # 标注数据2的最后的值（右下角）
    plt.annotate(f'Data 2 Last: {last_value2}', 
                 xy=(last_index2, last_value2), 
                 xytext=(0.95, 0.05), 
                 textcoords='axes fraction',
                 ha='right')
    
    # 设置标题和轴标签
    plt.title(figure_name)
    plt.xlabel('Num of trained samples')
    plt.ylabel('Value')
    plt.legend()

    # 保存图表
    plt.savefig("./figures/"+figure_name+".png")
    plt.close()

f1_all = []
f1_test = []
data_path = "../Active_Spammer/logs/loss_5percent_global.txt"
with open(data_path, 'r') as file:
    for line in file:
        if 'f1_test_all:' in line and 'f1_test:' in line:
            data = line.split(", ")
            num = float(data[1].split(": ")[1])
            f1_all.append(num)
            num2 = float(data[3].split(": ")[1])
            f1_test.append(num2)

print(len(f1_all))


# plot([i*10 for i in range(40)], f1_all, "separate_test")
# plot([i*10 for i in range(40)], f1_test, "intersection_test")


separate_test_isr = [0.777961432506887, 0.8491306786315199, 0.8603603603603602, 0.8633540372670807]
num_train = [7959+400-7955, 7959+400-7537, 7959+400-5862, 7959+400-4187]
intersection_test = [0.8151534944480732, 0.8915662650602408, 0.8988970588235293, 0.899873257287706]

plot([i*10 for i in range(40)], f1_all, num_train, separate_test_isr, [(i+40)*10 for i in range(int((7959+400-4187)/10)-40)], [0.9293413173652694 for i in range(int((7959+400-4187)/10)-40)], "separate_test_compare")
plot([i*10 for i in range(40)], f1_test, num_train, intersection_test, [(i+40)*10 for i in range(int((7959+400-4187)/10)-40)], [0.9269521410579346 for i in range(int((7959+400-4187)/10)-40)],"intersection_test_compare")