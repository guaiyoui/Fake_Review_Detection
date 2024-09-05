# python run_baselines.py --dataset spammer --model GCN --epoch 300 --strategy random --file_io 1 --lr 0.01 --hidden 16

# import pickle as pkl
# import sys
# import numpy as np

num_labeled_list = [10, 20, 30, 40] + [i for i in range(50,1001,50)]
print(num_labeled_list)

# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index

# names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
# objects = []

# dataset_str = "cora"
# for i in range(len(names)):
#     with open("./data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
#         if sys.version_info > (3, 0):
#             objects.append(pkl.load(f, encoding='latin1'))
#         else:
#             objects.append(pkl.load(f))

# x, y, tx, ty, allx, ally, graph = tuple(objects)
# test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
# test_idx_range = np.sort(test_idx_reorder)

# print(y)
for i in range(10,151,10):
    print(i)