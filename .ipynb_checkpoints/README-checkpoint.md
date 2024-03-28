这里支持两种操作，一种是给定一个query找knn相似的review, 另外一种是聚类的方法把review分成几类。

0: File Description
--comment_data.csv: 存所有的review的数据
--comment_data.csv: 存query review的数据
--results: 最终所有找到的结果
--query_knn.py: 给定一个query review找knn相似的review
--cluster.py: 聚类的方法把review分成几类


1: Clustering

python clustering.py --cluster_num 100


2: Query KNN

python query_knn.py --top_k 20
