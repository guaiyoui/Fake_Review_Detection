from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression(penalty='l2', max_iter=10000,solver='saga')

data_train = np.random.rand(100, 5)
L2 = np.random.rand(100)
L3 = np.random.rand(100)

print(data_train.shape, L2.shape, L3.shape)
model.fit(data_train, L2, L3)
