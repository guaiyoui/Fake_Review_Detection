import random
import string
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# path = './dataset/'
path = '../Data/'
# rf_features = 'userfeatures2021.txt'#Read User Feature for LR Training
rf_features = 'UserFeature.txt'#Read User Feature for LR Training
# rf_label = '5percent/test_4.csv'  #Load Testing data: index and label; You should revise when training data is changed.
# rf_label = 'Training_Testing/5percent/test_4.csv'  #Load Testing data: index and label; You should revise 
# rf_label = 'Training_Testing/10percent/test_4.csv'  #Load Testing data: index and label; You should revise 
# rf_label = 'Training_Testing/30percent/test_4.csv'  #Load Testing data: index and label; You should revise 
rf_label = 'Training_Testing/50percent/test_4.csv'  #Load Testing data: index and label; You should revise 


# when training data is changed.


def LR_First(L1, L2, L3):
    print("Going to Python.......")
    print(len(L1))
    print("________________________________________")
    print(len(L2))
    print("________________________________________")
    print(len(L3))
    # print("########## I am here. #############")
    data = pd.read_csv(path+rf_features, sep='\t', header=None)
    print(data.shape)
    data_test = pd.read_csv(path+rf_label, sep='\t', header=None)
    data.index = data.index.values 
    #Train_Data_Preprocessor
    temp1 = np.zeros((len(L1), data.shape[1]))
    data_train = pd.DataFrame(temp1)  
    index = 0
    index_number = 0
    print(data_train.shape)
    for index_number in range(len(L1)):
        temp2 = list(data.iloc[L1[index_number]-1, :].values)  
        data_train.iloc[index, :] = temp2  
        index += 1
    print(index)
    
    data_train = np.array(data_train.values)
    #Test_Data_Preprocessor
    test_label = []
    test_index_list = []
    for index, row in data_test.iterrows():
        temp = row.values.tolist()
        index_, label = temp[0].split(' ')
        test_label.append(int(label))
        test_index_list.append(int(index_))
    
    print(len(test_index_list))
    #Extract_Test_Data
    index_number = 0
    temp1 = np.zeros((len(test_index_list), data.shape[1]))
    data_test = pd.DataFrame(temp1)
    for index_number in range(len(test_index_list)):
        temp2 = list(data.iloc[test_index_list[index_number]-1, :].values)
        data_test.iloc[index_number, :] = temp2
    #print(data_test)
    data_test = data_test.values
    print(data_test.shape)
    #LR
    # model = LogisticRegression(penalty='none', max_iter=10000,solver='saga')
    model = LogisticRegression(penalty='l2', max_iter=10000,solver='saga')
    print("Training Data Size:")
    print(data_train.shape, np.array(L2).shape, np.array(L3).shape)
    # X_train: Training data， Y_train: Training Labels
    # print(data_train, L2, L3)
    # model.fit(data_train, np.array(L2), np.array(L3))
    model.fit(data_train, L2, L3)
    
    pre_Y = list(model.predict(data_test))
    pre_prob = model.predict_proba(data_test)
    # print(pre_Y, pre_prob)
    new_pre_porb = list(pre_prob[:, 1])
    report = classification_report(test_label, pre_Y)
    print(report)
    return test_index_list,new_pre_porb
