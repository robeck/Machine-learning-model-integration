''':
@Author: haozhi chen
@Date: 2022-10
@Target: we implement to do feature pre-selection based on RandomForest algorithm

一般而言，RF方法可以进行分类，回归等预测工作。但是，根据一些相关研究，以及我们在Sklearn包总发现的其中features_importance_ 参数能够为我们提供
一个额外的特征选择视角，因此，我们在这里进行模拟实验


'''

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

'''从sklearn的数据库导入数据
1）导入iris数据
2）重构数据为dataframe结构，便于使用
'''
def data_import():
    iris = datasets.load_iris()
    boston = datasets.load_boston()

    iris_data = iris.get('data')
    iris_target = iris.get('target')

    colnames = ['setosa','versicolor','virginica','feature4']
    irisdf = pd.DataFrame(iris_data,columns=colnames)
    irisdf['target'] = iris_target
    print(irisdf)

    return colnames,irisdf




def Model_implement():
    features,dataframe = data_import()
    "随机森林模型拟合"
    train_X = dataframe[features]
    train_y = dataframe['target']

    rf = RandomForestClassifier(n_estimators=100, max_depth=None)
    rf_pip = Pipeline([('Standardize',StandardScaler()),
                       ('rf',rf)])
    rf_pip.fit(train_X,train_y)

    "根据随机森林拟合结果选择特征"
    rf = rf_pip.__getitem__('rf') # 获取pipline中的RandomForest模型
    features_importance = rf.feature_importances_ # 获取feature 重要结果
    print(features_importance)
    index_sort = np.argsort(features_importance)[::-1] # np.argsort() 给出了数据从小到大排序的索引位置，[::-1]转换为从大到小

    "循环输出重要性结果-特征名称"
    for i in index_sort:
        print('特征 %s 的重要性得分为 %f'%(features[i],features_importance[i]))
    
    
    feat_labels = [features[i] for i in index_sort] # 按顺序的结果
    "绘制以下"
    fig = plt.figure(figsize=(12,8))
    plt.bar(range(len(index_sort)),features_importance[index_sort],color='lightblue',align='center')
    plt.xticks(range(len(index_sort)),feat_labels,rotation=90)
    plt.xlim([-1,len(index_sort)])
    plt.show()

    return None




if __name__ == '__main__':
    Model_implement()