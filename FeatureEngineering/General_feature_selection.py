''':
@Author: haozhi chen
@Date: 2022-11
@Target: 测试基础的feature filter方法

'''

from sklearn.datasets import load_boston,load_breast_cancer # 导入boston房地产，乳腺癌数据
from sklearn.feature_selection import GenericUnivariateSelect,mutual_info_classif,mutual_info_regression,chi2,RFE,RFECV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt


def breast_cancer_test():

    trainX,trainY = load_breast_cancer(return_X_y=True)
    trainX,testX,trainY,testY = train_test_split(trainX, trainY, random_state=10, test_size=0.3)
    print(f'初始的数据维度{trainX.shape}')

    "1. filter方法"
    filter_transformer = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile',param=70)
    # filter_transformer = GenericUnivariateSelect(score_func=chi2, mode='percentile', param=80)
    '''
    score_func = mutual_info_classif: 互信息熵分类方法
                 chi2: 卡方检验方法
    param：筛选剩余的比例，其实就是一个Threshold 阈值
    '''
    newtrainX = filter_transformer.fit_transform(trainX,trainY)
    newtrainX,newtestX,newtrainY,newtestY = train_test_split(newtrainX, trainY, random_state=10, test_size=0.3)
    print(f'经过filter方法后的X数据维度{newtrainX.shape}')

    "筛选前测试"
    rfc = RandomForestClassifier(n_estimators=20,n_jobs=-1,random_state=0)
    # rfc.fit(trainX,trainY)
    # train_scores = rfc.score(trainX,trainY)
    # print(f'model train scores in basesline datasets are{train_scores}') # 训练数据的得分必然是 1.0
    # scores = rfc.score(testX, testY)
    # print(f'model train-test scores on baseline datasets are {scores}')
    # "筛选后测试"
    # rfc.fit(newtrainX,newtrainY)
    # scores = rfc.score(newtestX,newtestY)
    # print(f'model train-test scores on filtered datasets are {scores}')
    # "多测试结果的绘制！"
    # scoreslist = []
    # paramlist = []
    # for param in range(10,95,5):
    #     filter_transformer = GenericUnivariateSelect(score_func=mutual_info_classif,mode='percentile',param=param)
    #     newtrainX = filter_transformer.fit_transform(trainX,trainY)
    #     newtrainX,newtestX,newtrainY,newtestY = train_test_split(newtrainX,trainY,random_state=10, test_size=0.3)
    #     rfc.fit(newtrainX,newtrainY)
    #     scores = rfc.score(newtestX,newtestY)
    #     scoreslist.append(scores)
    #     paramlist.append(param)
    #
    # fig = plt.figure(figsize=(12,8))
    # plt.plot(paramlist,scoreslist,'b--',label='chi2 features filter')
    # plt.title('chi2 features selection based on different params')
    # plt.xlabel('Params')
    # plt.ylabel('Scores')
    # plt.legend()
    # plt.show()

    "2. wrapper方法"
    selector = RFECV(estimator=rfc,step=1,cv=5)
    newtrainX = selector.fit_transform(trainX,trainY)
    print(selector.support_)
    print(selector.ranking_)
    print(selector.support_.__index__())
    print(f'经过wrapper方法后的X数据维度{newtrainX.shape}')
    newtrainX,newtestX,newtrainY,newtestY = train_test_split(newtrainX,trainY,random_state=10,test_size=0.3)
    rfc.fit(newtrainX,newtrainY)
    scores = rfc.score(newtestX, newtestY) # 测试分数
    print(f'Wrapper function for model train-test scores on filtered datasets are {scores}')



def boston_test():
    
    
    pass


if __name__ == '__main__':
    breast_cancer_test()