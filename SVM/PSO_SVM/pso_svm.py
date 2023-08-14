'''
@Author: haozhi chen
@Date: 2022-08
@Target: 实现以下基于粒子群优化的SVM模型

特点和注意事项！
（1）目前测试的是对参数 gamma，C 的优化。如需要调整则对 particle_position_vector 进行变化即可
（2）
'''
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import LinearSVC,SVC,NuSVC

from SourceCode.ML_dataprocess.Dataprocess_Class import SVC_Dataprocess
from SourceCode.ML.PSO_SVM.Config import args,kernel
from SourceCode.ML.PSO_SVM.utile import confusion_martrix_disp,plot


'''SVM模型拟合过程(适应度函数）
1）拟合
2）绘制混淆矩阵
3）输出混淆矩阵结果
'''
def fittess_function(params,data):
    '''
    :param params:
    :param data:

    :return confusion matrix:
        [0] 训练集 的预测错误结果统计
        [1] 测试集 的预测错误结果统计
    '''
    train_x,train_y,test_x,test_y = data # 返回的是多个数据，可以直接这样赋值
    classifer = make_pipeline(StandardScaler(),
                              SVC(kernel=kernel, gamma=params[0], C=params[1], max_iter=2000, random_state=66, probability=True))
    classifer.fit(train_x,train_y)
    y_train_pred = classifer.predict(train_x)
    y_test_pred = classifer.predict(test_x)

    "绘制以下测试数据集的 混淆矩阵"
    # label_names = ['label1','label2']
    # titles = [("confusion matrix without norm",None),
    #           ("confusion matrix with noem","true")]
    # confusion_martrix_disp(classifer,test_x,test_y,label_names,titles)

    return confusion_matrix(train_y,y_train_pred)[0][1]+confusion_matrix(train_y, y_train_pred)[1][0], \
           confusion_matrix(test_y, y_test_pred)[0][1] + confusion_matrix(test_y, y_test_pred)[1][0]


def pso_svm_model(data):
    # 初始化参数
    # 参数1：代表粒子位置，其实也是适应函数输入的参数（gamma，c）
    particle_position_vector = np.array([np.array([random.random() * 10, random.random() * 10]) for _ in range(args.n_particles)]) # 初始化每一个粒子的位置
    # 参数2：粒子自身历史最优位置
    pbest_position = particle_position_vector
    # 参数3：粒子自身最优的适应函数值 初始化为 inf
    pbest_fitness_value = np.array([float('inf') for _ in range(args.n_particles)])
    # 参数4，5：全局位置初始，全局适应函数初始化
    gbest_fitness_value = np.array([float('inf'),float('inf')])
    gbest_position = np.array([float('inf'),float('inf')])
    # 参数6：速度向量初始化
    velocity_vector = ([np.array([0,0]) for _ in range(args.n_particles)])
    iteration = 0 #初始迭代标记

    '进行不断的迭代'
    while iteration < args.n_iterations:
        plot(particle_position_vector) # 绘制初始化的粒子分布散点图
        '遍历100个粒子'
        for i in range(args.n_particles):
            fitness_res = fittess_function(particle_position_vector[i],data) #统计预测结果
            print("error of priticle ",i,'is (training,test)',fitness_res,"At (gamma,c): ",
                  particle_position_vector[i])

            """
            初始化的 自身历史最优 进行迭代替换
            （1）比较
            （2）用较好结果 替换 自身历史最优：这是一个自己比较的过程
            （3）粒子位置信息（参数）替换
            """
            if (pbest_fitness_value[i] > fitness_res[1]): # 因为初始的局部结果是无穷的，模型拟合结果显示错误数量会比其更小，因此用当前粒子逐步迭代替换
                pbest_fitness_value[i] = fitness_res[1] # 比较好的结果（错误数量）赋值给局部最优
                pbest_position[i] = particle_position_vector[i] # 这个局部最优的位置信息（gamma，c 参数）就是那个粒子的参数

            """
            粒子的 自身历史最优 是否替换 全局最优
            """
            if (gbest_fitness_value[1] > fitness_res[1]): # 全局的结果
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]
            elif (gbest_fitness_value[1] == fitness_res[1] and gbest_fitness_value[0] > fitness_res[0]):
                gbest_fitness_value = fitness_res
                gbest_position = particle_position_vector[i]

        '遍历每一个粒子，更新速度，位置参数'
        for i in range(args.n_particles):
            new_velocity = (args.W * velocity_vector[i]) + (args.c1 * random.random()) * (
                pbest_position[i] - particle_position_vector[i]) + (args.c2 * random.random()) *(
                    gbest_position - particle_position_vector[i])
            new_position = new_velocity + particle_position_vector[i]
            particle_position_vector[i] = new_position

        iteration = iteration + 1



def main():
    df = pd.read_csv('/home/haozhic1/FinancialRisk_detection/Data/stockdaydata.csv')
    datas = SVC_Dataprocess(df).maindataprocess()
    print(datas)
    pso_svm_model(datas)



if __name__ == '__main__':
    main()