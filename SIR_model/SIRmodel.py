'''
@Author: haozhi chen
@Date : 2022.03
@Target :
1）实现对微分方程的计算
2）对传染模型SIR的构建核基础实验
3）尝试学习如何复刻或者在其他领域使用这一模型


'''

import scipy.integrate as spi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
研究测试求解微分方程的方法
'''
def odeint_function():

    '''
    1) 一阶微分方程
    '''
    def diff_entlowd(y,x):
        return np.array(x)

    x = np.linspace(0, 10, 100)
    # 参数1 ：diff_entlowd 为微分方程
    # 参数2 ： 0 为初始值
    # 参数3 ： x 为自变量
    y = spi.odeint(diff_entlowd,0,x)

    plt.plot(x,y)  #由于x和y的维度想多，如果是多维数据不能直接绘制
    plt.grid()
    plt.show()


    '''
    2)高阶微分方程，可以拆解成多个一阶微分方程计算
    '''
    g = 9.8
    l = 1
    def diff_enthighd(d_list,x):
        omega,theta = d_list
        return np.array([-g/l*theta,omega])

    x = np.linspace(0,20,1000)
    y_list = spi.odeint(diff_enthighd, [0,35/100*np.pi], x)

    plt.plot(x,y_list[:,0],color='blue',label='omega')
    plt.plot(x,y_list[:,1],color='red',label='theta')
    plt.legend()
    plt.show()

'''
SIR传染模型是通过在一定时间内，计算传染着的微分（变化率）对传染模型进行模拟的
影响器结果的最大因素：参数的设置
'''
def SIR_model():
    """:arg
    :beta 感染系数
    :gamma 治愈系数
    :input ()一个长度为3的list或者tupe，存储初始的参数
        （易感者，感染者，免疫者）

    """
    beta = 1.478 #被感染系数，t时刻单位时间内被感染的人数
    gamma = 0.142 #治愈系数，
    start_date = 0
    end_date = 70
    interval = 1
    input = (1-1e-5,1e-6,0.0)
    
    
    def diff_model(INT,t):
        V = INT
        Y = np.zeros((3))
        Y[0] = -beta * V[0] * V[1]  # -beta * S * I
        Y[1] = beta * V[0] * V[1] - gamma * V[1]
        Y[2] = gamma * V[1]
        
        return Y
        
    t = np.arange(start_date,end_date+interval,interval)  #0，70 间隔为1，总共70个时间戳
    res = spi.odeint(diff_model,input,t)

    plt.plot(t,res[:,0],label = 'susceptible')  #易感染者的变化
    plt.plot(t,res[:,1],label = 'infective') #感染者的变化
    plt.plot(t,res[:,2],label = 'recovered') #康复者的变化（免疫者）
    plt.grid()
    plt.legend()
    plt.show()


    return res



if __name__ == '__main__':

    # odeint_function()
    SIR_model()
