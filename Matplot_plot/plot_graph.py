import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts

from mpl_toolkits import mplot3d

ts.set_token('9be5506e3cb2894880eecd0a8be11db9f755ad95710951bbf22e2f25')
data = ts.get_k_data('000001','2018-01-01','2019-01-01','W')


def line_plot(data):
    print(data.head())


    plt.xlim((-1,len(data['date'])))
    plt.xticks(np.linspace(-1,len(data['date']),4),['1','2','3','4'])

    plt.xlabel('X label',fontsize=12)
    plt.ylabel('Y label',fontsize=12)

    plt.plot(data['date'],data['close'])
    plt.grid(True)
    plt.show()


    return None

def threed_plot():

    # ax1 = plt.subplot(111)
    # ax1 = plt.axes(projection='3d')
    # # ax2 = plt.axes(projection='3d')
    # # 简单测试三维绘制
    # def f(x,y):
    #     return np.sin(np.sqrt(x**2+y**2))
    #
    # r = np.linspace(0,10,20)
    # theta = np.linspace(0,10,40)
    # r,theta = np.meshgrid(r,theta)
    #
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)
    # z = f(x,y)
    #
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    #
    # ax1.plot_surface(x,y,z)
    # plt.show()


    # 绘制三维图像
    data['new_date'] = np.arange(0,51,1)
    print(data)

    ax2 = plt.axes(projection='3d')
    x,y = np.meshgrid(data['high'],data['close'])
    z,m = np.meshgrid(data['new_date'],data['new_date'])

    ''':args
        rstride : 行之间的跨度
        lstride ：列之间的跨度
        camp ： 颜色
    '''
    ax2.set_zlim((-1,len(data['new_date'])))
    ax2.set_zticks(np.linspace(-1,len(data['new_date']),4))
    ax2.set_zticklabels(['A','B','C','D'])
    ax2.plot_surface(x,y,z,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
    ax2.set_title('surface')
    plt.show()


    return None

def three_d_plot(data,xname,yname,zname,n,timeticklabel):

    ax = plt.axes(projection='3d')
    x,y,z = data[xname],data[yname],data[zname]
    # 我们需要假设一个输入的轴为时间轴，这里假设X轴为时间序列
    Xlist = np.arange(0,len(x),1)
    X,X = np.meshgrid(Xlist,Xlist)
    Y,Z = np.meshgrid(y,z)

    print(X.shape,Y.shape,Z.shape)

    ax.set_xlim((0,len(x)))
    ax.set_xticks(np.linspace(0,len(x),n))
    ax.set_xticklabels(timeticklabel)

    # 这里设置的坐标轴只是为了测试使用
    ax.set_xlabel('Time period')
    ax.set_ylabel('Y test data')
    ax.set_zlabel('Z test data')
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='viridis')

    plt.show()


if __name__ == '__main__':

    # line_plot(data)
    # threed_plot()

    ts.set_token('9be5506e3cb2894880eecd0a8be11db9f755ad95710951bbf22e2f25')
    data = ts.get_k_data('000001', '2018-01-01', '2019-01-01')  # 'code'列是代码

    for i in range(2):
        data = pd.concat([data, data], axis=0)

    data.sort_values('date', inplace=True)  # 时间排序

    three_d_plot(data, 'date', 'open', 'high', 3, ['201801', '201806', '201812'])





