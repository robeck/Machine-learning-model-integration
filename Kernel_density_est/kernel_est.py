import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

plt.rcParams['font.sans-serif'] = 'SimHei'
# 随机生成一个正太分布的序列，并绘制图
def data_gen():
    x = np.random.normal(0, 1, (500))  # 0,100. 100个正太分布的数据
    df = pd.DataFrame(x)
    df.hist()
    plt.show()

    return x

def kernel_gen(data):
    X = data.reshape(-1,1)  #输入的训练数据
    X_plot = np.linspace(0,0.1,1000)[:,np.newaxis] #使用 [:,np.newaxis] 转换成 2d array
    kde = KernelDensity(kernel='gaussian',bandwidth=0.75).fit(X)  # 高斯核密度估计
    log_dens = kde.score_samples(X_plot) # 返回的是点x_plot对应概率密度的log值，需要使用exp求指数还原

    params = kde.get_params()
    print(params)
    print(np.exp(log_dens))

    #绘制图像
    plt.figure(figsize=(10,8))
    plt.plot(X_plot,np.exp(log_dens),marker='.',linewidth=1,c='b',label='核密度值')
    plt.tick_params(labelsize=20)
    font = {'size':20}
    plt.xlabel('变量',font)
    plt.ylabel('概率密度函数',font)
    plt.legend(fontsize=15)
    plt.show()

    return 0

if __name__ == '__main__':
    data = data_gen()
    kernel_gen(data)

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    kde.score_samples(X)
