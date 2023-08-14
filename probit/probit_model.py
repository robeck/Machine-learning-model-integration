"""
@Author : haozhi chen
@Date : 2022.03
@Target : Implement the probit regression for multivariate variables

“Li, Y., Zhuang, X., Wang, J., & Zhang, W. (2020). Analysis of the impact of Sino-US trade friction on China’s stock market based on complex networks.
The North American Journal of Economics and Finance, 52, 101185. http://doi.org/https://doi.org/10.1016/j.najef.2020.101185”

我们考虑，构建一个probit模型，参考上面的文献。
上面的文献实现了对异常波动和相关参数的 binary （0-1） probit regression
"""

import pandas as pd
import numpy as np
import arch.data.binary #没有arch package
import statsmodels.api as sm #没有statsmode package

def load_test_data():

    binary_data = arch.data.binary.load()
    data = binary_data.dropna()

    print(data)
    print(data.describe)


    return data




'''
这里对模型进行构建实例
1）明确dependent variable（endog）
2）明确independent variable （exdog） 可以是多元

在实例中，出现了对Const变量的构建：
3）const 和exdog相同时间长度，维度更低而已
'''
def probitmodel(data):
    endog = data[['admit']] #这是一个binary变量，因此作为依赖变量
    exdog = data[['gre','gpa']] #独立变量，作为回归的X
    const = pd.Series(np.ones(data.shape[0]),index = data.index) #默认生成一个全是 1 的序列作为回归的截距
    const.name = 'Const'
    fitdata = pd.DataFrame([const,exdog.gre,exdog.gpa]).T  #组合后的数据需要进行一次转置

    # 这个方法和上面的方法是等价的
    data['Const'] = pd.Series(np.ones(data.shape[0])) #直接向数据集合中增加一个Const列
    data.drop('rank',axis=1,inplace=True) #删除不需要的列
    #单独构建X和Y
    data_X = data[['Const','gre','gpa']]
    data_Y = data[['admit']]  #这里出来的是Series

    print(endog)
    print(fitdata)


    model = sm.Probit(endog,fitdata)
    res = model.fit()
    print(res.summary())
    #
    parms = res.params
    print(parms)



    # return model
    return None




if __name__ == '__main__':

    data = load_test_data()
    probitmodel(data)