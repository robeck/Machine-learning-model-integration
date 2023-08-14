import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str  #str格式转换成dates

import matplotlib.pyplot as plt
import seaborn as sns

'''
这里的方法来自于：statsmodels 库
主要是对数据进行一整套完整的处理，从数据计算，时间调整方面
1）数据截取，计算（计算对数收益率）
2）数据时间的调整，将原始时间增加季度信息，并且使用date_from_str方法将时间调整为想要的格式

'''
def data_process_model():
    # 导入数据
    mdata = sm.datasets.macrodata.load_pandas().data
    #日期数据添加 和 转换
    dates = mdata[['year','quarter']].astype(int).astype(str)
    quarterly = dates['year'] + 'Q' + dates['quarter']
    quarterly = dates_from_str(quarterly) #季度数据转换成datetime格式
    # 输出工作数据集 + 日期
    mdata = mdata[['realgdp','realcons','realinv']]
    mdata.index = quarterly

    data = np.log(mdata).diff().dropna() #计算对数收益,得到的是带有季度时间index的各项指标的收益率
    print(data)

    return data


'''
工作流程
1) ADF 和 协整检验
2）定阶，建模，系数平稳检验
3）预测

4）脉冲效应，方差分解
# 因为进行方差分解和脉冲响应分析的时候，要求模型的残差为白噪声。但是！现实中，我们很难把所有影响Y的因素都囊括进方程，这就导致，现实中VAR模型的残差一般都不是白噪声。因此使用乔里斯基正交化来处理模型的残差。
'''
def VAR_csdn(data):


    # 1）ADF Conit
    listadfres = []
    for i in range(len(data.columns)):
        coldata = data.iloc[:,i]
        adfres = sm.tsa.stattools.adfuller(coldata,maxlag=5)
        listadfres.append(adfres)

    output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                         "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                          columns=['value_gdp','value_con','value_inv'])

    for j in range(len(listadfres)):
        adfres_ = listadfres[j]
        colname = output.columns.tolist()
        output[colname[j]]['Test Statistic Value'] = adfres_[0]
        output[colname[j]]['p-value'] = adfres_[1]
        output[colname[j]]['Lags Used'] = adfres_[2]
        output[colname[j]]['Number of Observations Used'] = adfres_[3]
        output[colname[j]]['Critical Value(1%)'] = adfres_[4]['1%']
        output[colname[j]]['Critical Value(5%)'] = adfres_[4]['5%']
        output[colname[j]]['Critical Value(10%)'] = adfres_[4]['10%']
    print(output)

    # coint协整检验, 逐列构建序列对进行检查
    for i in range(len(data.columns)):
        for j in range(i+1,len(data.columns)):
            data1,data2 = data.iloc[:,i],data.iloc[:,j]
            coinres = sm.tsa.stattools.coint(data1,data2)
            print(coinres)


    # 2.定阶 + 建模（提供了2个方法）
    # VAR模型
    model = VAR(data)
    res = model.fit(maxlags=10,method='ols',ic='aic',trend='nc')
    print(res.k_ar) # 输出模型的阶数
    print(res.summary()) #确定最大阶数，输出结果

    # 方法VARMAX，我们使用VARMAX构建模型，并且迭代训练，找到残差最小的模型为最好的阶数的模型
    model2 = sm.tsa.VARMAX(data,order=(10,0)) # 在这里就设定了阶数范围
    res2 = model2.fit(maxiter=100,disp=False)  # 通过迭代训练，拟合出最合适的阶数
    print(res2.summary())
    # 残差最后调整模型
    resid = res2.resid
    results= {'fitmod':res2,'resid':resid}
    print(resid.values[0])
    # 残差的系数平稳性检验cumsum检验
    for resi in resid.values: #这里循环的目的是逐个变量的系数进行检验（无法一次性对dataframe进行检验）
        coeftest = statsmodels.stats.diagnostic.breaks_cusumolsresid(resi)
        print(coeftest)


    # 3.预测
    res.forecast(data.values[-10:],5)
    res.plot_forecast(20)  # 绘制的向前预测步长
    plt.show()

    # res2.forecast(data.values[-10:])  #向前5步预测
    # print(res2)

    # 4.脉冲响应，方差分解
    irf = res.irf(10)  # 脉冲响应，10为周期period
    irf.plot(orth=False)  # 正价 = True ： 使用的是乔里斯基正交！
    irf.plot(impulse='realgdp')
    irf.plot_cum_effects(orth=False)


    ir = res2.impulse_responses(10,orthogonalized=True)
    ir.plot()
    #
    # #################################################################
    # # FEVD
    fevd = res.fevd()
    print(fevd.summary())
    print(fevd.decomp)
    fevd.plot()
    plt.show()


    return None




def VAR_statsmodels(data):

    # 1.模型建立
    varmodel = VAR(data)
    res = varmodel.fit(3) # fit（n-lag）滞后的阶数
    print(res.summary())

    # 2.绘制输出
    '''
    这里都是对时序的数据进行可视化展示，区别在于
    data.plot() 会旨在一个图上
    res.plot（）分别对数据在不同图上可视化
    '''
    # data.plot() # 时序数据可视化
    # res.plot() # 时序数据可视化
    # res.plot_acorr() #绘制时序的自相关函数 time series autocorrelation function
    # plt.show()

    # 3.预测
    lag_order = res.k_ar  #返回设定的滞后阶数
    '''
    Y: ndarray
    steps：int
    '''
    # res.forecast(data.values[-lag_order:],5)  #向前5步预测
    # res.plot_forecast(20)  # 绘制的向前预测步长
    # plt.show()

    # 4.脉冲响应在计量经济学研究中很有意义：它们是对其中一个变量的单位脉冲的估计响应
    irf = res.irf(10) #脉冲响应，10为周期period
    irf.plot(orth=False) # 正价 = True ： 使用的是乔里斯基正交！
    irf.plot(impulse='realgdp')
    irf.plot_cum_effects(orth=False)
    plt.show()


    # 5.方差分解 fevd （forecast error variance decomposition）预测误差方差分解
    fevd = res.fevd(5)  #向前5步
    fevd.summary()
    fevd.plot()
    print(fevd.decomp)  #输出的就是方差分解的向前N步，每一列数据对其他数据的脉冲影响

    return None


'''测试Google上的Generalized forecast error variance decomposition
使用DY的数据集！
1) 根据结果显示，我们这里的测试是对的，已经实现广义方差分解的输出
'''
import statsmodels.tsa.api as ts
from collections import OrderedDict

def Generalized_forecast_error_variance_decomposition():

    lags = 2
    nsteps = 10
    data = pd.read_csv('DY_data.csv',header=1).drop(columns=['no'])
    data = data.set_index(pd.to_datetime(data.date))
    data = data.drop('date',1)
    print(data)
    model = ts.VAR(data)
    names = model.endog_names # 其实就是column的名称
    nvar = len(names) # 变量个数
    results = model.fit(lags) #拟合之后2阶

    sigma_u = np.asarray(results.sigma_u)
    sd_u = np.sqrt(np.diag(sigma_u))
    fevd = results.fevd(nsteps,sigma_u/sd_u) #向前 n 步方差分解

    "decomposition of variance for series at nsteps"
    fe = fevd.decomp[:,-1,:] #分解的结果是一个三维的数据，19个国家*19个国家，并且这里还是向前nsteps，因此维度：19*10*19
    fevd_normalized = (fe / fe.sum(1)[:,None] * 100)
    print(fe)

    count_inc = fevd_normalized.sum(0) #求和的方向通过0,1控制
    count_to = fevd_normalized.sum(0) - np.diag(fevd_normalized)
    count_from = fevd_normalized.sum(1) - np.diag(fevd_normalized)

    resdf = pd.DataFrame(np.round(fevd_normalized,1),columns=names)
    resdf.index = names
    print(resdf)


    return None



if __name__ == '__main__':
    # data = data_process_model()
    # VAR_csdn(data)
    # VAR_statsmodels(data)
    Generalized_forecast_error_variance_decomposition()



