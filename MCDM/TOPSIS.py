import pandas as pd
import numpy as np


# data input
# 这里需要输入的原始数据集X，各指标权重W


# Processing
# step 1 对原始输入数据进行同向化处理，并进行极大化处理
# 极小型指标 -> 几大型指标
def direction1(datas, flags):
    # 对于极小型的指标，比如死亡率，越小越好处理方法应该
    def normalization(data):
        return 1 / (data + 0)

    def normalization2(data):
        return np.max(data) - data

    if flags == 0:
        return list(normalization2(datas))
    else:
        return list(map(normalization, datas))


# 中间型指标 ->  极大型指标
def direction2(datas):
    def dmax(datas):
        return np.max(datas)

    def dmin(datas):
        return np.min(datas)

    def normalization(data):
        xmax = dmax(datas)
        xmin = dmin(datas)

        if data <= xmin or data >= xmax:
            return 0
        elif data > xmin and data < (xmin + xmax) / 2:
            return 2 * (data - xmin) / (xmax - xmin)
        elif data >= (xmin + xmax) / 2 and data < xmax:
            return 2 * (xmax - data) / (xmax - xmin)

    return list(map(normalization, datas))


# 区间型指标 ->  极大型指标
''':arg
amin,bmax对应的是最佳稳定区间
a_min,b_max对应的是最大容忍区间
'''


def direction3(datas, amin, bmax, a_min, b_max):
    def normalization(data):
        if data < amin:
            return 1 - (amin - data) / (amin - a_min)
        elif data >= amin and data <= bmax:
            return 1
        elif data > bmax:
            return 1 - (data - bmax) / (b_max - bmax)

    return list(map(normalization, datas))


# step2 主程序部分
# 包括了（归一化，最优劣方案确定，计算各评价对象和优劣方案的接近程度，计算各评价对象和最有方案的贴近程度C，C的结果从小到大就是最后的结论）
# 输入方式是逐渐将每一组数据导入，而非一次型全部输入。因此所做的操作更加简单
''':arg
data: (dataframe结构才行）多维是可以的，一起进行运算
weigth：权系数，注意选择合适的方法确定
'''


def topsis(data, weight):
    print(data)
    # 数据归一化处理
    data = data / np.sqrt(np.sum((data ** 2)))
    print(data)

    # 最有最劣方案确定
    Z = pd.DataFrame([data.max(), data.min()], index=['正理想解', '负理想解'])  # 这是单独一个df用来存储最优，最差解。这里的正负理想解是Index！

    # 计算数据和最优，最劣方案的接近程度
    weight = weight
    result = data.copy()
    result['正理想解距离'] = np.sqrt((weight * ((data - Z.loc['正理想解']) ** 2)).sum(axis=1))  # 这里计算数据到理想解之间的距离
    result['负理想解距离'] = np.sqrt((weight * ((data - Z.loc['负理想解']) ** 2)).sum(axis=1))

    # 计算到最优方案的程度分C
    result['综合得分指数'] = result['负理想解距离'] / (result['负理想解距离'] + result['正理想解距离'])
    result['排序'] = result.rank(ascending=True)['综合得分指数']

    return result


# 对于权重的处理,基于熵值法
def entropyWeight(data):
    data = np.array(data)
    # 归一化
    P = data / data.sum(axis=0)
    # 计算熵值
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)

    # 计算权系数
    return (1 - E) / (1 - E).sum()


# 对于权值的处理，层次分析法
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}


def ahp(data):
    data = np.array(data)
    m = len(data)

    # 计算特征向量
    weight = (data / data.sum(axis=0)).sum(axis=1) / m

    # 计算特征值
    Lambda = sum((weight * data).sum(axis=1) / (m * weight))

    # 判断一致性
    CI = (Lambda - m) / (m - 1)
    CR = CI / RI[m]

    if CR < 0.1:
        print(f'最大特征值：lambda = {Lambda}')
        print(f'特征向量：weight = {weight}')
        print(f'\nCI = {round(CI, 2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR, 2)} < 0.1，通过一致性检验')
        return weight
    else:
        print(f'\nCI = {round(CI, 2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR, 2)} >= 0.1，不满足一致性')


if __name__ == '__main__':
    # 综合测试，学校师生科研等数据分析
    data = pd.DataFrame(
        {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
         '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])
    entroweight = entropyWeight(data)
    # print(entroweight)
    data['生师比'] = direction3(data['生师比'], 5, 6, 2, 12)  # 师生比数据为区间型指标
    data['逾期毕业率'] = 1 / data['逾期毕业率']  # 逾期毕业率为极小型指标

    out = topsis(data, weight=[0.2, 0.3, 0.4, 0.1])  # 设置权系数
    print(out)

    outweight = topsis(data, weight=entroweight)
    print(outweight)