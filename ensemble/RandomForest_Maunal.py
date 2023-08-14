''':
@Author: haozhic
@Date: 2022-7-15
@Target: 实现对训练数据的 Random-Forest 模型的训练

进一步考虑：
1）参数的优化：GridsearchCV

'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  # 网格搜索的参数优化
from sklearn.model_selection import cross_val_score  # 学习曲线寻优
from sklearn.metrics import roc_auc_score  # 计算auc得分
from sklearn.metrics import RocCurveDisplay  # 即将替代上面的plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import PrecisionRecallDisplay  # 绘制precision-recall

import matplotlib.pyplot as plt
import shap
from tqdm import tqdm


sc = StandardScaler()  # 标准化实例

"测试部分的数据处理"


def data_preprocess():
    filename = '~/EnergyriskProject/Data/Energy_companies_trade_data.csv'
    df = pd.read_csv(filename)
    stocklist = list(set(df.Stkcd.tolist()))

    for stockid in stocklist:
        stockdata = df[df.Stkcd == stockid].sort_values('Trddt', axis=0)
        stockdata['date'] = pd.to_datetime(stockdata.Trddt.tolist(), format='%Y-%m-%d')
        stockdata = stockdata.set_index('date').drop(columns=['Stkcd', 'Trddt'])
        rolldata = stockdata[['Clsprc']]
        '''我们考虑一个这样的标准来设置标签
        1）使用过去一年的clsprc，计算平均值
        2）滚动比较，如果当前的数据大于过去一年的平均值，设置为 1，否则为 0
        '''
        # 滚动计算clsprc的均值
        rolldata = rolldata.rolling(window=250, min_periods=200).mean().dropna(axis=0).rename(
            columns={'Clsprc': 'meanClsprc'})
        # 合并这个计算的均值，和正常的数据
        mergdata = pd.merge(rolldata, stockdata, how='left', left_index=True, right_index=True).dropna(axis=1).drop(
            columns=['Markettype', 'Trdsta', 'Capchgdt'])  # 剔除一些空数据，空列
        # 实现 if clsprc > meanclsprc：1 else：0
        mergdata['flag'] = [1 if x > y else 0 for x, y in
                            zip(mergdata.Clsprc.tolist(), mergdata.meanClsprc.tolist())]  # 同构构建一个列表推导式输出flag标签
        if mergdata.empty:
            pass
        else:
            print(f'{stockid}')
            print(mergdata)
            RF_model(mergdata)

    return df


"针对RandomForest模型的输入数据进行处理"


def input_data_process(data, tags):
    X, y = data.iloc[:, 0:-1], data.iloc[:, -1]
    if tags == 'train_test':
        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.3)
        return train_X, test_X, train_y, test_y  # 返回训练和测试数据
    elif tags == 'prediction':
        return X, y  # 返回的就是预测数据
    else:
        raise MyError.Myexception('RandomForest数据处理遇到了问题')


"Random-Forest 使用GridSearchCV优化的模型"


def RF_model(data, finaldata):
    train_x, test_x, train_y, test_y = data
    Train_X, Train_y = finaldata
    # model = RandomForestClassifier(n_estimators=10,criterion='gini',max_depth=20,min_samples_leaf=25,random_state=0)

    ####################################################################################################
    "1.使用学习曲线来选择合适的n_estimators参数"
    scores = []
    for i in tqdm(range(0, 200, 10)):  # 循环0-200，间隔10。这些都是estimators的设置
        model = RandomForestClassifier(n_estimators=i + 1,
                                       n_jobs=-1,
                                       random_state=0)
        score = cross_val_score(model, train_x, train_y, cv=5).mean()
        scores.append(score)
    best_scores = max(scores)
    best_n_estimators = (scores.index(max(scores)) * 10) + 1
    print(f'训练数据最优得分：{best_scores}, 其参数为{best_n_estimators}')  # 输出最优的得分，并且给出这个对应的n_estimators
    # plt.plot(range(1,201,10),scores) # 参数和拟合得分绘制
    # plt.show()

    "2. 逐步寻优，优化其他参数：max_depth,min_samples_leaf,criterion..."
    "调整max depth"
    max_depth_params = {'max_depth': np.arange(1, 30, 1)}
    model1 = RandomForestClassifier(n_estimators=best_n_estimators,
                                    n_jobs=-1,
                                    random_state=90)
    GS = GridSearchCV(model1, max_depth_params, cv=5)
    GS.fit(train_x, train_y)
    best_max_scores = GS.best_score_
    best_max_param = GS.best_params_
    print(f'训练数据中max depthe调整下得分：{best_max_scores}, 其参数为：{best_max_param}')
    # 如果max_depth的结果合适，可以进行参数设置

    "调整min simple leaf"
    min_sample_leaf_params = {'min_samples_leaf': np.arange(1, 20, 1)}
    model2 = RandomForestClassifier(n_estimators=best_n_estimators,
                                    n_jobs=-1,
                                    max_depth=best_max_param.get('max_depth'),
                                    random_state=90)
    GS = GridSearchCV(model2, min_sample_leaf_params, cv=5)
    GS.fit(train_x, train_y)
    best_max_scores = GS.best_score_
    best_min_sample_leaf_params = GS.best_params_
    print(f'训练数据在min sample leaf 调整下得分：{best_max_scores}, 其参数为：{best_min_sample_leaf_params}')
    # 如果min_sample_leaf结果合适，进行参数调整。标准一般看scores得分,越高越高

    "调整criterion"
    criterion_params = {'criterion': ['gini', 'entropy']}
    model3 = RandomForestClassifier(n_estimators=best_n_estimators,
                                    n_jobs=-1,
                                    max_depth=best_max_param.get('max_depth'),
                                    min_samples_leaf=best_min_sample_leaf_params.get('min_samples_leaf'),
                                    random_state=90)
    GS = GridSearchCV(model3, criterion_params, cv=5)
    GS.fit(train_x, train_y)
    best_criterion_scores = GS.best_score_
    best_criterion_para = GS.best_params_
    print(f'训练数据中最优criterion得分，结合了全部数据：{best_criterion_scores},其参数为：{best_criterion_para}')
    # 如果criterion合适，进行参数调整

    "最终的final_model"
    final_model = RandomForestClassifier(n_estimators=best_n_estimators,
                                         n_jobs=-1,
                                         max_depth=best_max_param.get('max_depth'),
                                         min_samples_leaf=best_min_sample_leaf_params.get('min_samples_leaf'),
                                         criterion=best_criterion_para.get('criterion'),
                                         random_state=90)
    final_model.fit(train_x, train_y)
    best_final_scores = final_model.score(train_x, train_y)
    print(f'将所有的参数汇总到一个模型中，其最后的训练得分为：{best_final_scores}')

    "总结上面的训练结果，训练数据，选出最合适的模型"
    scores_lists = [best_max_scores, best_max_scores, best_criterion_scores, best_final_scores]  # 得分list
    model_lists = [model1, model2, model3, final_model]  # model的lists
    max_score_index = scores_lists.index(max(scores_lists))  # 提取上面模型得分list中最佳得分的位置索引
    Output_model = model_lists[max_score_index]  # 最优得分下，选出的参数模型

    ########################################################################################################
    "3.一次性gridsearchcv的方法，寻找最优的参数"
    parameters = {'n_estimators': np.arange(0, 200, 10), 'criterion': ['gini', 'entropy'],
                  'max_depth': np.arange(1, 30, 1), 'min_samples_leaf': np.arange(1, 20, 1)}
    model = RandomForestClassifier(n_jobs=-1,
                                   random_state=90)
    GS = GridSearchCV(model, parameters, cv=5)
    GS.fit(train_x, train_y)
    best_scores_total = GS.best_score_
    best_parameters = GS.best_params_
    print(
        f'一次性gridsearch搜索n_estimators, criterion, max depth, min samples leaf的最佳得分为：{best_scores_total},最佳参数为：{best_parameters}')

    "4.最优模型实例化，"
    if best_scores_total > best_final_scores:  # 一次性gridsearch结果好于逐步的！
        final_model = RandomForestClassifier(n_estimators=best_n_estimators,
                                             n_jobs=-1,
                                             max_depth=best_parameters.get('max_depth'),
                                             min_samples_leaf=best_parameters.get('min_samples_leaf'),
                                             criterion=best_parameters.get('criterion'),
                                             random_state=90)
        final_model.fit(Train_X, Train_y)
    else:
        final_model = Output_model
        final_model.fit(Train_X, Train_y)

    "模型测试得分"
    prediction = final_model.predict(test_x)
    pred_prob = final_model.predict_proba(test_x)[:, 1]
    "相关得分计算"
    scores = final_model.score(test_x, test_y)
    f1_scores = f1_score(test_y, prediction, average='binary')
    auc = roc_auc_score(test_y, pred_prob)
    print(f'模型在样本内测试集的得分为：{scores}')
    print(f'模型在样本内测试集 f1 scores得分为{f1_scores}')
    print(f'模型在样本内测试集的 AUC得分为：{auc}')

    return final_model


"返回训练好的模型，使用该模型对样本外进行预测"


def Model_prediction(model, date, predicted_data):
    pred_X, pred_y = predicted_data
    "1.预测"
    prediction = model.predict(pred_X)
    prediction_prob = model.predict_proba(pred_X)[:, 1]  # 预测值为1的那一列的概率！
    # print(f'预测数据集为：{prediction}')
    "各项得分"
    scores = model.score(pred_X, pred_y)
    f1_scores = f1_score(pred_y, prediction, average='binary')
    auc_scores = roc_auc_score(pred_y, prediction_prob)
    print(f'模型在样本外预测数据集的得分为：{scores}')
    print(f'模型在样本外预测数据集 f1 scores得分为：{f1_scores}')
    print(f'模型在样本外预测数据集 AUC得分为：{auc_scores}')

    # plt.plot(prediction_y ,color='red',label='test')
    # plt.plot(prediction,color='blue',label='prediction')
    # plt.show() # True 和 prediction 结果比较！

    "2. 绘制roc，正确率和错误率图"
    "传统方法"
    # plot_roc_curve(final_model,test_x,test_y)
    "2.1 sklearn的其他方法"
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(model, pred_X, pred_y, ax=ax, alpha=0.8)
    plt.show()

    "2.2 precison-recall"
    "绘制方案"
    display = PrecisionRecallDisplay.from_estimator(
        model, pred_X, pred_y, name="RandomForest"
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    plt.show()

    "2.3 机器学习可解释性分析"
    "shap_value: 创建这个解释器"
    explainer = shap.TreeExplainer(model)
    shape_value = explainer.shap_values(pred_X)
    ""
    shap.initjs()
    shap.force_plot(explainer.expected_value[1], shape_value[1], pred_X)

    "Advanceed shape value:"
    shap.summary_plot(shape_value[1], pred_X, title=f'SHAP value on {date}')

    return None


def main():
    data = data_preprocess()
    model = RF_model(data)
    return None


if __name__ == '__main__':
    main()