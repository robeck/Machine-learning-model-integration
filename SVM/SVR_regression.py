'''
@Author: haozhi chen
@Date:2022.04.06
@Target: 测试实现，支持向量机的回归

支持向量机SVM SVR是最为常用的，在实证科研中可以使用的机器学习的方法。因此我们简要的进行一下测试
'''
from sklearn.svm import LinearSVR,SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def linear_svr():
    X,y = make_regression(n_features=4,random_state=0)
    print(X)
    print(y)
    regr = make_pipeline(StandardScaler(),LinearSVR(random_state=0,tol=1e-05))
    regr.fit(X,y)

    #输出
    print(regr.named_steps['linearsvr'].coef_)
    print(regr.named_steps['linearsvr'].intercept_)
    print(regr.get_params(True))
    print(regr.predict([[0,0,0,0]]))

    # fig = plt.figure(figsize=(12,8),dpi=100)
    # ax = fig.add_subplot(111)
    # ax.scatter(X,y,color='c',label="scatter graph")
    # plt.show()


    return None

'''
简单测试：非线性的支持向量机回归

操作说明：如果需要显示详细的绘图，需要使用IPython，jupyter进行实现
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
# from IPython.display import display

def nolinear_svr():
    n_sample,n_features = 1000,5
    random_pick = np.random.RandomState(0)
    y = random_pick.randn(n_sample) # 随机生成100各Y 标签变量
    X = random_pick.randn(n_sample,n_features) # 生成100*5的100个5个特征的 X变量
    data = pd.DataFrame({'var1':X[:,0],'var2':X[:,1],'var3':X[:,2],'var4':X[:,3],'var5':X[:,4]})
    featurenames = data.columns.tolist()
    newx = data[featurenames]
    newy = pd.Series({'labels':y})
    print(newx)

    # 构架训练，验证数据集，构建pipeline模型,形成自己的模型
    train_X,val_x,train_y,val_y = train_test_split(newx,y,random_state=1,train_size=0.8)  #划分训练，测试
    regr = make_pipeline(StandardScaler(),SVR(C=1.0,epsilon=0.2)) #形成模型的管道架构
    model = regr.fit(train_X,train_y) #拟合模型

    print('root mean squared test error = {0}'.format(np.sqrt(np.mean((model.predict(val_x) - val_y) ** 2)))) #输出预测误差


    # Kaggle对模型的解释力进行说明
    "permutation" #排列顺序的重要性
    # 测试
    perm = PermutationImportance(model,random_state=1).fit(val_x,val_y)
    explanation = eli5.explain_weights(perm,feature_names = val_x.columns.tolist())
    text = eli5.format_as_text(explanation)  # explanation难以直接读取，eli5提供将explanation转换成human-readable的结构的方法，这里转换成text结构
    print(text)
    ###########################################

    "Partial plots" # 部分依赖图！
    from  matplotlib import pyplot as plt
    from pdpbox import pdp,get_dataset,info_plots
    pdp_goals = pdp.pdp_isolate(model=model,dataset=val_x,model_features=featurenames,feature='var2') #这里以var2为测试对象，我们可以使用不同变量进行替换

    # plot it
    pdp.pdp_plot(pdp_goals,'var2')
    plt.show()

    # 2D partial depedence plots
    feature_to_plot = ['var1','var2'] #检验var1 和 var2
    pdp_multi = pdp.pdp_interact(model=model,dataset=val_x,model_features=featurenames,features=feature_to_plot)
    pdp.pdp_interact_plot(pdp_interact_out=pdp_multi,feature_names=feature_to_plot,plot_type='contour')
    plt.show()
    #######################################

    "SHAP values" #an acronym from SHapley Additive exPlanations 详细的输出和图需要参见jupyter进行实现和展示
    # 测试single row的解释力,也就是一列数据对结果的预测，这里其实就是一个简单的单feature预测
    row_to_predic = 5
    data_for_prediction = val_x.iloc[row_to_predic]
    data_for_prediction_array = data_for_prediction.values.reshape(1,-1)
    res = model.predict(data_for_prediction_array) #这里没有predict_prob检验预测的准确概率
    print(res)

    # get shap values for single prediction
    import shap
    # 计算shap value
    explainer = shap.KernelExplainer(model.predict,train_X)
    k_shape_value = explainer.shap_values(data_for_prediction) # data_for_prdiction是单个列，变量
    print(k_shape_value)

    # shap.initjs()
    # shap.force_plot(explainer.expected_value[1], k_shape_value[1], data_for_prediction)
    # plt.show()
    #######################################

    "Advanced uses of SHAP values"
    ## summary plot
    # Create object that can calculate shap values
    explainer = shap.KernelExplainer(model.predict,train_X)
    # calculate shap values. This is what we will plot.
    # Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
    shap_values = explainer.shap_values(val_x)
    # Make plot. Index of [1] is explained in text below.
    shap.summary_plot(shap_values, val_x)

    ## dependence contribution plot
    # Create object that can calculate shap values
    explainer = shap.KernelExplainer(model.predict,train_X)
    # calculate shap values. This is what we will plot.
    shap_values = explainer.shap_values(X)
    # make plot.
    shap.dependence_plot('Ball Possession %', shap_values, X, interaction_index="var1")

    shap.force_plot(explainer.expected_value, shap_values, val_x)



    return None




if __name__ == '__main__':
    linear_svr()

    nolinear_svr()
    