import pandas as pd
from sklearn.model_selection import train_test_split

X = pd.read_csv('train.csv',index_col='Id')
X_test = pd.read_csv('test.csv',index_col='Id')

#remove row with missing data, separate data for predicetion
col_missing = [col for col in X.columns if X[col].isnull().any()]
X.dropna(subset=['SalePrice'],axis=0,inplace=True) # 剔除Saleprice是空的数据
y = X.SalePrice
X.drop('SalePrice',axis=1,inplace=True) #剔除整个SalePrce列，作为全部X features

# keep things simple, we will drop columns with missing values
X.drop(col_missing,axis=1,inplace=True)
X_test.drop(col_missing,axis=1,inplace=True)

#set train val datasets
X_train,X_valid,Y_train,Y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)


#定义scorefunction用来计算平均误差（mean absolute error）
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def score_datasets(X_train,X_valid,y_train,y_valid):
    model = RandomForestRegressor(n_estimators=100,random_state=1)
    model.fit(X_train,y_train)
    precdiction = model.predict(X_valid)
    return mean_absolute_error(precdiction,y_valid)

# 第一种处理分类变量的方法：删除即可
# 首先找到分类变量有哪些
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)

# 其次剔除这些列
drop_X_train = X_train.select_dtypes(exclude=['object'])
drom_X_valid = X_valid.select_dtypes(exclude=['object'])
print(score_datasets(drop_X_train,drom_X_valid,Y_train,Y_valid))

# # 第二种方法顺序编码（ordinal encoding）
# # 首先制作copy
label_X_trian = X_train.copy()
label_X_valid = X_valid.copy()
#
# #使用ordianlencode对分类变量进行编码
from sklearn.preprocessing import OrdinalEncoder
# encoder = OrdinalEncoder()
# label_X_trian[object_cols] = encoder.fit_transform(X_train[object_cols])
# label_X_valid[object_cols] = encoder.transform(X_valid[object_cols])
#
# print(score_datasets(label_X_trian,label_X_valid,Y_train,Y_valid)) # 啊哦，这里直接就报错了，为什么呢？
'''
为训练数据中的某一列拟合一个序数编码器，为训练数据中出现的每个独特的值创建一个相应的整数值标签。
如果验证数据中包含的数值没有出现在训练数据中，编码器就会出错，因为这些数值不会被分配到一个整数。
请注意，验证数据中的 "Condition2 "列包含 "RRAn "和 "RRNn "两个值，但这些值并没有出现在训练数据中--因此，
如果我们试图使用scikit-learn的序数编码器，代码会出现错误。

简而言之，训练集中需要分类的数据在验证集中是没有的，因此出现无法对应的问题
'''
# 解决方法
# 方法二：
# Categorical columns in the training data
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply ordinal encoder
my_ordinalencoder = OrdinalEncoder()
label_X_train[good_label_cols] = my_ordinalencoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = my_ordinalencoder.transform(X_valid[good_label_cols])

print(score_datasets(label_X_trian,label_X_valid,Y_train,Y_valid))