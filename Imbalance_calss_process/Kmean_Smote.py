''':
@ author: haozhi chen
@ date: 2023-03
@ target : 实现对非均衡数据集的处理，解决非均衡问题

'''

import pandas as pd
import numpy as np
from kmeans_smote import KMeansSMOTE  # 非均衡数据集解决方案
from imblearn.datasets import fetch_datasets  # 测试数据集


def test():
    datasets = fetch_datasets(filter_data=['oil'])
    X, y = datasets['oil']['data'], datasets['oil']['target']
    print(X)

    [print('Class {} has {} instances'.format(label, count))
     for label, count in zip(*np.unique(y, return_counts=True))]

    kmeans_smote = KMeansSMOTE(
        kmeans_args={
            'n_clusters': 100
        },
        smote_args={
            'k_neighbors': 10
        }
    )
    X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)

    [print('Class {} has {} instances after oversampling'.format(label, count))
     for label, count in zip(*np.unique(y_resampled, return_counts=True))]

    return None



if __name__ == '__main__':
    test()