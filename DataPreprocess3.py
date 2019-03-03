import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn
train=pd.read_csv('Otto_train.csv')
# 标签
y_train = train['target']  # 形式为Class_x

# 暂存id，其实id没什么用
train_id = train['id']
# drop ids and targets
X_train = train.drop(['id', 'target'], axis=1)

# 保存特征名称
column_org = X_train.columns

X_train_log = np.log1p(X_train)

# 重新组成DataFrame
feat_names_log = column_org + "_log"
X_train_log = pd.DataFrame(columns=feat_names_log, data=X_train_log.values)

#print(X_train_log.head())
# transform counts to TFIDF features
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

# 输出稀疏矩阵
X_train_tfidf = tfidf.fit_transform(X_train).toarray()

# 重新组成DataFrame，为了可视化
feat_names_tfidf = column_org + '_tfidf'
X_train_tfidf = pd.DataFrame(columns=feat_names_tfidf, data=X_train_tfidf)
# 对原始数据缩放
from sklearn.preprocessing import MinMaxScaler

# 构造输入特征的标准化器
ms_org = MinMaxScaler()

# 保存特征名字，用于结果保存为csv
feat_names_org = X_train.columns

# 训练模型：fit
# 并对数据进行特征缩放：transform
X_train = ms_org.fit_transform(X_train)

# 对log数据缩放
X_train_log = ms_org.fit_transform(X_train_log)

# 对tf-idf数据缩放
X_train_tfidf = ms_org.fit_transform(X_train_tfidf)

# 保存原始特征
y = pd.Series(data=y_train, name='target')
feat_names = column_org
train_org = pd.concat([train_id, pd.DataFrame(columns=feat_names_org, data=X_train), y], axis=1)
train_org.to_csv('Otto_FE_train_org.csv', index=False, header=True)

# 保存log特征变换结果
y = pd.Series(data=y_train, name='target')
train_log = pd.concat([train_id, pd.DataFrame(columns=feat_names_log, data=X_train_log), y], axis=1)
train_log.to_csv('Otto_FE_train_log.csv', index=False, header=True)

# 保存tf-idf特征变换结果
y = pd.Series(data=y_train, name='target')
train_tfidf = pd.concat([train_id, pd.DataFrame(columns=feat_names_tfidf, data=X_train_tfidf), y], axis=1)
train_tfidf.to_csv('Otto_FE_tran_tfidf.csv', index=False, header=True)


