
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.core.pylabtools import figsize

train=pd.read_csv('Otto_train.csv')
#print(train.head())
#print(train.describe())
#sn.countplot(train.feat_1)
#plt.xlabel('feat_1')
#plt.ylabel('Number of occurrences')
#plt.show()

'''cols=train.columns
feat_corr=train.corr().abs()
plt.subplot(figsize(13,9))
sn.heatmap(feat_corr,annot=True)
sn.heatmap(feat_corr,mask=feat_corr<1,cbar=False)
plt.show()'''

y_train=train['target']
train_id=train['id']
X_train=train.drop(['id','target'],axis=1)
#print(X_train.head())
column_org=X_train.columns
#print(column_org)

#取log运算，更接近人对数字的敏感度,同时降低长维分布中大数值的影响，减弱长维分布的长尾性
X_train_log=np.log1p(X_train)
feat_names_log=column_org+"_log"
X_train_log=pd.DataFrame(columns=feat_names_log,data=X_train_log.values)
#print(X_train_log.head())

#feat编码：TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf=TfidfTransformer()
feat_names_tfidf=column_org+"_tfidf"
#输出稀疏矩阵
X_train_tfidf=tfidf.fit_transform(X_train).toarray()
#重新组成DataFrame,为了可视化
X_train_tfidf=pd.DataFrame(columns=feat_names_tfidf,data=X_train_tfidf)
#print(X_train_tfidf.head())

#归一化处理
from sklearn.preprocessing import MinMaxScaler
ms_org=MinMaxScaler()
feat_names_org=X_train.columns
X_train=ms_org.fit_transform(X_train)
#对log数据缩放
X_train_log=ms_org.fit_transform(X_train_log)

#对tf-idf数据缩放
X_train_tfidf=ms_org.fit_transform(X_train_tfidf)

#保存原始特征
y=pd.Series(data=y_train,name='target')
feat_names=column_org
train_org=pd.concat([train_id,pd.DataFrame(columns=feat_names_org,data=X_train),y],axis=1)
train_org.to_csv('Otto_FE_train_org.csv',index=False,header=True)

#保存log特征变换
y=pd.Series(data=y_train,name='target')
train_log=pd.concat([train_id,pd.DataFrame(columns=feat_names_log,data=X_train_log),y],axis=1
                    )
train_log.to_csv('Otto_FE_trian_log.csv',index=False,header=True)

#保存tf-idf
y=pd.Series(data=y_train,name='target')
train_tfidf=pd.concat([train_id,pd.DataFrame(columns=feat_names_tfidf,data=X_train_tfidf),y],axis=1)
train_tfidf.to_csv("Otto_FE_train_tfidf.csv",index=False,header=True)

#保存特征编码过程中用到的模型，用于后续对测试数据的特征编码
import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(ms_org,open('MinMaxScaler_org.pkl','wb'))
pickle.dump(ms_org,open('MinMaxScaler_log.pkl','wb'))
pickle.dump(ms_org,open('MinMaxScaler_tfidf.pkl','wb'))


