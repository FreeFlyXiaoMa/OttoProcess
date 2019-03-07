import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics

train=pd.read_csv('Otto_train.csv')
#print(train.head())
y_train=train['target']
X_train=train.drop(['target'],axis=1)

#用于每个样本的聚类结果
train_id=train['id']

#数据进行归一
X_train=normalize(X_train,norm='l2',copy=False)

#聚类分析方法
def K_cluster_analysis(K,X):
    print('K-means begin with cluster:{}'.format(K))

    #kmeans,在训练集上训练
    mb_means=MiniBatchKMeans(n_clusters=K)

    #mb_means.fit_transform(X)
    #mb_means.fit(X)
    y_pred=mb_means.fit_predict(X)

    #评估标准
    CH_score=metrics.calinski_harabaz_score(X,y_pred)
    #print(CH_score.shape())
    print('CH_score:{}'.format(CH_score))

    return CH_score

#设置超参数搜索范围
#Ks=[10,20,30,40,50]
Ks=[8,9,10,11,12]
#Ks=range(2,20)
CH_scores=[]
for K in Ks:
    ch=K_cluster_analysis(K,X_train)
    CH_scores.append(ch)

#绘制不同K对应的聚类性能，找到最佳的模型/参数
plt.plot(Ks,np.array(CH_scores),'b--',label='CH_score')
#plt.subplot(Ks,np.array(CH_scores),'b--',label='CH_score')
#plt.show()

#输出最佳超参数
index=np.unravel_index(np.argmax(CH_scores,axis=None),len(CH_scores))
Best_k=Ks[index[0]]
print('Best K is:',Best_k)

#用最佳的K再次聚类，得到聚类结果
mb_kmeans=MiniBatchKMeans(n_clusters=Best_k)
y_pred=mb_kmeans.fit_predict(X_train)

print(y_pred)

#保存聚类结果
feat_names_Kmeans='Kmeans_'+str(Best_k)
y=pd.Series(data=y_train,name='target')
train_kmeans=pd.concat([train_id,pd.Series(name=feat_names_Kmeans,data=y_pred),y],axis=1)
train_kmeans.to_csv('Otto_FE_KMeans.csv',index=False,header=True)

#保存模型
import pickle
pickle.dump(mb_kmeans,open('mb_kmeans.pkl','wb'))

'''
#测试数据
test=pd.read_csv('Otto_test.csv')
X_test=test.drop(['id'],axis=1)
print(X_test.shape)
test_id=test['id']
normalize(X_test,norm='l2',copy=False)

#加载模型
mb_kmeans=pickle.load(open('mb_kmeans.pkl','rb'))
y_test_pred=mb_kmeans.predict(X_test)

#保存测试结果
feat_names_Kmeans='Kmean_'+str(mb_kmeans.cluster_centers_)
test_kmeans=pd.concat([test_id,pd.Series(name=feat_names_Kmeans,data=y_test_pred)],axis=0)
test_kmeans.to_csv('Otto_test_KMeans.csv')'''

