import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.core.pylabtools import figsize
from sklearn.metrics import accuracy_score, confusion_matrix

train1=pd.read_csv('Otto_FE_train_org.csv')
train2=pd.read_csv('Otto_FE_train_tfidf.csv')
train2=train2.drop(['id','target'],axis=1)
train=pd.concat([train1,train2],axis=1,ignore_index=False)

#print(train.head())
y_train=train['target']
X_train=train.drop(['id','target'],axis=1)

#
feat_names=X_train.columns

#稀疏化
from scipy.sparse import csr_matrix
X_train=csr_matrix(X_train)
#print(X_train.info())

from sklearn.tree import DecisionTreeClassifier
DT1=DecisionTreeClassifier()

from sklearn.model_selection import cross_val_score
#loss=cross_val_score(DT1,X_train,y_train,cv=3,scoring='neg_log_loss')

#print("logloss of each fold is:",-loss)
#print("cv logloss is:",-loss.mean())
from sklearn.model_selection import GridSearchCV
max_depth=range(10,100,3)
min_samples_leaf=range(1,10,4)
tuned_parameters=dict(max_depth=max_depth,min_samples_leaf=min_samples_leaf)

DT2=DecisionTreeClassifier()
if __name__ == '__main__':
    grid=GridSearchCV(DT2,tuned_parameters,cv=2,scoring='neg_log_loss',n_jobs=4,refit=False)
    grid.fit(X_train,y_train)
    print("Best score:%f using %s"%(-grid.best_score_,grid.best_params_))

    test_means=-grid.cv_results_['mean_test_score']
    test_scores=np.array(test_means).reshape(len(max_depth),len(min_samples_leaf))

    #for i,value in enumerate(max_depth):
        #plt.plot(min_samples_leaf,test_scores[i],label='test_max_score'+str(value))
   # plt.legend()
   # plt.xlabel('min_samples_leaf')
   # plt.ylabel('logloss')
    #plt.show()
    plt.plot(min_samples_leaf,test_scores[3],label='test_max_depth:'+str(9))
    plt.show()