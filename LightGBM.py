import  pandas as  pd
import numpy as np
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train1=pd.read_csv('Otto_FE_train_org.csv')
train2=pd.read_csv('Otto_FE_tran_tfidf.csv')

trian3=train2.drop(['id','target'],axis=1)
#print(trian3.head())
train=pd.concat([train1,trian3],axis=1,ignore_index=False)
#print(train.head())
del train1
del train2
y_train=train['target']
y_train=y_train.apply(lambda s:s[6:])
y_train=y_train.apply(lambda s:int(s)-1)#将Otto类别型的特征class_X变成 0-8之间的整数
#print('长度：',len(y_train))
#print(y_train.head())

#for i in range(len(y_train)):
    #print(y_train[i])
   # str=y_train[i]
  #  y_train[i]=str[6:]
    #print(y_train[i])
    #y_train[i]=y_train[i][6:]
#print(y_train.head())
X_train=train.drop(['id','target'],axis=1)

feat_names=X_train.columns

from scipy.sparse import csr_matrix
X_train=csr_matrix(X_train)

MAX_ROUNDS=10000
from sklearn.model_selection import StratifiedKFold
kfold=StratifiedKFold(n_splits=3,shuffle=True,random_state=3)

def get_n_estimators(params,X_train,y_train,early_stopping_rounds=100):
    lgbm_params=params.copy()
    lgbm_params['num_class']=9
    lgbmtrain=lgbm.Dataset(X_train,y_train)
    cv_results=lgbm.cv(lgbm_params,lgbmtrain,num_boost_round=MAX_ROUNDS,
                       nfold=3,metrics='multi_logloss',
                        early_stopping_rounds=early_stopping_rounds
                       ,seed=3)
    print('best n_estimators:',len(cv_results['multi_logloss-mean']))
    print('best cv score:',cv_results['multi_logloss-mean'][-1])
    return len(cv_results['multi_logloss-mean'])
'''
#调节n_estimators
params={'boosting_type':'gbdt',
        'objective':'multiclass',
        'num_class':9,
        'n_jobs':4,
        'learning_rate':0.1,
        'num_leaves':60,
        'max_depth':6,
        'max_bin':127,
        'subsample':0.7,
        'bagging_freq':1,
        'colsample_bytree':0.7
}
n_estimators_1=get_n_estimators(params,X_train,y_train)'''
n_estimators_1=420
#print(n_estimators_1)
params={
'boosting_type':'gbdt',
        'objective':'multiclass',
        'num_class':9,
        'n_jobs':4,
        'learning_rate':0.1,
        'n_estimators':n_estimators_1,
        'max_depth':7,
        'max_bin':127,
        'subsample':0.7,
        'bagging_freq':1,
        'colsample_bytree':0.7
}
'''
#调节num_leaves
if __name__=='__main__':
    lg=LGBMClassifier(silent=False,**params)
    num_leaves_s=range(50,90,10)
    tuned_parameters=dict(num_leaves=num_leaves_s)

    grid_search=GridSearchCV(lg,n_jobs=4,param_grid=tuned_parameters,cv=kfold,
                         scoring='neg_log_loss',refit=False,verbose=5)
    grid_search.fit(X_train,y_train)
    print('best score:',-grid_search.best_score_)
    print('best params:',grid_search.best_params_)
    test_means=grid_search.cv_results_['mean_test_score']
    n_leafs=len(num_leaves_s)
    x_axis=num_leaves_s
    plt.plot(x_axis,-test_means)
    plt.xlabel('num_leaves')
    plt.ylabel('Log Loss')
    plt.show()'''

#调节min_child_samples
min_child_sample_s=range(10,50,10)
params={
'boosting_type':'gbdt',
        'objective':'multiclass',
        'num_class':9,
        'n_jobs':4,
        'learning_rate':0.1,
        'n_estimators':n_estimators_1,
        'max_depth':7,
        'num_leaves':60,
        'max_bin':127,
        'subsample':0.7,
        'bagging_freq':1,
        'colsample_bytree':0.7
}
if __name__=='__main__':
    lg=LGBMClassifier(silent=False,**params)
    tuned_parameters=dict(min_child_samples=min_child_sample_s)
    grid_search=GridSearchCV(lg,param_grid=tuned_parameters,n_jobs=4
                             ,scoring='neg_log_loss',refit=False,verbose=5)
    grid_search.fit(X_train,y_train)
    #print('best estimator:',grid_search.best_estimator_)
    print('best score:',grid_search.best_score_)
    print('best params:',grid_search.best_params_)

    test_means=grid_search.cv_results_['mean_test_score']
    x_axis=min_child_sample_s
    plt.plot(x_axis,-test_means)
    plt.xlabel('min_child_sample')
    plt.ylabel('test_mean')
    plt.show()
