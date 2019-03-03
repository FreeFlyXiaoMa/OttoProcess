import  pandas as  pd
import numpy as np
import lightgbm as lgbm
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

train1=pd.read_csv('Otto_FE_train_org.csv')
train2=pd.read_csv('Otto_FE_train_tfidf.csv')

trian2=train2.drop(['id','target'],axis=1)
train=pd.concat([train1,train2],axis=1,ignore_index=False)
#print(train.head())
del train1
del train2
y_train=train['target']
y_train=y_train.apply(lambda s:s[6:])
y_train=y_train.apply(lambda s:int(s[6:])-1)#将Otto类别型的特征class_X变成 0-8之间的整数
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
n_estimators_1=get_n_estimators(params,X_train,y_train)
print(n_estimators_1)