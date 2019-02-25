import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from IPython.core.pylabtools import figsize

train=pd.read_csv('Otto_FE_train_org.csv')
#print(train.head(2000))
y_trian=train['target']
X_train=train.drop(['id','target'],axis=1)
#print(X_train.head())

feat_names=X_train.columns

#数据稀疏化
from scipy.sparse import csr_matrix
X_train=csr_matrix(X_train)

#默认参数的Logistic回归
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')

#交叉验证
from sklearn.model_selection import cross_val_score
#loss=cross_val_score(lr,X_train,y_trian,cv=3,scoring='neg_log_loss')

#print('cv accuracy score is:',-loss)
#print('cv logloss is:',-loss.mean())

#使用网格搜索
from sklearn.model_selection import GridSearchCV
if __name__=='__main__':
    penalty=['l1','l2']
    Cs=[0.001,0.01,0.1,1,10,100,1000]
    tuned_parameter={'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}
    tuned_parameters=dict(penalty=penalty,C=Cs)
    grid=GridSearchCV(lr,param_grid=tuned_parameter,cv=3,scoring='neg_log_loss',n_jobs=4)
    grid.fit(X_train,y_trian)

    #print('best_score:',-grid.best_score_)
    #print('best_params:',grid.best_params_)
    #cv误差曲线
    test_means=grid.cv_results_['mean_test_score']
    test_stds=grid.cv_results_['std_test_score']
    train_means=grid.cv_results_['mean_train_score']
    train_stds=grid.cv_results_['std_train_score']

    n_Cs=len(Cs)
    number_penaltys=len(penalty)
    test_scores=np.array(test_means).reshape(n_Cs,number_penaltys)
    train_scores=np.array(train_means).reshape(n_Cs,number_penaltys)
    test_stds=np.array(test_stds).reshape(n_Cs,number_penaltys)
    train_stds=np.array(train_stds).reshape(n_Cs,number_penaltys)

    X_axis=np.log10(Cs)
    for i,value in enumerate(penalty):
        plt.errorbar(X_axis,-test_scores[:,i],yerr=test_stds[:,i],label=penalty[i]+str(value))
        plt.errorbar(X_axis,-train_scores[:,i],yerr=train_stds[:,i],
                     label=penalty[i]+'Train')
        plt.legend()
        plt.xlabel('log(C)')
        plt.ylabel('logloss')
        plt.show()
