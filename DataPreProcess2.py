import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
train=pd.read_csv('FE_pima-indians-diabetes.csv')

y_trian=train['Target']
X_trian=train.drop(['Target'],axis=1)

#稀疏化
X_trian=csr_matrix(X_trian)
from sklearn.model_selection import train_test_split
X_trian_part,X_val,y_train_part,y_val=train_test_split(X_trian,y_trian,test_size=0.2
                                                       ,random_state=0)
#print(y_train_part.shape)

gamma_range=np.logspace(-3,9,13)
C_range=np.logspace(-2,10,13)
param_grid=dict(gamma=gamma_range,C=C_range)
#k折
cv=StratifiedKFold(n_splits=5)
#网格搜索
grid=GridSearchCV(estimator=SVC(kernel='rbf'),param_grid=param_grid,cv=cv)

grid.fit(X_trian_part,y_train_part)
y_predict=grid.predict(X_val)
accuracy=accuracy_score(y_val,y_predict)
print("正确率：",accuracy)
print("the best parameters are %s with a score of %.2f".format(-grid.best_params_,-grid.best_score_))



