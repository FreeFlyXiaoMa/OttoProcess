import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

from sklearn import metrics

train=pd.read_csv('Otto_train.csv')
#print(train.head())



