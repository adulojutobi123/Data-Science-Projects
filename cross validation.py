# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 09:25:50 2020

@author: User
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# In[2]
boston=load_boston()
boston_data=pd.DataFrame(boston.data, columns=boston.feature_names)
X=boston_data[['RM','AGE','TAX','RAD']].values
Y=boston.target

X_train, X_test, y_train, y_test=train_test_split(X,Y, test_size=0.3, random_state=100)

