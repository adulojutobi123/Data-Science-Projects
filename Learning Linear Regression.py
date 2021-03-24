# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:20:24 2020

@author: User
"""
# In[1]
# # Import needed packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# In[2]
data=pd.read_excel('C:\\Users\\User\\Data Science Learning\Data.xls')
reg_dat=data[['Order Quantity','Unit Price']]

sns.distplot(reg_dat['Order Quantity'], kde=False, bins=100)
sns.barplot(reg_dat['Order Quantity'],reg_dat['Unit Price'])

plt.bar(reg_dat['Order Quantity'],reg_dat['Unit Price'])
sns.heatmap(reg_dat.corr())

# In[3]
#Spliting data to training and testing
x=reg_dat.iloc[:,:-1].values
y=reg_dat.iloc[:,1].values
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=1/3, random_state=0)

# In[4]
#Fit model
lr=LinearRegression()
lr.fit(X_train,y_train)

#Predict Test Result
y_pred=lr.predict(X_test)

plt.scatter(X_test, y_test,color='blue',edgecolor='black')


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
