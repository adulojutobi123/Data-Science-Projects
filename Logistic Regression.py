# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 00:16:25 2020

@author: User
"""


# In[1]
# # Import needed packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
# In[2]
data=pd.read_excel('C:\\Users\\User\\Data Science Learning\Data.xls')
reg_dat=data[['Order Quantity','Total Price','Ship Mode']]
x_log=reg_dat.iloc[:,[0,1]].values
y_log=reg_dat.iloc[:,2].values
sns.heatmap(reg_dat.corr())
sc_X=StandardScaler()
X_train, X_test, y_train, y_test=train_test_split(x_log,y_log, test_size=1/3, random_state=0)
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
LogR=LogisticRegression(random_state=0)

LogR.fit(X_train,y_train)
ypred=LogR.predict(X_test)
cm=confusion_matrix(y_test, ypred)
Acc=(21+2372)/2800
misc=(20+387)/2800

# In[3]
from matplotlib.colors import ListedColormap
X_set,y_set=X_train, y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].min()+1,step=0.01),
                  np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].min()+1,step=0.01))
plt.contourf(X1,X2,LogR.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1], 
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic Regression Training set')
plt.xlabel('Order Quantity')
plt.ylabel('Total Sales')
plt.legend()