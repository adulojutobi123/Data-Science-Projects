# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:46:55 2020

@author: User
"""
# In[1]
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
# In[2]
data_rgt=pd.read_excel('C:\\Users\\User\\Data Science Learning\Data.xls')
regt_dat=data_rgt[['Order Quantity','Total Price','Product Category','Customer Segment2']]
X_regt=regt_dat.drop('Product Category',axis=1)
y_regt=regt_dat['Product Category']

# In[3]
sns.barplot(x='Product Category', y='Order Quantity',data=regt_dat)
sns.pairplot(regt_dat, hue='Product Category',palette='Set1')
sns.heatmap(regt_dat.corr())

# In[4]
X_train, X_test, y_train, y_test=train_test_split(X_regt,y_regt, test_size=0.3, random_state=100)
testdata=X_test
testdata['Product Category']=y_test
dtree=DecisionTreeClassifier()
trye=dtree.fit(X_train,y_train)
pred=dtree.predict(X_test)
# print(classification_report(y_test, pred))
cm1=(confusion_matrix(y_test, pred))
nm=testdata['Product Category'].value_counts()
#-----------------------------------------------------------------------------------------------------
dtree2=DecisionTreeClassifier(criterion='gini',max_depth=5000,min_samples_split=10,ccp_alpha=0.00009)
trye2=dtree2.fit(X_train,y_train)
pred2=dtree2.predict(X_test)
# print(classification_report(y_test, pred2))
cm2=(confusion_matrix(y_test, pred2))
print(cm1,'\n',cm2,'\n', nm)
# In[5]
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
predrf=rf.predict(X_test)
# print(classification_report(y_test, predrf))
cmrf=(confusion_matrix(y_test, predrf))

#-----------------------------------------------------------------------------------------------------
rf2=RandomForestClassifier(n_estimators=100, criterion='gini',oob_score=True, bootstrap=True,
                           min_samples_split=10,ccp_alpha=0.00009)
rf2.fit(X_train,y_train)
predrf=rf2.predict(X_test)
# print(classification_report(y_test, predrf))
cmrf2=(confusion_matrix(y_test, predrf))

print(cmrf,'\n',cmrf2,'\n', nm)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print(cm1,'\n',cmrf)

print(cm2,'\n',cmrf2)

