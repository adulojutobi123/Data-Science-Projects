# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:09:27 2020

@author: User
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer_data=datasets.load_breast_cancer()

X_train, X_test, y_train, y_test=train_test_split(cancer_data.data,cancer_data.target, test_size=0.3, random_state=100)

clsvm=svm.SVC(kernel='linear')

clsvm.fit(X_train,y_train)

svmpred=clsvm.predict(X_test)

print('Accuracy:',metrics.accuracy_score(y_test, svmpred))

print('Precision:',metrics.precision_score(y_test, svmpred))

print('Recall:',metrics.recall_score(y_test, svmpred))


print(metrics.classification_report(y_test, svmpred))

# In[2]
import matplotlib.pyplot as plt

digit=datasets.load_digits()

clf=svm.SVC(gamma=0.000001, C=150)

X=digit.data
Y=digit.target
clf.fit(X,Y)

plt.imshow(digit.images[3],interpolation='nearest')