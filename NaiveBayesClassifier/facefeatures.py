# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 03:04:50 2015

@author: Naman
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


csvfile=pd.read_csv('FacialFeaturesJaffeNew.csv')
#csvfile=pd.read_csv('FacialFeaturesJaffe.csv',delimiter='\t')
csvfile=csvfile.drop(csvfile.columns[[0]],axis=1)


#X_=csvfile.drop(csvfile.columns[[12]],axis=1)
y_=csvfile['Emotion']

n_labels=len(np.unique(y_))
train=csvfile[csvfile.Emotion==-1]
test=csvfile[csvfile.Emotion==-1]
for i in range(n_labels):
    A=csvfile[csvfile.Emotion==i]
    if i==1:
        A.Emotion=0
    X1=A.drop(A.tail(5).index)
       
    X2=A.tail(5)
    #tmp1=A-A.mean()
    #tmp2=A.std()
    #A=tmp1/tmp2
    #A.Emotion=i
    #X1=A

    if i!=2 and i!=4:
        train=pd.concat([train,X1])
        test=pd.concat([test,X2])
    


   
X=train.drop(train.columns[[train.shape[1]-1]],axis=1)

y=train['Emotion']
y_test=test['Emotion']
X_test=test.drop(test.columns[[train.shape[1]-1]],axis=1)
    
X=X.values
y=y.values
X_test=X_test.values
y_test=y_test.values

rf=RandomForestClassifier(n_estimators=200)
rf=rf.fit(X,y)
print rf.score(X_test,y_test)



clfG = GaussianNB()
clfM = MultinomialNB()
clfB = BernoulliNB()
yG=clfG.fit(X, y)
yM=clfM.fit(X,y)
yB=clfB.fit(X,y)
joblib.dump(yG,'Bayes_Gaussian.pkl')

print "On Test Data: ",yG.score(X_test,y_test),yM.score(X_test,y_test),yB.score(X_test,y_test)
print "On Training Data: ",yG.score(X,y),yM.score(X,y),yB.score(X,y)
Predicted=clfG.predict(X)
Predicted_class=clfG.predict_proba(X)

