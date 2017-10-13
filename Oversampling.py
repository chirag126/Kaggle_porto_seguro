# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:06:56 2017

@author: Vineet
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem.porter import PorterStemmer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
train=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/train.csv")
test=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/test.csv")

train.head(n=5)
test.head(n=5)

X_train = train[train.columns[2:]]
X_test = test[test.columns[1:]]

y_train = train[train.columns[1]]

X_train=np.array(X_train)
y_train = np.array(y_train)


#Oversampling
sm = SMOTE(random_state=12, ratio = 'minority')
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)

clf_rf = RandomForestClassifier( random_state=12)
clf_rf.fit(x_train_res, y_train_res)
Y_test_pred_rf = clf_rf.predict(X_test)

Y_test_pred_prob_rf = clf_rf.predict_proba(X_test)

output = pd.DataFrame({'id': test['id'], 'target': np.ravel(Y_test_pred_prob_rf[:, 1])})
output.to_csv(r"C:\Users\Vineet\Desktop\Kaggle\NVC_v5_rf_oversampling.csv", index=False)   
print ('process done')

#Logistic Regression on Train data

logreg = linear_model.LogisticRegression(C=.001, multi_class='multinomial',solver='lbfgs',random_state=1)
results=logreg.fit(X_train, y_train)   


Y_test_pred = logreg.predict(X_test)
Y_test_pred_prob = logreg.predict_proba(X_test)


#Decision Tree
clf = tree.DecisionTreeClassifier(random_state=1)
results = clf.fit(X_train ,y_train)
Y_test_pred_DT = logreg.predict(X_test)
Y_test_pred_prob_DT = clf.predict_proba(X_test)


#Ramdom Classifier on Train Data


clf1 = RandomForestClassifier(random_state=1)
results = clf1.fit(X_train ,y_train)


Y_test_pred_rf = clf1.predict(X_test)
Y_test_pred_prob_rf = clf1.predict_proba(X_test)

#Voting Classifier

eclf = VotingClassifier(estimators=[('lr', logreg), ('rf', clf), ('ran', clf1)],
                        voting='soft', weights=[1,1,1])

eclf = eclf.fit(X_train, y_train)
Y_test_pred_VC = eclf.predict_log_proba(X_test)

output = pd.DataFrame({'id': test['id'], 'target': np.ravel(Y_test_pred_prob[:, 1])})
output.to_csv(r"C:\Users\Vineet\Desktop\Kaggle\NVC_v4_logistic_50.csv", index=False)   
print ('process done')