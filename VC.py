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
train=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/train.csv")
test=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/test.csv")

train.head(n=5)
test.head(n=5)

X_train = train[train.columns[2:]]
X_test = test[test.columns[1:]]

y_train = train[train.columns[1]]

X_train=np.array(X_train)
y_train = np.array(y_train)


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
Y_test_pred_VC = clf1.predict_log_proba(X_test)
print(eclf.score(X_test_tf, y_test)) #0.332358104154
save_list(Y_test_pred,r'C:/Users/Vineet/Desktop/Adv Text/Session3/664694644_voting.txt')
