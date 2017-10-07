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

train=pd.read_csv("/home/chirag212/Kaggle_porto_seguro/train.csv")
test=pd.read_csv("/home/chirag212/Kaggle_porto_seguro/test.csv")

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
dt = tree.DecisionTreeClassifier(random_state=1)
results = dt.fit(X_train ,y_train)

Y_test_pred_prob_DT = dt.predict_proba(X_test)


#Ramdom Classifier on Train Data
rc = RandomForestClassifier(random_state=1)
results = rc.fit(X_train ,y_train)

Y_test_pred_prob_rf = rc.predict_proba(X_test)

#Voting Classifier

eclf = VotingClassifier(estimators=[('lr', logreg), ('dt', dt), ('rc', rc)],
                        voting='soft', weights=[1,1,1])
eclf = eclf.fit(X_train, y_train)

Y_test_pred_VC = eclf.predict_proba(X_test)

output = pd.DataFrame({'id': test['id'], 'target': np.ravel(Y_test_pred_VC[:, 1])})
output.to_csv("/home/chirag212/Kaggle_porto_seguro/NVC_v5_voting.csv", index=False)
