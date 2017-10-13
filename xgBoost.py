#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:33:04 2017

@author: chirag212
"""

#==============================================================================
# XGBOOST
#==============================================================================

import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# Loading Dataset
seed = 2612
X=pd.read_csv("/home/chirag212/Kaggle_porto_seguro/train.csv")
Y=pd.read_csv("/home/chirag212/Kaggle_porto_seguro/test.csv")

X.head(n=5)
Y.head(n=5)

X_train = X[X.columns[2:]]
X_test = Y[Y.columns[1:]]

y_train = X[X.columns[1]]

X_train=np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)

#Oversampling
sm = SMOTE(random_state=seed, ratio = 'minority', m_neighbors=5)
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x_train_res, y_train_res):

    # Train the model
    model = XGBClassifier(learning_rate=0.01, reg_alpha= 0.01, reg_lambda=0.01, max_depth=7, 
                          max_delta_step= 1.8, colsample_bytree= 0.4,
                          subsample= 0.8, gamma= 0.65, n_estimators= 700)
    model.fit(x_train_res[train], y_train_res[train])    
    
    # evaluate the model
    y_pred = model.predict(x_train_res[test])
    y_pred_prob = model.predict_proba(x_train_res[test])
    
    # evaluate predictions
    accuracy = accuracy_score(y_train_res[test], y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_train_res[test], y_pred))
    cvscores.append(accuracy * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Evaluate on test data
y_test_pred_prob = model.predict_proba(X_test)

output = pd.DataFrame({'id': test['id'], 'target': np.ravel(y_test_pred_prob[:, 1])})
output.to_csv("/home/chirag212/Kaggle_porto_seguro/NVC_xgboost.csv", index=False)
# 0.277
