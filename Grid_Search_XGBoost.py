# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:48:31 2017

@author: Vineet
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt

# Loading Dataset
seed = 2612
X=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/train.csv")
Y=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/test.csv")

X.head(n=5)
Y.head(n=5)

#X_train = X[X.columns[2:]]
#X_test = Y[Y.columns[1:]]

#y_train = X[X.columns[1]]

#X_train=np.array(X_train)
#y_train = np.array(y_train)

#X_test = np.array(X_test)

dtrain=np.array(X[X.columns[1:]]) #keeping target in X

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['target'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['target'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ( "\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    
#######################Step 1: Fix learning rate 0.1 and number of estimators for tuning tree-based parameters= 1000#############################
#Choose all predictors except target & IDcols
predictors = [x for x in X.columns if x not in ['id','target']]
xgb1 = XGBClassifier(
 learning_rate =0.1,#0.01
 n_estimators=1000,#700
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, dtrain, predictors)   

######################Step 2: Tune max_depth and min_child_weight#########################################################################
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier
( learning_rate =0.1,
n_estimators=140, #change it based on previous output
max_depth=5,
min_child_weight=1, 
gamma=0, 
subsample=0.8, 
colsample_bytree=0.8,
objective= 'binary:logistic', 
nthread=4, scale_pos_weight=1,
seed=27), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(dtrain[predictors],dtrain['target'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


####################Step 3: Tune gamma##################################################################################################


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier
( learning_rate =0.1,
 n_estimators=140, #change
 max_depth=4,#change 
 min_child_weight=6, #change
 gamma=0, 
 subsample=0.8, 
 colsample_bytree=0.8,
 objective= 'binary:logistic', 
 nthread=4, 
 scale_pos_weight=1,
 seed=27), 
param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(dtrain[predictors],dtrain['target'])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

#This shows that our original value of gamma, i.e. 0 is the optimum one. 
#Before proceeding, a good idea would be to re-calibrate the number of boosting rounds for the updated parameters.


##################Step 4: Tune subsample and colsample_bytree################################################

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier
( learning_rate =0.1, 
 n_estimators=177,
 max_depth=4,
 min_child_weight=6, 
 gamma=0, 
 subsample=0.8, 
 colsample_bytree=0.8,
 objective= 'binary:logistic', 
 nthread=4, 
 scale_pos_weight=1,
 seed=27), 
param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(dtrain[predictors],dtrain['target'])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_



###########################Step 5: Tuning Regularization Parameters############################################
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier
( learning_rate =0.1, 
 n_estimators=177, 
 max_depth=4,
 min_child_weight=6, 
 gamma=0.1, 
 subsample=0.8, 
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4, 
 scale_pos_weight=1,
 seed=27), 
param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(dtrain[predictors],dtrain['target'])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


############################Final Model after tuning using the above values##################################


xgb3 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb3, dtrain, predictors)

###########################Step 6: Reducing Learning Rate#####################################################

xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, dtrain, predictors)