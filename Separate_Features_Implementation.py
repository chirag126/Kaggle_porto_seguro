# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:13:52 2017

@author: Vineet
"""
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

train=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/train.csv")
test=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/test.csv")

train.head(n=5)
test.head(n=5)

X_train = train[train.columns[2:]]
X_test = test[test.columns[1:]]

y_train = train[train.columns[1]]

X_train_ind=train.filter(like="ind")
X_train_car=train.filter(like="car")
X_train_reg=train.filter(like="reg")
X_train_calc=train.filter(like="calc")



X_train_ind =np.array(X_train_ind)
X_train_car =np.array(X_train_car)
X_train_reg =np.array(X_train_reg)
X_train_calc =np.array(X_train_calc)


y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train)

X_test =np.array(X_test)

train_size = [20, 18, 19]
# create model
score = []
for i in range(3):
    
    if i==0:
        X = X_train_calc
    elif i==1:
        X = np.concatenate((X_train_car, X_train_reg), axis=1)    
    else:
        X = X_train_ind
        
    model = Sequential()
    model.add(Dense(30, input_dim=train_size[i], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit the model
    model.fit(X, y_train, nb_epoch=50, batch_size=32)
    
    # predict from model
    y_pred = model.predict_proba(X_test)
    score.append(y_pred[:, 1])

    np.unique(y_pred[:, 1])
    
output = pd.DataFrame({'id': test['id'], 'target': np.ravel(y_pred[:, 1])})
output.to_csv(r"C:\Users\Vineet\Desktop\Kaggle\NVC_v3_epoch_50.csv", index=False)   
print ('process done')



