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

X_train =np.array(X_train)
y_train = np.array(y_train)
y_train = np_utils.to_categorical(y_train)

X_test =np.array(X_test)

# create model
model = Sequential()
model.add(Dense(30, input_dim=57, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, nb_epoch=50, batch_size=32)

# predict from model
y_pred = model.predict_proba(X_test)
np.unique(y_pred[:, 1])
output = pd.DataFrame({'id': test['id'], 'target': np.ravel(y_pred[:, 1])})
output.to_csv(r"C:\Users\Vineet\Desktop\Kaggle\NVC_v3_epoch_50.csv", index=False)   
print ('process done')



