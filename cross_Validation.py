# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:13:12 2017

@author: Vineet
"""

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
train=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/train.csv")
test=pd.read_csv("C:/Users/Vineet/Desktop/Kaggle/test.csv")

train.head(n=5)
test.head(n=5)
# split into input (X) and output (Y) variables
X_train = train[train.columns[2:]]
X_test = test[test.columns[1:]]

y_train = train[train.columns[1]]

X_train =np.array(X_train)
y_train = np.array(y_train)
y_train_1 = np_utils.to_categorical(y_train)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X_train, y_train):
    cvscores
  # create model
    model = Sequential()
    model.add(Dense(60, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    
    model.fit(X_train[train], y_train_1[train], nb_epoch=50, batch_size=10, verbose=1)
    # evaluate the model
    scores = model.evaluate(X_train[test], y_train_1[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))