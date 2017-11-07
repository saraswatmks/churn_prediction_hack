#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:43:01 2017

@author: manish

In this script, we optimise adam.

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:48:03 2017

@author: manish
"""

import os

## set working directory
os.chdir('/home/manish/Desktop/Data2017/september/churn_prediction')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU

## declare global variables
parent_dir = '/home/manish/Desktop/Data2017/september/churn_prediction/'
child_dir = '/home/manish/Desktop/Data2017/september/churn_prediction/l1/'

nb_epoch = 120
nb_batch_size = 100

# model architechture
def keras_model(train):
    
    input_dim = train.shape[1]
    classes = 2
    
    model = Sequential()
    model.add(Dense(100, activation = 'relu', kernel_initializer='glorot_normal',input_shape = (input_dim, ))) #layer 1
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='glorot_normal',activation = 'relu')) #layer 2
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(classes, kernel_initializer='glorot_normal',activation = 'sigmoid')) #output
    model.compile(optimizer = 'adam', loss='binary_crossentropy')
    return model

## cross validate to generate train meta features
## train on all instances to generate test meta features

def fit_model(ptr, pte, target, features_names):
    
    #feature_names = [f for f in ptr.columns if f not in ['UCIC_ID','Responders']]
    #target = ptr['Responders']
    
    print('\n Starting validation.....')
    
    eval_matrix = pd.DataFrame({'UCIC_ID': ptr['UCIC_ID'], 'Responders':0})
    test_matrix = pd.DataFrame({'UCIC_ID':pte['UCIC_ID']})
    
    skf = StratifiedKFold(n_splits = 5, random_state = 420)
    
    for i, (train_index, test_index) in enumerate(skf.split(ptr, target)):
        
        print('\n Training Fold [{}/ 5]'.format(i+1))
        
        X_train, X_valid = ptr.iloc[train_index], ptr.iloc[test_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[test_index]
        
        y_train = np_utils.to_categorical(np.array(y_train))
        y_valid = np_utils.to_categorical(np.array(y_valid))
        
        X_train = np.matrix(X_train[feature_names])
        X_valid = np.matrix(X_valid[feature_names])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        
        class_weights = class_weight.compute_class_weight('balanced', np.unique(target), target)
        
        model = keras_model(X_train)
        model.fit(X_train, y_train, batch_size = nb_batch_size, epochs=nb_epoch,validation_data=(X_valid, y_valid), callbacks=[early_stopping], class_weight = class_weights)
        
        preds = model.predict_proba(X_valid, batch_size=128)[:,1]
        
        eval_matrix.loc[test_index, 'Responders'] = preds
        del model, X_train, y_train, X_valid, y_valid, preds
        
    ## generate test predictions
    X_train = np.matrix(ptr[feature_names])
    y_train = np_utils.to_categorical(np.array(target))
    X_test = np.matrix(pte[feature_names])
    
    model = keras_model(X_train)
    model.fit(X_train, y_train, batch_size=nb_batch_size, epochs=58)
    tpreds = model.predict_proba(X_test, batch_size=128)[:,1]
    
    test_matrix['Responders'] = tpreds
    return eval_matrix, test_matrix


if __name__ == '__main__':
    
    print('set path...\n')
    os.chdir('/home/manish/Desktop/Data2017/september/churn_prediction')
    
    print('Loading train and test data........\n')
    # with open('papa_train.pkl','r') as tr, open('papa_test.pkl','r') as te:
    train = pd.read_pickle('papa_train.pkl')
    test = pd.read_pickle('papa_test.pkl')
    
    print ('select features.....\n')    
  
    feature_names = [f for f in train.columns if f not in ['UCIC_ID','Responders']]
    
    target = train['Responders']
    
    train_bag = np.zeros((train.shape[0], 2))
    test_bag = np.zeros((test.shape[0], 2))
    
    seed_list = [12, 136, 458, 957, 140, 369, 475, 1234, 470, 5555]
    
    for x in range(len(seed_list)):
        
        print ("################################################")
        print ("##### Starting bag {0} ###########".format(x + 1))
        print ("################################################")
        
        
        np.random.seed(seed_list[x])
        
        eval_mat, test_mat = fit_model(train, test, target, feature_names)
        eval_mat, test_mat  = np.matrix(eval_mat), np.matrix(test_mat)
        
        train_bag += eval_mat
        test_bag += test_mat
        
    train_bag /= len(seed_list)
    test_bag /= len(seed_list)
    
    train_bag, test_bag = pd.DataFrame(train_bag), pd.DataFrame(test_bag)
    train_bag.columns = ["UCIC_ID", "Responders"]
    test_bag.columns = ["UCIC_ID", "Responders"]                
    
    train_prediction_file = child_dir + "keras_train2.csv"
    test_prediction_file = child_dir + "keras_test2.csv"
    
    # save the meta features to disk
    train_bag.to_csv(train_prediction_file, index = False)
    test_bag.to_csv(test_prediction_file, index = False)
    
   #  del train_bag, test_bag,standardize_df, train, test, tr, te, standardize_cols

        










