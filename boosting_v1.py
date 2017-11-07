#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:43:40 2017

@author: manish
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import os

## declare global variables
parent_dir = '/home/manish/Desktop/Data2017/september/churn_prediction/'
child_dir = '/home/manish/Desktop/Data2017/september/churn_prediction/l1/'

def runXGB(train, test, target, features, wparams = 0):
    
    n_folds = 5
    skf = StratifiedKFold(n_splits = n_folds, random_state= 1001)
    
    oof_pred = pd.DataFrame({'UCIC_ID':train['UCIC_ID'], 'Responders':0})
    all_predictions = pd.DataFrame()
    
    if wparams == 0:
        print('Default Run is Tree....')
        set_params = {"objective": "binary:logistic",
                      "booster": "gbtree",
                      "nthread": 4,
                      "eta": 0.1, # 0.1
                      "max_depth": 6, # 7
                      "subsample": 1,
                      #"min_child_weight": 2**3,
                      "seed": 2016,     
                      #         "tree_method": "exact",
                      "eval_metric":"auc",
                      "silent":1}
    elif wparams == 1:
        print('Now running Logistic Regression....')
        set_params = {"objective": "binary:logistic",
                      "booster": "gblinear",
                      "nthread": 4,
                      "alpha":4,
                      "lambda":2,
                      "seed": 2016, 
                      "eval_metric":"auc",
                      "silent":1}
    elif wparams == 2:
        print('Now running Random Forest.......')
        set_params = {'booster': 'dart',
                      'max_depth': 5, 'learning_rate': 0.1,
                      'objective': 'binary:logistic',
                      'sample_type': 'uniform',
                      'normalize_type': 'tree',
                      'rate_drop': 0.1,
                      'skip_drop': 0.5,
                      'eval_metric':'auc',
                      "silent":1}        
   
    increase = True
    best_rounds = []
    
    
    for i, (train_index, valid_index) in enumerate(skf.split(train, target)):
        print('[{}/{} Folds]'.format(i+1, n_folds))
        
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        
        
        if increase:
            pos = pd.Series(target == 1)
            
            X_train = pd.concat([X_train, X_train[pos]], axis=0)
            y_train = pd.concat([y_train, y_train[pos]], axis=0)
            
            idx = np.arange(len(X_train))
            np.random.shuffle(idx)
            
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
            
        dtrain = xgb.DMatrix(X_train[features], y_train, missing=-1)
        dvalid = xgb.DMatrix(X_valid[features], y_valid, missing=-1)
        dtest = xgb.DMatrix(test[features])
            
        watchlist = [(dtrain, 'train'),(dvalid, 'valid')]
        clf1 = xgb.train(set_params, dtrain, num_boost_round=5000, evals=watchlist, maximize=True, verbose_eval=20, early_stopping_rounds=40)    
        
        best_rounds.append(clf1.best_iteration)
        
        preds1 = clf1.predict(dvalid)
        oof_pred.loc[valid_index, 'Responders'] = preds1
        
    
    print ('now training on whole train instances.....\n')    
    ## for test, predict on whole data
    Ndtrain = xgb.DMatrix(data=train[features], label = target, missing=np.nan)
    
    n_round = int(np.round(np.mean(best_rounds)))
    clf2 = xgb.train(set_params, Ndtrain, n_round)
    preds2 = clf2.predict(dtest)
    
    all_predictions['pred' + str(i)] = preds2
    
    return oof_pred, all_predictions
    


def runLGB(train, test, target, features, wparams = 0):
    
    n_folds = 5
    skf = StratifiedKFold(n_splits = n_folds, random_state= 1001)
    
    oof_pred = pd.DataFrame({'UCIC_ID':train['UCIC_ID'], 'Responders':0})
    all_predictions = pd.DataFrame()
    
    if wparams == 0:
        print('Default Run is Tree....')
        set_params = {'learning_rate':0.1,
                      'max_depth':6,
                      'boosting':'gbdt',
                      'objective':'binary',
                      'metric':'auc',
                      'seed':2017,
                      'feature_fraction':1,
                      'bagging_fraction':1,
                      'num_leaves':30,
                      'lambda_l1':16,
                      'lambda_l2':16,
                      "verbose":0}
    elif wparams == 1:
        print('Now running Random Forest....')
        set_params = {'learning_rate':0.1,
                      'max_depth':6,
                      'boosting':'rf',
                      'objective':'binary',
                      'metric':'auc',
                      'seed':2017,
                      'feature_fraction':1,
                      'bagging_fraction':1,
                      'num_leaves':30,
                      'lambda_l1':16,
                      'lambda_l2':16,
                      "verbose":0}    
    elif wparams == 2:
        print('Now running Additive Regression Trees (DART).......')
        set_params = {'learning_rate':0.1,
                      'max_depth':6,
                      'boosting':'dart',
                      'objective':'binary',
                      'metric':'auc',
                      'seed':2017,
                      'feature_fraction':1,
                      'bagging_fraction':1,
                      'num_leaves':30,
                      'lambda_l1':16,
                      'lambda_l2':16,
                      "verbose":0}

   
    increase = True
    best_rounds = []
    
    
    for i, (train_index, valid_index) in enumerate(skf.split(train, target)):
        print('[{}/{} Folds]'.format(i+1, n_folds))
        
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
        
        
        if increase:
            pos = pd.Series(target == 1)
            
            X_train = pd.concat([X_train, X_train[pos]], axis=0)
            y_train = pd.concat([y_train, y_train[pos]], axis=0)
            
            idx = np.arange(len(X_train))
            np.random.shuffle(idx)
            
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
            
        dtrain = lgb.Dataset(X_train[features], y_train)
        dvalid = lgb.Dataset(X_valid[features], y_valid)
            
        clf1 = lgb.train(set_params, dtrain, num_boost_round=5000, valid_sets=dvalid, verbose_eval=20, early_stopping_rounds=40)
        
        best_rounds.append(clf1.best_iteration)
        
        preds1 = clf1.predict(X_valid[features])
        oof_pred.loc[valid_index, 'Responders'] = preds1
         
    print ('now training on whole train instances.....\n')
    Ndtrain = lgb.Dataset(data=train[features], label = target)
    
    n_round = int(np.round(np.mean(best_rounds)))
    clf2 = lgb.train(set_params, Ndtrain, n_round)
    preds2 = clf2.predict(test[features])
    
    all_predictions['pred' + str(i)] = preds2
    return oof_pred, all_predictions


    
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
    
    print('Running lightgbm....\n')
    
#    oof_tree, tree_test = runLGB(train, test, target, feature_names) 
#    print('save tree predictions....')
#    oof_tree.to_csv(child_dir + 'lgbTree_train.csv', index=False)
#    tree_test.to_csv(child_dir + 'lgbTree_test.csv', index=False)
#      
#    oof_rf, rf_test = runLGB(train, test, target, feature_names, wparams=1)  ## this is too slow
#    print('save forest predictions....')
#    oof_rf.to_csv(child_dir + 'lgbRF_train.csv', index=False)
#    rf_test.to_csv(child_dir + 'lgbRF_test.csv', index=False)
#    
#    oof_dart,dart_test = runLGB(train, test, target, feature_names, wparams=2)
#    print('save dart predictions....')
#    oof_dart.to_csv(child_dir + 'lgbDart_train.csv', index=False)
#    dart_test.to_csv(child_dir + 'lgbDart_test.csv', index=False)
#
#    del oof_rf, rf_test
    
    print('Running xgboost.....\n')
    
#    oof_tree, tree_test = runXGB(train, test, target, feature_names) 
#    print('save tree predictions....')
#    oof_tree.to_csv(child_dir + 'xgbTree_train.csv', index=False)
#    tree_test.to_csv(child_dir + 'xgbTree_test.csv', index=False)
#    
    oof_linear,linear_test = runXGB(train, test, target, feature_names, wparams=1)
    print('save logistic predictions....')
    oof_linear.to_csv(child_dir + 'xgbLR_train.csv', index=False)
    linear_test.to_csv(child_dir + 'xgbLR_test.csv', index=False)
     
    oof_rf, rf_test = runXGB(train, test, target, feature_names, wparams=2) 
    print('save forest predictions....')
    oof_rf.to_csv(child_dir + 'xgbRF_train.csv', index=False)
    rf_test.to_csv(child_dir + 'xgbRF_test.csv', index=False)

    del oof_tree, test_tree, oof_linear,linear_test, oof_rf, rf_test
    

    


    
    
    
    


    


