#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:58:58 2017

@author: manish
"""

from sklearn.decomposition import TruncatedSVD, PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import StandardScaler

def engineer2(train, test):
    
    myfeats = [f for f in train.columns if f not in ['UCIC_ID','Responders']]
    
    scaler = StandardScaler()
    
    slr = scaler.fit(train[myfeats])
    
    dim_train = slr.transform(train[myfeats])
    dim_test = slr.transform(test[myfeats])
    
    n_comp = 10
    
    print('Starting decomposition.........\n')
    
    tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
    tsvd_train = tsvd.fit_transform(dim_train)
    tsvd_test = tsvd.transform(dim_test)

    pca = PCA(n_components=n_comp, random_state=420)
    pca_train = pca.fit_transform(dim_train)
    pca_test = pca.transform(dim_test)
    
    ica = FastICA(n_components=n_comp, random_state=2030)
    ica_train = ica.fit_transform(dim_train)
    ica_test = ica.transform(dim_test)
    
    grp = GaussianRandomProjection(n_components=n_comp, random_state=42)
    grp_train = grp.fit_transform(dim_train)
    grp_test = grp.transform(dim_test)    
    
    srp = SparseRandomProjection(n_components=n_comp, random_state=42)
    srp_train = srp.fit_transform(dim_train)
    srp_test = srp.transform(dim_test)    
    
    for i in range(1, n_comp + 1):
        train['pca_' + str(i)] = pca_train[:,i-1]
        test['pca_' + str(i)] = pca_test[:,i-1]
        
        train['tsvd_' + str(i)] = tsvd_train[:,i-1]
        test['tsvd_' + str(i)] = tsvd_test[:,i-1]
        
        train['ica_' + str(i)] = ica_train[:,i-1]
        test['ica_' + str(i)] = ica_test[:,i-1]
        
        train['grp_' + str(i)] = grp_train[:,i-1]
        test['grp_' + str(i)] = grp_test[:,i-1]
        
        train['srp_' + str(i)] = srp_train[:,i-1]
        test['srp_' + str(i)] = srp_test[:,i-1]
        
    
    del dim_train, dim_test
    
    return train, test
    