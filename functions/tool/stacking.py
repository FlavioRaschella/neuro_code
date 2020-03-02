#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:48:05 2020

@author: raschell
"""

import numpy as np

# Function for out-of-fold prediction
def get_oof(clf, x_train, y_train, x_test, kf):
    train_n = x_train.shape[0]
    test_n = x_test.shape[0]
    oof_train = np.zeros((train_n,))
    oof_test = np.zeros((test_n,))
    oof_test_skf = np.empty((kf.n_splits, test_n))
    
    # Split data in kf.n_splits training vs testing samples
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # Select train and test sample
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        # Train classifier on training sample
        clf.train(x_tr, y_tr)
        
        # Predict classifier for testing sample
        oof_train[test_index] = clf.predict(x_te)
        # Predict classifier for original test sample
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # Take the median of all kf.n_splits test sample predictions
    # (changed from mean to preserve binary classification)
    oof_test[:] = np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)