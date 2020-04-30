#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:25:02 2020

@author: raschell
"""

import numpy as np
import scipy.stats
from utils import transpose

# =============================================================================
# Confidence interval
# =============================================================================

def confidence_interval_sample(data, confidence=0.95):
    '''
    Compute confidence interval over samples.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples,)
        Samples on which compute the confidence interval.
    confidence : float, optional
        Range of the confidence interval, between 0 and 1. The default is 0.95.

    Returns
    -------
    m : np.ndarray, shape (1,)
        Mean of the sample.
    m-h : np.ndarray, shape (1,)
        Upper bound of the confidence interval.
    m-h : np.ndarray, shape (1,)
        Lower bound of the confidence interval.

    '''
    a = 1.0 * np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
    # scipy.stats.t.interval(0.95, 196, loc=np.mean(a), scale=scipy.stats.sem(a))
    return m, m-h, m+h

def confidence_interval(data, confidence=0.95):
    '''
    Compute confidence interval over a signal.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_signals)
        Signals on which compute the confidence interval.
    confidence : float, optional
        Range of the confidence interval, between 0 and 1. The default is 0.95.

    Returns
    -------
    m : np.ndarray, shape (n_samples,)
        Mean of the signal.
    up : np.ndarray, shape (n_samples,)
        Upper bound of the confidence interval.
    dw : np.ndarray, shape (n_samples,)
        Lower bound of the confidence interval.

    '''
    
    if type(data) != np.ndarray:
        data = np.array(data)
    
    data = transpose(data,'column')
    
    if data.ndim == 1:
        m, up, dw = confidence_interval_sample(data, confidence)
    elif data.ndim == 2:
        m = np.empty((data.shape[0],))
        up = np.empty((data.shape[0],))
        dw = np.empty((data.shape[0],))
        for iSmp, data_smpl in enumerate(data):
            m[iSmp], dw[iSmp], up[iSmp] = confidence_interval_sample(data_smpl, confidence)
    else:
        raise Exception('ERROR: dimensioin of input data is > 2!')
    
    return m, dw, up


# =============================================================================
# Boostrapping
# =============================================================================
def boostrap(data, n_straps = 10000):
    
    if data.ndim == 1:
        raise Exception('ERROR: 1d boostrap still not implemented!')
    elif data.ndim == 2:
        n_data = np.sum(data)
        bins = np.concatenate([[0.5],np.cumsum(np.reshape(data.T,(np.prod(data.shape),))) + 0.5])
        boot_idx = np.random.randint(low=0,high=n_data,size=(n_data, n_straps))
        
        boot_count = []
        for iStr in range(n_straps):
            boot_count.append(np.histogram(boot_idx[:,iStr], bins = bins)[0])
            # boot_count.append(np.histogram(np.digitize(boot_idx[:,iStr], bins = bins, right = False)-1, bins = len(bins), range = (0,len(bins)))[0])
        boot_count = np.array(boot_count).T
        
        boot_data = np.moveaxis(np.reshape(boot_count,(data.shape[0], data.shape[0], n_straps)),0,1)
    
    return boot_data
    


# EOF