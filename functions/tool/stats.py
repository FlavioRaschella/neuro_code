#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:25:02 2020

@author: raschell
"""

import numpy as np
import scipy.stats
from utils import transpose

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