#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:25:02 2020

@author: raschell
"""

import numpy as np
import scipy.stats


def confidence_interval_sample(data, confidence=0.95):
    a = 1.0 * np.array(data)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return m, m-h, m+h

def confidence_interval(data, confidence=0.95):
    
    if type(data) != np.ndarray:
        data = np.array(data)
        
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