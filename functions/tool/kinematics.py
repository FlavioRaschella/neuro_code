#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:28:20 2020

@author: raschell
"""

import math
import numpy as np

def dotproduct(_v1, _v2):
    return sum((a*b) for a, b in zip(_v1, _v2))

def crossproduct(_v1, _v2):
    # return
    return np.cross(_v1,_v2)

def length(_v):
    return math.sqrt(dotproduct(_v, _v))

def magnitude(_v):
    return math.sqrt(np.sum(_v**2))

def compute_angle_3d(_v1, _v2, method = 'acos'):
    if type(_v1) == list:
        _v1 = np.array(_v1)
        if _v1.ndim == 1:
            _v1 = np.expand_dims(_v1, axis=0)
        
    if type(_v2) == list:
        _v2 = np.array(_v2)
        if _v2.ndim == 1:
            _v2 = np.expand_dims(_v2, axis=0)
    
    if 3 not in _v1.shape or 3 not in _v2.shape:
        raise Exception('ERROR: compute_angle_3d requires 3d vectors!')
    
    if _v1.shape[1] != 3:
        _v1 = _v1.T
        
    if _v2.shape[1] != 3:
        _v2 = _v2.T
    
    if method == 'acos':
        angle = np.degrees(np.array([np.arccos(dotproduct(v1_el, v2_el) / (length(v1_el) * length(v2_el))) for v1_el, v2_el in zip(_v1, _v2)]))
    elif method == 'asin':
        angle = np.degrees(np.array( [np.arcsin(magnitude(crossproduct(v1_el, v2_el)) / (length(v1_el) * length(v2_el))) for v1_el, v2_el in zip(_v1, _v2)] ))
    elif method == 'atan':
        angle = np.degrees(np.array( [math.atan2(magnitude(crossproduct(v1_el, v2_el)),dotproduct(v1_el, v2_el) ) for v1_el, v2_el in zip(_v1, _v2)] ))
    else:
        raise Exception('ERROR: slected methods is not implemented!')
         
    
    return angle


# Test code
if __name__ == '__main__':
    v1 = [0, -1, 0]
    v1_l = 1
    v2 = [1, 0, 0]
    v1_dot_v2 = 0
    
    if dotproduct(v1,v2) != 0 or dotproduct(v1,v1) != 1:
        raise Exception('ERROR: dotproduct gives the wrong result.')
    else:
        print('Test dotproduct passed!')
        
    if length(v1) != 1 or length(v2) != 1:
        raise Exception('ERROR: length gives the wrong result.')
    else:
        print('Test length passed!')

    v1_exp = np.tile(v1,(10,1))
    v2_exp = np.tile(v2,(10,1))
    dg_out = np.tile(90,(1,10)).astype('float')
    if (compute_angle_3d(v1, v2) - 90 > 0.1).any():
        raise Exception('ERROR: compute_angle_3d gives the wrong result.')
    else:
        print('Test compute_angle_3d passed!')
        
    if (compute_angle_3d(v1_exp, v2_exp) - dg_out > 0.1).any():
        raise Exception('ERROR: compute_angle_3d gives the wrong result.')
    else:
        print('Test compute_angle_3d passed!')
        
    print('All implemented tests passed!')
    
    