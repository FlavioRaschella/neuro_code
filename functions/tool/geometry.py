#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:28:20 2020

@author: raschell
"""

import numpy as np

import math

from utils import transpose

def dotproduct(_v1, _v2):
    '''
    Compute the dot product of two vectors.

    Parameters
    ----------
    _v1 : np.ndarray, shape (n_array,)
        Vector for computing the dot product.
    _v2 : np.ndarray, shape (n_array,)
        Vector for computing the dot product.

    '''
    if len(_v1) != len(_v2):
        raise Exception('ERROR: vectors in input have different length: {} != {} !'.format(len(_v1),len(_v2)))
    
    return sum((a*b) for a, b in zip(_v1, _v2))

def crossproduct(_v1, _v2):
    '''
    Compute the cross product of two vectors.

    Parameters
    ----------
    _v1 : np.ndarray, shape (n_array,)
        Vector for computing the cross product.
    _v2 : np.ndarray, shape (n_array,)
        Vector for computing the cross product.

    '''
    if len(_v1) != len(_v2):
        raise Exception('ERROR: vectors in input have different length: {} != {} !'.format(len(_v1),len(_v2)))
    
    return np.cross(_v1,_v2)

def length(_v):
    '''
    Compute the length of a vector.

    Parameters
    ----------
    _v1 : np.ndarray, shape (n_array,)
        Vector on which compute the length.

    '''
    return math.sqrt(dotproduct(_v, _v))

def compute_angle_3d(_v1, _v2, method = 'acos'):
    '''
    This function computes the angle between 2 3d vectors.

    Parameters
    ----------
    _v1 : np.ndarray, shape (n_array,)
        Vector for computing the cross product.
    _v2 : np.ndarray, shape (n_array,)
        Vector for computing the cross product.
    method : str, optional
        Method used for computing the angle. The default is 'acos'.

    Returns
    -------
    angle : np.ndarray, shape (n_array,)
        Angle between the 2 vectors.

    '''
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
    
    if method not in ['acos','asin','atan']:
        raise Exception('ERROR: method can only be "acos", "asin", "atan"! You inputed {}.'.format(method))
    
    if _v1.shape[1] != 3:
        _v1 = _v1.T
        
    if _v2.shape[1] != 3:
        _v2 = _v2.T
    
    if method == 'acos':
        angle = np.degrees(np.array([np.arccos(dotproduct(v1_el, v2_el) / (length(v1_el) * length(v2_el))) for v1_el, v2_el in zip(_v1, _v2)]))
    elif method == 'asin':
        angle = np.degrees(np.array( [np.arcsin(length(crossproduct(v1_el, v2_el)) / (length(v1_el) * length(v2_el))) for v1_el, v2_el in zip(_v1, _v2)] ))
    elif method == 'atan':
        angle = np.degrees(np.array( [math.atan2(length(crossproduct(v1_el, v2_el)),dotproduct(v1_el, v2_el) ) for v1_el, v2_el in zip(_v1, _v2)] ))
    else:
        raise Exception('ERROR: slected methods is not implemented!')
         
    return angle


def euclidean_distance(array1, array2):
    '''
    This function computes the euclidean distance between 2 points

    Parameters
    ----------
    array1 : list / np.ndarray
        First array.
    array2 : list / np.ndarray
        Second array.

    Returns
    -------
    distance : list / np.array
        Array of the euclidean distance instant by instance.

    '''
    
    input_list = False
    if type(array1) is list:
        array1 = np.array(array1)
        input_list = True
        
    if type(array2) is list:
        array2 = np.array(array2)
        
    if type(array1) is not np.ndarray or type(array2) is not np.ndarray:
        raise Exception('ERROR: array(s) in input is(are) not list or np.ndarray!')
    
    array_diff = transpose(array1-array2, direction = 'column')
    distance = np.linalg.norm(array_diff, axis = 1)
    
    if input_list:
        distance = distance.tolist()
    
    return distance

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
        
    # Test the euclidean_distance function
    array_ones = np.array([np.ones(100), np.zeros(100)])
    array_zeros = np.array([np.zeros(100), np.zeros(100)])
    if (euclidean_distance(array_ones,array_zeros) - np.ones(100) > 0.1).any():
        raise Exception('ERROR: Test euclidean_distance NOT passed!')
    else:
        print('Test euclidean_distance passed!')
        
    print('All implemented tests passed!')
    
    