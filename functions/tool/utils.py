#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:57:11 2020

@author: raschell
"""

import numpy as np

def find_first(point, vector):
    '''
    This function finds the index of the first element in the vector greater than the points
    
    Parameters
    ----------
    point : int,float
        Point to find in the vector.
    vector : list,np.ndarray
        vector containing the point.

    Returns
    -------
    point_idx : int
        index of the point in the vactor.

    '''
    
    if type(vector) is not list and type(vector) is not np.ndarray:
        raise Exception('ERROR: vector input must be either a list or a np.ndarray')
    
    try:
        point_idx = next(x for x, val in enumerate(vector) if val > point)
    except:
        point_idx = None
    
    return point_idx


if __name__ == '__main__':
    vector = np.arange(10)
    point = 3.1
    
    if find_first(point, vector) == 4:
        print('Test find_first passed!')
    else:
        raise Exception('ERROR: Test find_first NOT passed!')
    
    print('All implemented tests passed!')
    
    