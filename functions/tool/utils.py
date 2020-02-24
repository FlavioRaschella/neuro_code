#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:57:11 2020

@author: raschell
"""

import numpy as np

def flatten_list(_list, _tranform_to_array = False):
    '''
    This function flattens a list of lists to a simple list

    Parameters
    ----------
    _list : list
        List to be flattened.
    _tranform_to_array : bool, optional
        Transform to np.array. The default is False
    
    Returns
    -------
    list_flat : list
        Flattened list.

    '''
    
    
    list_flat = _list
    list_not_flat_flag = True
    
    while list_not_flat_flag:
        list_not_flat_flag = False
        list_flat_new = []
        for sublist in list_flat:
            if type(sublist) is list:
                for el in sublist:
                    list_flat_new.append(el)
            else:
                list_flat_new.append(sublist)
         
        for sublist in list_flat:
            if type(sublist) is list:
                list_not_flat_flag = True
                
        list_flat = list_flat_new
    
    if _tranform_to_array:
        list_flat - np.array(list_flat)
    
    return list_flat

def bipolar(array1, array2):
    '''
    This function computes the difference between 2 arrays

    Parameters
    ----------
    array1 : list / np.array
        Array of monopolar recording.
    array2 : list / np.array
        Array of monopolar recording.

    Returns
    -------
    array : list / np.array
        Array of bipolar recording.

    '''
    
    input_list = False
    if type(array1) is list:
        array1 = np.array(array1)
        input_list = True
        
    if type(array2) is list:
        array2 = np.array(array2)
        
    if type(array1) is not np.ndarray or type(array2) is not np.ndarray:
        raise Exception('ERROR: array in input are not list or np.ndarray!')
    
    array = array1 - array2
    
    if input_list:
        array = array.tolist()
    
    return array

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
    vector1 = np.arange(10)
    vector2 = np.arange(10)
    point = 3.1
    
    # Test the find_first function
    if find_first(point, vector1) == 4:
        print('Test find_first passed!')
    else:
        raise Exception('ERROR: Test find_first NOT passed!')
    
    # Test the bipolar function
    if (bipolar(vector1, vector2)-np.zeros(10) > 0.1).any():
        raise Exception('ERROR: Test bipolar NOT passed!')
    else:
        print('Test bipolar passed!')
    
    # Test the flatten_list function
    if (flatten_list([[[1, 2], [3]], [4, 5], [6, 7]], True) - np.array([1,2,3,4,5,6,7]) > 0.1).any() or \
        (flatten_list([[[1, 2], [3]], [4, [5]], [[6, 7]]], True) - np.array([1,2,3,4,5,6,7]) > 0.1).any():
        raise Exception('ERROR: Test flatten_list NOT passed!')
    else:
        print('Test flatten_list passed!')
    
    print('All implemented tests passed!')
    
    