#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:57:11 2020

@author: raschell
"""

import numpy as np
import pickle
import os
import copy

def group_fields(_dict, _fields, direction = 'column'):
    '''
    This function groups the data in the fields of a dict

    Parameters
    ----------
    _dict : dict
        Dictionary from which we take the signals.
    _fields : list of str
        List of the fields containing the signals.

    Returns
    -------
    data : np.ndarray
        Data containing the concatenated signals.

    '''
    
    if type(_dict) is not dict:
        raise Exception('ERROR: _dict input must be a dictionary!')
    
    if type(_fields) is not list:
        raise Exception('ERROR: _fields input must be a list of str!')

    # Check that signal in fields have the same length
    fields_len = [len(_dict[field]) for field in _fields]
    if (np.diff(fields_len) > 0.1).any():
        raise Exception('ERROR: signals have different length!')
    
    data = transpose(np.array([_dict[field] for field in _fields]), direction)
    
    return data

def convert_list_to_array(data_list, axis = 1):
    '''
    This function converts data in a list to np.array

    Parameters
    ----------
    data_list : list
        List of np.array signals with different dimensions.

    Returns
    -------
    data : np.ndarray
        Data containing the concatenated signals.

    '''
    
    data_out = []
    if type(data_list) is not list:
        raise Exception('ERROR: data_list input must be a list!')
        
    for iEl, el in enumerate(data_list):
        if type(el) != np.ndarray:
            raise Exception('ERROR: list elements must be np.ndarray!')
        else:
            if el.ndim == 1:
                el = el.reshape((el.shape[0],1))
            data_out.append(transpose(el, 'column'))
    
    # Check that signal in fields have the same length
    data_out_n = [dl.shape[0] for dl in data_out]
    if (np.diff(data_out_n) > 0.1).any():
        raise Exception('ERROR: data_list have different length!')
    
    return np.concatenate(data_out, axis = axis)

def flatten_list(_list, _tranform_to_array = False, unique = True):
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
    
    if unique:
        list_flat = unique_list(list_flat)
    
    if _tranform_to_array:
        list_flat - np.array(list_flat)
    
    return list_flat

def unique_list(_list):
    '''
    This function takes the unique elements in a list.

    Parameters
    ----------
    _list : list
        List of elements.

    Returns
    -------
    list_unique : list
        List with unique elements.

    '''
    
    if type(_list) is not list:
        raise Exception('ERROR: Input list is not a list!')
    
    list_unique = list(set(_list))
    
    return list_unique

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

def add(array1, array2):
    '''
    This function computes the sum of 2 arrays

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
    
    array = array1 + array2
    
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


def find_values(array, value = 1):
    '''
    This functions returns all indexes where an array has a certain value

    Parameters
    ----------
    array : np.ndarray / list
        Array of value.
    value : int, optional
        Value to look for in the array. The default is 1.

    Returns
    -------
    array
        Array of indexes.

    '''

    if type(array) is not list and type(array) is not np.ndarray:
        raise Exception('ERROR: array input must be either a list or a np.ndarray')
        
    if type(array) is list:
        array = np.array(array)
        
    return np.where(array == value)[0]
    

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


def transpose(array, direction = 'column'):
    '''
    This function transpose an array to a column or row array

    Parameters
    ----------
    array : list / np.ndarray
        Array of monopolar recording.
    direction : str, optional
        Direction of transposition: column or row. The default is 'column'.

    Returns
    -------
    array : np.ndarray
        Transposed array.

    '''
    
    if type(array) is list:
        array = np.array(array)
    
    if type(array) is not np.ndarray:
        raise Exception('ERROR: array in input is not list or np.ndarray!')
    
    if direction == 'column' and array.ndim>1:
        if array.shape[0]<array.shape[1]:
            array = array.T
            
    if direction == 'row' and array.ndim>1:
        if array.shape[0]>array.shape[1]:
            array = array.T
            
    return array
  

def copy_dict(_dict):
    '''
    This function copies the content of a list in another list, without referencing.

    Parameters
    ----------
    _dict : dict / list of _dict
        List of dictionaries.

    Returns
    -------
    dict_copy : list
        Copy of the list in input.

    '''
    
    input_dict = False
    if type(_dict) is dict:
        input_dict = True
        _dict = [_dict]
    
    if type(_dict) is not list:
        raise Exception('ERROR: list is input is not a list, but {}!'.format(type(_dict)))
        
    for element in _dict:
        if type(element) is not dict:
            raise Exception('ERROR: element in input list is not a list, but {}!'.format(type(element)))
            
    # Create copy list
    dict_copy = []
    for element in _dict:
        dict_copy.append(copy.deepcopy(element))
    
    if input_dict:
        dict_copy = dict_copy[0]
    
    return dict_copy

def open_figures_pickle(folder):
    '''
    This function opens all the pickle-saved figures in a folder

    Parameters
    ----------
    folder : str
        Folder containing the pickle-saved figures.

    Returns
    -------
    The figures.

    '''
    
    # Find all the files finishing in .pickle
    for file in os.listdir(folder):
        if file.endswith('.pickle'):
            figure_to_show = os.path.join(folder, file)
            figx = pickle.load(open(figure_to_show, 'rb'))
            figx.show()

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
    
    # Test the add function
    if (add(vector1, vector2) - (np.arange(10)+np.arange(10)) > 0.1).any():
        raise Exception('ERROR: Test add NOT passed!')
    else:
        print('Test add passed!')
    
    # Test the flatten_list function
    if (flatten_list([[[1, 2], [3]], [4, 5], [6, 7]], True) - np.array([1,2,3,4,5,6,7]) > 0.1).any() or \
        (flatten_list([[[1, 2], [3]], [4, [5]], [[6, 7]]], True) - np.array([1,2,3,4,5,6,7]) > 0.1).any():
        raise Exception('ERROR: Test flatten_list NOT passed!')
    else:
        print('Test flatten_list passed!')
    
    # Test the group_fields function
    dict_test = {'test1': np.arange(10), 'test2': np.arange(10)}
    if (group_fields(dict_test, ['test1','test2']) - np.array([np.arange(10),np.arange(10)]) > 0.1).any():
        raise Exception('ERROR: Test group_fields NOT passed!')
    else:
        print('Test group_fields passed!')
    
    # Test the transpose function
    if transpose(np.random.rand(100,3),'column').shape[0] != 100 or transpose(np.random.rand(100,3),'row').shape[0] != 3 or \
        transpose(np.random.rand(3,100),'column').shape[0] != 100 or transpose(np.random.rand(3,100),'row').shape[1] != 100:
        raise Exception('ERROR: Test transpose NOT passed!')
    else:
        print('Test transpose passed!')
    
    # Test the euclidean_distance function
    array_ones = np.array([np.ones(100), np.zeros(100)])
    array_zeros = np.array([np.zeros(100), np.zeros(100)])
    if (euclidean_distance(array_ones,array_zeros) - np.ones(100) > 0.1).any():
        raise Exception('ERROR: Test euclidean_distance NOT passed!')
    else:
        print('Test euclidean_distance passed!')
    
    print('All implemented tests passed!')
    
    