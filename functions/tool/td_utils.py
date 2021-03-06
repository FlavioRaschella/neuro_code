#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:48 2020

@author: raschell
"""

'''
td is the trial data structure.
The format of td is as following:
    NAME keys :  contain signals
    PARAMS key : contain trial data information
        folder : Name of the folder where the data are taken from
        
        file :   Name of the file where the data are taken from
        
        data :   Data_1  : signals : name of the signals to be used for analysis
                           time : name of the signal containing time information
                           fs : sampling frequency
                 Data_n  : signals : name of the signals to be used for analysis
                           time : name of the signal containing time information
                           fs : sampling frequency         
        
        event :  Event_1 : signals : name of the signals to be used for analysis
                           kind : saved in time or samples
                 Event_n : signals : name of the signals to be used for analysis
                           kind : saved in time or samples
        
'''

'''
There are several method categories:
    - Control fields
    - Removing
    - Adding
    - Combining
    - Separating
    - Extracting
    - Plotting
'''


# Numpy lib
import numpy as np
# Plot lib
import matplotlib.pyplot as plt
from matplotlib import cm
# Enumerator library
from enum import Enum
# Import utilities
from utils import bipolar, add, flatten_list, transpose, copy_dict, find_values

import pickle
import copy

# Plotting events characteristics
class event_color(Enum):
    R = [1,0,0]
    L = [0,1,1]
    O = [1,1,0]

class event_linestyle(Enum):
    FS = '-'
    HS = '-'
    FO = '--'
    TO = '--'
    O = '--'
    

# =============================================================================
# Control fields
# =============================================================================
def is_field(_td, _fields, verbose = False):
    '''
    This function checks whether fields are in a dict.
    
    Parameters
    ----------
    _td : dict / list of dict
        dict of trial data.
    _fields : str / list of str
        Fields in the trial data dict.
    verbose : bool, optional
        Describe what's happening in the code. The default is False.

    Returns
    -------
    return_val : bool
        Return whether fields are in the trial data dict or not.

    '''
    
    return_val = True
    
    # check dict input variable
    if type(_td) is dict:
        _td = [_td]
        
    if type(_td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _fields must be a list of strings!')
    
    # Flatten list of fields
    _fields = flatten_list(_fields)
    
    for idx, td_tmp in enumerate(_td):
        for field in _fields:
            if field not in td_tmp.keys():
                return_val = False
                if verbose:
                    print('Field {} not in dict #{}'.format(field, idx))
    
    return return_val

def td_subfield(td, subfield):
    '''
    This function selects a hidden field in the td structure.

    Parameters
    ----------
    td : dict / list of dict
        dict of trial data.
    subfield : str / list of str
        Fields in the trial data dict.

    Returns
    -------
    subfield_value : everything
        The value in the pointed subfield

    '''
    
    if type(td) is not dict:
        raise Exception('ERROR: td must be a dict! It is a "{}"'.format(type(td)))
        
    if type(subfield) is not str:
        raise Exception('ERROR: subfield must be a str! It is a "{}"'.format(type(subfield)))
    
    layers = subfield.split('/')
    subfield_value = td
    for layer in layers:
        subfield_value = subfield_value[layer]
    
    return subfield_value

# =============================================================================
# Removing
# =============================================================================
def remove_fields(_td, _field, exact_field = False, inplace = True):
    '''
    This function removes fields from a dict.

    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data.
    _field : str / list of str
        Fields to remove.
    exact_field : bool, optional
        Look for the exact field name in the dict. The default is False.
    inplace : bool, optional
        Perform operaiton on the input data dict. The default is False.

    Returns
    -------
    td : dict/list of dict
        trial data dict with added items

    '''
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_field) is str:
        _field = [_field]
        
    if type(_field) is not list:
        raise Exception('ERROR: _field must be a list of strings!')
    
    for td_tmp in td:
        td_copy = td_tmp.copy()
        for iStr in _field:
            any_del = False
            for iFld in td_copy.keys():
                if exact_field:
                    if iStr == iFld:
                        del td_tmp[iFld]
                        any_del = True
                else:
                    if iStr in iFld:
                        del td_tmp[iFld]
                        any_del = True
            if not any_del:
                print('Field {} not found. I could not be removed...'.format(iStr))
    
    if input_dict:
        td = td[0]
        
    if not inplace:
        return td


def remove_all_fields_but(_td, _field, exact_field = False, inplace = True):
    '''
    This function removes all fields from a dict but the one selected.

    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data.
    _field : str / list of str
        Field to keep.
    exact_field : bool, optional
        Look for the exact field name in the dict. The default is False.
    inplace : bool, optional
        Perform operaiton on the input data dict. The default is False.

    Returns
    -------
    td : dict/list of dict
        trial data dict with added items

    '''
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_field) is str:
        _field = [_field]
        
    if type(_field) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    for td_tmp in td:
        td_copy = td_tmp.copy()
        for field in td_copy.keys():
            for field_name in _field:
                del_field = True
                if exact_field:
                    if field_name == field:
                        del_field = False
                        break
                else:
                    if field_name in field:
                        del_field = False
                        break
            if del_field == True:
                del td_tmp[field]
    
    if input_dict:
        td = td[0]
        
    if not inplace:
        return td


# =============================================================================
# Adding
# =============================================================================
def add_params(_td, params, **kwargs):
    '''
    This function adds parameters to the trial data dictionary

    Parameters
    ----------
    _td : dict / list of dict
        dict of trial data.
    params : dict
        Params of the data. See below for examples
    data_struct : str, optional
        Type of _td struct in input. The default value is 'flat'.
        Options:
            flat (all data available on the first layer of the dict)
            layer (data separated among inside layers of the dict)
        
    Returns
    -------
    td : dict / list of dict.
        dict of trial data.
        
    Examples for params in input:
    params = {'folder' : 'FOLDER_FIELD',
              'file' : 'FILE_FIELD',
              'data':{'EMG': {'signals':['SIGNAL_1','...','SIGNAL_N'], 'Fs': 'FS_FIELD', 'time': 'TIME_FIELD'},
                      'LFP': {'signals':['SIGNAL_1','...','SIGNAL_N'], 'Fs': 'FS_FIELD', 'time': 'TIME_FIELD'},
                      'KIN': {'signals':['SIGNAL_1','...','SIGNAL_N'], 'Fs': 'FS_FIELD', 'time': 'TIME_FIELD'}}}
    
    params = {'folder' : 'FOLDER_FIELD',
              'file' : 'FILE_FIELD',
              'data': {'Data':{'signals':['SIGNAL_1','...','SIGNAL_N'], 'Fs': 'FS_FIELD', 'time': 'TIME_FIELD'}}}
    '''

    data_struct = 'flat'
    inplace = True
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'data_struct':
            data_struct = value
        elif key == 'inplace':
            inplace = value
    
    if data_struct not in ['flat','layer']:
        raise Exception('ERROR: data_struct nor "flat", or "layer". It is: {}'.filename(data_struct))
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
    
    # Loop over the trials
    for td_tmp in td:
        # Params initiation
        if 'params' not in td_tmp.keys():
            signals_2_use = ['params']
            td_tmp['params'] = dict()
            td_tmp['params']['data'] = dict()
            td_tmp['params']['event'] = dict()
        else:
            signals_2_use = set(td_tmp.keys())
            params = td_tmp['params']
        
        params_c = copy.deepcopy(params)
        # Check input variables
        for key,val in params_c.items():
            # key = 'EMG'; val = params[key];
            if key in ['folder','file']:
                td_tmp['params'][key] = td_tmp[val]
            
            elif key in ['data']:
                for ke,va in val.items():
                    td_tmp['params']['data'][ke] = dict()
                    for k,v in va.items():
                        # k = 'signals'; v = va[k];
                        if k in ['signals','time']:
                            td_tmp['params']['data'][ke][k] = v
                            if type(v) is list:
                                signals_2_use.extend(v)
                                if data_struct == 'layer': # place data on the main layer
                                    for el in v:
                                        td_tmp[el] = td_tmp[ke][el]
                            elif type(v) is str:
                                signals_2_use.append(v)
                                if data_struct == 'layer': # place data on the main layer
                                    td_tmp[v] = td_tmp[ke][v]
                            else:
                                raise Exception('ERROR: Value "{}" in params is nor str or list! It is {}...'.format(k,v))
                        elif k in ['fs']:
                            if data_struct == 'layer':
                                val2take = td_tmp[ke][v]
                                if type(val2take) is np.ndarray and val2take.size == 1:
                                    val2take = val2take[0]
                                td_tmp['params']['data'][ke][k] = val2take
                            elif data_struct == 'flat':
                                val2take = td_tmp[v]
                                if type(val2take) is np.ndarray and val2take.size == 1:
                                    val2take = val2take[0]
                                td_tmp['params']['data'][ke][k] = val2take
                        else:
                            raise Exception('ERROR: Value in params dict "{}" is not "signal", "time", or "fs"! It is {}...'.format(k,v))
                    
                    # Check existance of 'fs' and 'time' data info
                    keys_in_dict = set(td_tmp['params']['data'][ke].keys())
                    if 'fs' not in keys_in_dict and 'time' not in keys_in_dict:
                        print('WARNING: neither time or fs are available for "{}"'.format(ke))
                    elif 'fs' not in keys_in_dict and 'time' in keys_in_dict:
                        td_tmp['params']['data'][ke]['fs'] = 1/np.diff(td_tmp[td_tmp['params']['data'][ke]['time']][:2])[0]
                    elif 'fs' in keys_in_dict and 'time' not in keys_in_dict:
                        fs = td_tmp['params']['data'][ke]['fs']
                        sign_len = np.max(td_tmp[td_tmp['params']['data'][ke]['signals'][0]].shape)
                        td_tmp['params']['data'][ke]['time'] = ke + '_time'
                        td_tmp[ke + '_time'] = np.linspace(0,sign_len/fs, sign_len)
                        signals_2_use.append(ke + '_time')
            
            elif key in ['event']:
                for ke,va in val.items():
                    td_tmp['params']['event'][ke] = dict()
                    for k,v in va.items():
                        # k = 'signals'; v = va[k];
                        if k in ['signals']:
                            td_tmp['params']['event'][ke][k] = v
                            if type(v) is list:
                                signals_2_use.extend(v)
                                if data_struct == 'layer': # place data on the main layer
                                    for el in v:
                                        td_tmp[el] = td_tmp[ke][el]
                            elif type(v) is str:
                                signals_2_use.append(v)
                                if data_struct == 'layer': # place data on the main layer
                                    td_tmp[v] = td_tmp[ke][v]
                            else:
                                raise Exception('ERROR: Value "{}" in params is nor str or list! It is {}...'.format(k,v))
                        elif k in ['kind']:
                            td_tmp['params']['event'][ke][k] = v
                        else:
                            raise Exception('ERROR: Value in params dict "{}" is not "signal", "time", or "fs"! It is {}...'.format(k,v))
    
    remove_all_fields_but(td,flatten_list(set(signals_2_use)),exact_field = True, inplace = True)
    
    if input_dict:
        td = td[0]
        
    if not inplace:
        return td

# =============================================================================
# Combining
# =============================================================================
def combine_fields(_td, _fields, **kwargs):
    '''
    This function combines the fields in one dictionary

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
        
    _fields : list of str, len (2)
        Two fields to combine.
        
    method : str, optional
        Method for combining the arrays. The default is subtract.
        
    remove_fields : bool, optional
        Remove the fields before returning the dataset. The default is True.
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''

    method = 'subtract'
    remove_selected_fields = True
    save_to_params = False
    inplace = True

    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'method':
            method = value
        elif key == 'remove_fields':
            remove_selected_fields = value
        elif key == 'save_to_params':
            save_to_params = True
            save_to_params_field = value
        elif key == 'inplace':
            inplace = value

    if inplace:
        td = _td
    else:
        td = copy_dict(_td)

    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # check _fields input variable        
    if type(_fields) is not list:
        raise Exception('ERROR: _fields must be a list!')
    
    if type(_fields[0]) is not list:
        _fields = [_fields]
    
    for field in _fields:
        if len(field) != 2:
            raise Exception('ERROR: lists in _fields must cointain max 2 strings!')
    
    if method not in ['subtract','multuply','divide','add']:
        raise Exception('ERROR: specified method has not been implemented!')
    
    if save_to_params:
        subfield = td_subfield(td[0],save_to_params_field)
        if 'signals' not in set(subfield.keys()):
            raise Exception('ERROR: field "signals" does not exist in "{}"'.format(save_to_params_field))
    
    for td_tmp in td:
        signals_name = []
        for field in _fields:
            if len(field[0]) != len(field[1]):
                raise Exception('ERROR: the 2 arrays must have the same length!')
            
            if method == 'subtract':
                signal_name = '{}-{}'.format(field[0],field[1])
                td_tmp[signal_name] = bipolar(td_tmp[field[0]],td_tmp[field[1]])
            elif method == 'add':
                signal_name = '{}+{}'.format(field[0],field[1])
                td_tmp[signal_name] = add(td_tmp[field[0]],td_tmp[field[1]])
            elif method == 'multiply':
                raise Exception('Method must be implemented!')
            elif method == 'divide':
                raise Exception('Method must be implemented!')
            signals_name.append(signal_name)
            
        if save_to_params:
            subfield = td_subfield(td_tmp,save_to_params_field)
            if remove_selected_fields:
                subfield['signals'] = signals_name
            else:
                subfield['signals'].extend(signals_name)
    
    if remove_selected_fields:
        remove_fields(td,flatten_list(_fields), exact_field = True, inplace = inplace)
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td
  

def join_fields(_td, _fields, **kwargs):
    '''
    This function joins the fields in multiple dictionaries. It outputs only 
    one dictionary.

    Parameters
    ----------
    _td : list of dict
        List of trial data.
        
    _fields : list of str, len (2)
        Two fields to combine.

    Returns
    -------
    td_out : dict
        Trial data with fields joined together.

    '''

    # Check dict input variable
    if type(_td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # check _fields input variable      
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _fields must be a list!')
    
    if not is_field(_td,_fields):
        raise Exception('ERROR: _fields are missing from td!')
    
    # Set output variable
    td_out = copy_dict(_td[0])
    n_td = len(td_out[_fields[0]])
    remove_all_fields_but(td_out, _fields, inplace = True)
    
    for td_tmp in _td[1:]:
        for field in _fields:
            td_out[field] = np.concatenate((td_out[field], td_tmp[field]))
        n_td += len(td_tmp[field])
           
    # Check fields length
    for field in _fields:
        if len(td_out[field]) != n_td:
            raise Exception('ERROR: length in fields is different from total length!')
    
    if 'params' in _td[0].keys():
        td_out['params'] = copy_dict(_td[0]['params'])
        
    return td_out

    
def combine_dicts(td_tuple, inplace = True):
    """        
    This function combines N dicts. In case the dictionaries share some fields,
    the first dict dominates.
    
    Parameters
    ----------
    td_tuple : tuple of dict / tuple of list of dict
        Dictionaries to combine
    inplace : string, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td_out : dict/list of dict
        trial data variable

    """
    
    if type(td_tuple) is not tuple:
        raise Exception('ERROR: td_tuple in not a tuple! You inputed a "{}".'.format(type(td_tuple)))
    
    if len(td_tuple)<2:
        raise Exception('ERROR: td_tuple contains less than 2 dicts! It contains "{}".'.format(len(td_tuple)))
        
    if inplace:
        td = td_tuple[0]
    else:
        td = copy_dict(td_tuple[0])
    
    td_list = list(td_tuple[1:])
    
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    for iEl, el in enumerate(td_list):
        if type(el) is dict:
            td_list[iEl] = [el]
    
    # Check that tds have the same dimension
    for el in td_list:
        if len(td) != len(el):
            raise Exception('ERROR: tds have different dimension!')
    
    for el in td_list:
        for td1_el, td2_el in zip(td, el):
            for k,v in td2_el.items():
                if k not in set(td1_el.keys()):
                    td1_el[k] = v
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td

# =============================================================================
# Separating
# =============================================================================
def separate_fields(_td, _fields, **kwargs):
    '''
    This function combines the fields in one dictionary

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    _fields : list of str
        Two fields to combine.
    new_names : list of str
        Method for combining the arrays. The default is subtract.
    inplace : bool, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''

    new_names = None
    inplace = True
    save_to_params = False
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'new_names':
            new_names = value
        elif key == 'inplace':
            inplace = value
        elif key == 'save_to_params':
            save_to_params = True
            save_to_params_field = value

    if inplace:
        td = _td
    else:
        td = copy_dict(_td)

    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    if not is_field(td,_fields):
        raise Exception('ERROR: some _fileds are not in td!')
    
    # check _fields input variable    
    if type(_fields) is str:
        _fields = [_fields]
    if type(_fields) is not list:
        raise Exception('ERROR: _fields must be a list!')
    
    # Check new names
    if new_names == None:
        new_names = []
        for td_tmp in td:
            for field in _fields:
                field_size = transpose(np.array(td_tmp[field]),'column').shape[1]
                new_names.append([field+'_'+str(iS) for iS in range(field_size)])
    else:
        if type(new_names) is not list:
            raise Exception('ERROR: new_names must be a list!')
        if type(new_names[0]) is str:
            new_names = [new_names]
        
        for td_tmp in td:
            for iN, (names, field) in enumerate(zip(new_names,_fields)):
                field_size = transpose(np.array(td_tmp[field]),'column').shape[1]
                if len(names) != field_size:
                    raise Exception('ERROR: new_names[{}] has different length compared to the dimension of field "{}"!'.format(iN, field))
    
    if save_to_params:
        subfield = td_subfield(td[0],save_to_params_field)
        if 'signals' not in set(subfield.keys()):
            raise Exception('ERROR: field "signals" does not exist in "{}"'.format(save_to_params_field))
    
    for td_tmp in td:
        for names, field in zip(new_names, _fields):
            for iN, name in enumerate(names):
                td_tmp[name] = transpose(np.array(td_tmp[field]),'column')[:,iN]
            
        if save_to_params:
            subfield = td_subfield(td_tmp,save_to_params_field)
            for field in _fields:
                subfield['signals'].remove(field)
            subfield['signals'].extend(flatten_list(new_names))
    
    remove_fields(td,flatten_list(_fields), exact_field = True, inplace = inplace)
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td
  


# =============================================================================
# Extracting
# =============================================================================
def get_field(_td, _signals, save_signals_name = False):
    '''
    This function get fields from the trial data and place them in a another dict
    
    Parameters
    ----------
    _td : dict
        dict from which we collect the fields.
    _signals : str/list of str
        Fields to collect from the dict.
    save_signals_name : bool
        Add a field containing the name of the fileds. The default is False.
        
    Returns
    -------
    td_out : dict
        New dict containing the selected fields.

    '''
    td = copy_dict(_td)
    td_out = []
    
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
    
    if type(_signals) == str:
        print('Signals input must be a list. You inputes a string --> converting to list...')
        _signals = [_signals]
    
    # Check that _signals are in the dictionary
    if not is_field(td,_signals):
        raise Exception('ERROR: Selected signals are not in the dict')
    
    # Loop over the trials
    for td_tmp in td:
        td_out_tmp = dict()
        # List of signals name
        signals_name = []
        # Loop over the signals
        for sgl in _signals:
            if type(sgl) == list:
                signal_name = '{} - {}'.format(sgl[0],sgl[1])
                signals_name.append(signal_name)
                td_out_tmp[signal_name] = np.array(td_tmp[sgl[0]]) - np.array(td_tmp[sgl[1]])
            else:
                signal_name = sgl
                signals_name.append(signal_name)
                td_out_tmp[signal_name] = np.array(td_tmp[sgl])
        
        if save_signals_name:
            td_out_tmp['params'] = dict()
            td_out_tmp['params']['signals'] = signals_name
        
        td_out.append(td_out_tmp)
    
    if input_dict:
        td_out = td_out[0]
        
    return td_out


def extract_dicts(td, fields, **kwargs):
    """        
    This function extracts a dictionary containing only the fields in input.
    
    Parameters
    ----------
    td : dict/list of dict
        dict of the trial data
    fields : str/list of str
        Fields to combine in the new dict
    keep_name : str, optional
        Keep the name from the dict in the new keys (e.g. LFP: BIP1 --> LFP_BIP1). The default is True.
    all_layers : str, optional
        Perform operation on all the dictionary layers. The default is False.

    Returns
    -------
    td_out : dict/list of dict
        trial data variable

    """
    
    keep_name = True
    all_layers = False
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'keep_name':
            keep_name = value
        elif key == 'all_layers':
            all_layers = value
    
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
    
    if type(fields) is str:
        fields = [fields]
    
    td_out = []
    for td_tmp in td:
        dict_tmp = dict()
        for field in fields:
            if type(td_tmp[field]) is dict:
                dict_new = td_tmp[field]
                if keep_name:
                    dict2add = dict()
                    for k,v in dict_new.items():
                        dict2add[field+'_'+k] = v
                else:
                    dict2add = dict_new
                combine_dicts((dict_tmp, dict2add), inplace = True)
            else:
                dict_tmp[field] = td_tmp[field]
        
        if all_layers:
            all_layers_dict = dict_tmp.copy()
            for k,v in all_layers_dict.items():
                if type(v) is dict:
                    in_layer_dict = extract_dicts(all_layers_dict, k, keep_name = keep_name, all_layers = all_layers)
                    combine_dicts((dict_tmp, in_layer_dict), inplace = True)
                    remove_fields(dict_tmp,k,exact_field = True, inplace = True)
        
        td_out.append(dict_tmp)
        
        
    if input_dict:
        td_out = td_out[0]
        
    return td_out

# =============================================================================
# Plotting
# =============================================================================
def td_plot(td, y, **kwargs):
    '''
    This function plots signals from the td dict.

    Parameters
    ----------
    td : dict
        Trial data.
        
    y : str/list of str
        Signals to plot.
        
    x : str/list of str, optional
        Signal to use for x axis.
        
    subplot : tuple, optional
        Structure of the suvplots in the figure.
        
    title : str, optional
        Title of the plot.
        
    xlim : tuple, optional
        x min and max.
        
    ylim : tuple, optional
        y min and max.
        
    sharex : bool, optional
        Share the x axis among the plotted signals.
        
    sharey : bool, optional
        Share the y axis among the plotted signals.
        
    save : bool, optional
        Flag for saving the figure.

    Example
    ----------
        td_plot(td,['LFP_BIP7','LFP_BIP9'], events = ['RFS','LFS'], subplot = (2,1))

    '''
    
    # Input variables
    x = None
    subplot = ()
    title = None
    
    axs_external = []
    
    grid_plot = False
    maximise = False
    
    ylim = None
    xlim = None
    ylabel = None
    xlabel = None
    sharex = False
    sharey = False
    
    x_ticks = 'on'
    
    style = '-'
    colours = None
    colours_range = None
    save_figure = False
    save_format = 'pdf'
    
    kind = 'contour'
    
    events = None
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'events':
            events = value
        elif key == 'subplot':
            subplot = value
        elif key == 'x':
            x = value
        elif key == 'title':
            title = value
        elif key == 'grid_plot':
            grid_plot = value
        elif key == 'axs':
            axs_external = value
        elif key == 'ylabel':
            ylabel = value
        elif key == 'xlabel':
            xlabel = value
        elif key == 'ylim':
            ylim = value
        elif key == 'xlim':
            xlim = value
        elif key == 'sharex':
            sharex = value
        elif key == 'sharey':
            sharey = value
        elif key == 'colours':
            colours = value
        elif key == 'colours_range':
            colours_range = value
        elif key == 'style':
            style = value
        elif key == 'x_ticks':
            x_ticks = value
        elif key == 'maximise':
            maximise = value
        elif key == 'kind':
            kind = value
        elif key == 'save':
            save_figure = True
            save_name = value
        elif key == 'save_format':
            save_format = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))

    # Check input variables
    if type(td) is not dict:
        raise Exception('ERROR: td type must be a dict! It is a {}'.format(type(td)))
        
    if type(y) is str:
        y = [y]
        
    # Check whether _signals elements are in _td
    if not(is_field(td, y)):
        raise Exception('ERROR: _signals must be in _td!')
    
    # Saving paramters
    if type(save_format) is str:
        save_format = [save_format]
    if type(save_format) is not list:
        raise Exception('ERROR: type(save_format) is not a list!')
    for save_form in save_format:
        if save_form not in ['pickle','svg','png','pdf']:
            raise Exception('ERROR: save_format can only be: "pickle","svg","png" or "pdf"! You inputed "{}".'.format(save_form))
        
    if x_ticks not in ['on','off']:
        raise Exception('ERROR: x_ticks can be either "on" or "off"!')
        
    if kind not in ['contour','imshow']:
        raise Exception('ERROR: kind can be either "contour" or "imshow"! You inputed "{}".'.format(kind))
        
    ##########################################################################    
    # Create figure
    if len(axs_external) == 0:
        # Check subplot dimension
        if not(subplot):
            subplot = (len(y),1)
        fig, axs = plt.subplots(nrows = subplot[0], ncols = subplot[1], sharex=sharex, sharey=sharey)
    else:
        axs = axs_external
        if axs.ndim == 1:
            subplot = (axs.shape[0],1)
        else:
            subplot = axs.shape
    # Convert to flatten list
    if type(axs) is np.ndarray:
        if axs.ndim == 1:
            axs_list = axs.tolist()
        else:
            axs_list = []
            for sub_x in range(subplot[0]):
                for sub_y in range(subplot[1]):
                    axs_list.append(axs[sub_x][sub_y])
    else:
        axs_list = [axs]
    
    # Find bottom plots
    axes_pos = np.array([ax._position.bounds[1] for ax in axs_list])
    bottom_idx = np.where( (axes_pos - min(axes_pos)<0.01 ))[0]
    
    if len(axs_list) != len(y):
        raise Exception('ERROR: axs and y have different length: axs: {} != y: {}'.format(len(axs_list),len(y)))
    
    # Add title to figure
    if title != None:
        plt.suptitle(title)
    
    if maximise:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    
    # if type(axs).__module__ != np.__name__:
    #     axs  = [axs]
    
    # Prepare plotting variables
    # Get max number of signals in a plot
    max_signal_n = 1
    for signal in y:
        if type(signal) == list and len(signal) > max_signal_n:
            max_signal_n = len(signal)
        
    # Check x axis
    if x == None:
        x_list = [None] * len(y)
    elif x != None and type(x) == str:
        x_list = [x] * len(y)
    elif x != None and type(x) == list:
        if len(x) == len(y):
            x_list = x
        else:
            raise Exception('ERROR: x list have different length from y list!')
    else:
        raise Exception('ERROR: x must be None, str ot list of str!')
        
    # Check x label
    if xlabel == None:
        xlab_list = [None] * len(y)
    elif xlabel != None and type(xlabel) == str:
        xlab_list = [xlabel] * len(y)
    elif xlabel != None and type(xlabel) == list:
        if len(xlabel) == len(y):
            xlab_list = xlabel
        else:
            raise Exception('ERROR: xlabel list have different length from y list!')
    else:
        raise Exception('ERROR: xlabel must be None, str ot list of str!')
        
    # Check y label
    if ylabel == None:
        ylab_list = [None] * len(y)
    elif ylabel != None and type(ylabel) == str:
        ylab_list = [ylabel] * len(y)
    elif ylabel != None and type(ylabel) == list:
        if len(ylabel) == len(y):
            ylab_list = ylabel
        else:
            raise Exception('ERROR: ylabel list have different length from y list!')
    else:
        raise Exception('ERROR: ylabel must be None, str ot list of str!')
        
    # Check axis colours
    if colours == None:
        col_list = [[None] * max_signal_n] * len(y)
    elif type(colours) == str or type(colours) == cm.colors.ListedColormap or type(colours) == cm.colors.LinearSegmentedColormap:
        col_list = [[colours] * max_signal_n] * len(y)
    elif type(colours) == list and type(colours[0]) != list:
        col_list = [colours] * len(y)
    elif type(colours) == list and type(colours[0]) == list:
        col_list = colours 
        
    # Check axis colours
    if colours_range == None:
        col_range_list = [[None] * max_signal_n] * len(y)
    elif type(colours_range) == list and type(colours_range[0]) != list:
        if len(colours_range) == 2:
            col_range_list = [colours_range] * len(y)
        else:
            raise Exception('ERROR: colours_range can only have length == 2')
    elif type(colours_range) == list and type(colours_range[0]) == list:
        col_range_list = colours_range 
        
    # Check axis style
    if style == None:
        style_list = [[None] * max_signal_n] * len(y)
    elif type(style) == str:
        style_list = [[style] * max_signal_n] * len(y)
    elif type(style) == list and type(style[0]) != list:
        style_list = [style] * len(y)
    elif type(style) == list and type(style[0]) == list:
        style_list = style
        
    # Check x lim
    if type(xlim) == tuple or type(xlim) == np.ndarray:
        xlim_list = [xlim] * len(y)
    else:
        xlim_list = [None] * len(y)
        
    # Check y lim
    if type(ylim) == tuple or type(ylim) == np.ndarray:
        ylim_list = [ylim] * len(y)
    else:
        ylim_list = [None] * len(y)
    
    # Check x ticks
    if x_ticks == 'on':
        x_ticks_list = ['on'] * len(y)
    else:
        x_ticks_list = ['off'] * len(y)

    # Check events
    if events != None:
        if type(events) is str:
            events_list = [[events]] * len(y)
        elif type(events) is list and type(events[0]) is not list:
            events_list =  [events] * len(y)
        elif type(events) is list and type(events[0]) is list:
            events_list = events
            if len(events_list) != len(y):
                raise Exception('ERROR: events list of list length {} != y length {}!'.format(len(events),len(y)))
        
        if not(is_field(td, flatten_list(events))):
            raise Exception('ERROR: all events must be in td!')
    else:
        events_list = [[]] * len(y)

    # Plot
    for iPlt, (x, signal, xlim, ylim, xlab, ylab, col, style, x_tick, event_list, col_range, ax) in \
        enumerate(zip(x_list, y, xlim_list, ylim_list, xlab_list, ylab_list, col_list, style_list, x_ticks_list, events_list, col_range_list, axs_list)):
        # break
        # Set plot title
        if type(signal) is list:
            signal_name = ' + '.join(signal)
        else:
            signal_name = signal
            signal = [signal]
        
        # Set title
        ax.set_title(signal_name)
        
        # Set plot grid
        ax.grid(grid_plot)
        
        # Set plot labels
        if xlabel != None:
            ax.set_xlabel(xlabel)
        if ylabel != None:
            ax.set_ylabel(ylabel)
                
        # Get dimension of the singal in input
        signal_dim = np.array(td[signal[0]]).ndim
            
        # Set axes limits
        if signal_dim == 1:
            if type(ylim) == tuple:
                ylim_tmp = ylim
                ax.set_ylim(ylim_tmp)
            else:
                if ax.get_ylim() == (0,1): # It is a new figure
                    ylim_tmp = [+np.inf, -np.inf] # Start to set up the yaxis
                else:
                    ylim_tmp = list(ax.get_ylim()) # Use the existing yaxis
                # Extract ylim from the signals
                for sig in signal:
                    sig_tmp = transpose(td[sig],'column')
                    if np.min(sig_tmp) < ylim_tmp[0]:
                        ylim_tmp[0] = np.min(td[sig])
                    if np.max(sig_tmp) > ylim_tmp[1]:
                        ylim_tmp[1] = np.max(td[sig])
                ax.set_ylim(ylim_tmp)
            if type(xlim) == tuple:
                ax.set_xlim(xlim)
        elif signal_dim == 2:
            if type(xlim) == np.ndarray and type(ylim) == np.ndarray:
                X, Y = np.meshgrid(xlim, ylim)
            else:
                X, Y = np.meshgrid(np.arange(td[signal[0]].shape[0]), np.arange(td[signal[0]].shape[1]))
        else:
            raise Exception('ERROR: td_plot implemented only for signals up to 2d. You gave a {}d signal...'.format(signal_dim))
        
        # Plot signal
        for sig, c, s in zip(signal, col, style):
            # break
            # Get dimension of the singal
            signal_dim = np.array(td[sig]).ndim
            if signal_dim == 1:
                if x == None:
                    if c == None:
                        if s == None:
                            ax.plot(td[sig])
                        else:
                            ax.plot(td[sig], linestyle = s)
                    else:
                        if s == None:
                            ax.plot(td[sig], color = c)
                        else:
                            ax.plot(td[sig], color = c, linestyle = s)
                else:
                    if c == None:
                        if s == None:
                            ax.plot(td[x], td[sig])
                        else:
                            ax.plot(td[x], td[sig], linestyle = s)
                    else:
                        if s == None:
                            ax.plot(td[x], td[sig], color = c)
                        else:
                            ax.plot(td[x], td[sig], color = c, linestyle = s)
                            
            elif signal_dim == 2:
                if c == None:
                    if X.shape == td[sig].shape:
                        if kind == 'imshow':
                            if col_range == [None]:
                                cs = ax.imshow(td[sig], extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto')
                            else:
                                cs = ax.imshow(td[sig], extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', vmin = col_range[0], vmax = col_range[1])
                        else:
                            if col_range == [None]:
                                cs = ax.contourf(X, Y, td[sig])
                            else:
                                cs = ax.contourf(X, Y, td[sig], vmin = col_range[0], vmax = col_range[1])
                    else:
                        if kind == 'imshow':
                            if col_range == [None]:
                                cs = ax.imshow(np.flip(td[sig].T,0), extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto')
                            else:
                                cs = ax.imshow(np.flip(td[sig].T,0), extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', vmin = col_range[0], vmax = col_range[1])
                        else:
                            if col_range == [None]:
                                cs = ax.contourf(X, Y, td[sig].T)
                            else:
                                cs = ax.contourf(X, Y, td[sig].T, vmin = col_range[0], vmax = col_range[1])
                else:
                    if X.shape == td[sig].shape:
                        if kind == 'imshow':
                            if col_range == [None]:
                                cs = ax.imshow(td[sig].T, extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', cmap=c)
                            else:
                                cs = ax.imshow(td[sig].T, extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', cmap=c, vmin = col_range[0], vmax = col_range[1])
                        else:
                            if col_range == [None]:
                                cs = ax.contourf(X, Y, td[sig], cmap=c)
                            else:
                                cs = ax.contourf(X, Y, td[sig], cmap=c, vmin = col_range[0], vmax = col_range[1])
                    else:
                        if kind == 'imshow':
                            if col_range == [None]:
                                cs = ax.imshow(np.flip(td[sig].T,0),  extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', cmap=c)
                            else:
                                cs = ax.imshow(np.flip(td[sig].T,0),  extent = [X.min(), X.max(), Y.min(), Y.max()], aspect = 'auto', cmap=c, vmin = col_range[0], vmax = col_range[1])
                        else:
                            if col_range == [None]:
                                cs = ax.contourf(X, Y, td[sig].T, cmap=c)
                            else:
                                cs = ax.contourf(X, Y, td[sig].T, cmap=c, vmin = col_range[0], vmax = col_range[1])
                                
                # norm_cm = cm.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
                # sm = plt.cm.ScalarMappable(norm=norm_cm, cmap = cs.cmap)
                # sm.set_array([])
                # fig.colorbar(sm, ax = ax, ticks=cs.levels)
                if col_range == [None]:
                    fig.colorbar(cs, ax = ax)
                else:
                    m = plt.cm.ScalarMappable(cmap=c)
                    m.set_array(td[sig])
                    m.set_clim(col_range[0], col_range[1])
                    fig.colorbar(m, ax = ax, boundaries=np.linspace(col_range[0], col_range[1], 10))
        
        # Manage ticks
        if x_tick == 'off' and iPlt not in bottom_idx:
            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        
        # Plot gait events
        if len(event_list) != 0:
            for event in event_list:
                if 'R' in event:
                    col = event_color.R.value
                elif 'L' in event:
                    col = event_color.L.value
                else:
                    col = event_color.O.value
                
                if 'FS' in event or 'HS' in event:
                    line_style = event_linestyle.FS.value
                elif 'FO' in event or 'TO' in event:
                    line_style = event_linestyle.FO.value
                else:
                    line_style = event_linestyle.O.value
                
                if len(td[event]) == len(td[signal[0]]):
                    events_2_plot = find_values(td[event], value = 0.9, method = 'bigger')
                    if signal_dim == 1 and x != None:
                        events_2_plot = td[x][events_2_plot]
                    elif signal_dim == 2:
                        events_2_plot = X[0,events_2_plot]
                else:
                    events_2_plot = td[event]
                
                for ev in events_2_plot:
                    ax.axvline(x = ev, ymin = 0, ymax = 1, color = col + [0.7], linestyle = line_style)
    
    # Set tight plot
    plt.tight_layout()
        
    if subplot[1] == 1:
        hspace=0.7
        if x_ticks == 'off' or sharex == True:
            hspace-=0.3
    else:
        hspace=0.3
        if x_ticks == 'off' or sharex == True:
            hspace-=0.1
        
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=hspace)
    # left: the left side of the subplots of the figure
    # right: the right side of the subplots of the figure
    # bottom: the bottom of the subplots of the figure
    # top: the top of the subplots of the figure
    # wspace: the amount of width reserved for space between subplots,
    #         expressed as a fraction of the average axis width
    # hspace: the amount of height reserved for space between subplots,
    #         expressed as a fraction of the average axis height
    
    if save_figure:
        for form in save_format:
            if form == 'pickle':
                pickle.dump(fig, open(save_name +'.pickle', 'wb'))
            elif form == 'svg':
                fig.savefig(save_name + '.svg', bbox_inches='tight')
            elif form == 'png':
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                fig.savefig(save_name + '.png', bbox_inches='tight')
            elif form == 'pdf':
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                fig.savefig(save_name + '.pdf', bbox_inches='tight')
            else:
                raise Exception('ERROR: wrong save format assigned!')
    
    if len(axs_external) == 0:
        return fig, axs
    else:
        return axs


if __name__ == '__main__':
    td_test = {'test1': np.arange(10), 'test2': [1,2,3,4,5,6]}
    td_test2 = {'test1': np.arange(20), 'test3': [1,2,3,4,5,6]}
    td_test_list = [td_test,td_test]
    
    # Test combine_dicts
    td_comb = combine_dicts((td_test, td_test2), inplace = False)
    if set(['test1','test2','test3']) == td_comb.keys() and (td_comb['test1'] == td_test['test1']).all():
        print('Test combine_dicts passed!')
    else:
        raise Exception('ERROR: Test combine_dicts NOT passed!')
    
    # Test is_field
    if is_field(td_test,'test1') and is_field(td_test,'test3',False) == False and is_field(td_test_list,'test1') and is_field(td_test_list,'test3',False) == False:
        print('Test is_field passed!')
    else:
        raise Exception('ERROR: Test is_field NOT passed!')
    
    # Test get_field
    td_res = get_field(td_test, ['test1'])
    if 'test1' in td_res:
        print('Test get_field passed!')
    else:
        raise Exception('ERROR: Test get_field NOT passed!')
        
    # Test add_fields
    # td_new = add_field_from_dict(td_test,{'test3': [1,3,4]}, inplace = False)
    # if 'test3' in td_new.keys():
    #     print('Test add_field_from_dict passed!')
    # else:
    #     raise Exception('ERROR: Test add_field_from_dict NOT passed!')
    
    td_new1 = remove_all_fields_but(td_test, 'test1', exact_field = True, inplace = False)
    td_new2 = remove_all_fields_but(td_test, 'test', exact_field = False, inplace = False)
    if set(td_new1.keys()) != set(['test1']) or set(td_new2.keys()) != set(['test1','test2']):
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test remove_all_fields_but passed!')
    
    # Test combine_fields
    td_test_combine = {'test1': np.arange(10), 'test2': np.arange(10)}
    td_new = combine_fields(td_test_combine,['test1','test2'],method = 'subtract', inplace = False)
    if 'test1-test2' in td_new.keys() and (td_new['test1-test2'] - np.zeros(10) < 0.1).all():
        print('Test combine_fields passed!')
    else:
        raise Exception('ERROR: Test combine_fields NOT passed!')
        
    del td_test, td_test2, td_test_list, td_comb, td_res, td_new, td_new1, td_new2, td_test_combine
    print('All implemented tests passed!')
    
    