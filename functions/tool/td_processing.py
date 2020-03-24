#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:29:47 2020

@author: raschell
"""

# Numpy lib
import numpy as np

# Loading lib
from loading_data import load_data_from_folder

# Td utils lib
from td_utils import *

from utils import group_fields

# Processing libs
from processing import artefacts_removal, convert_points_to_target_vector, get_epochs

from power_estimation import moving_pmtm

from filters import butter_bandpass_filtfilt as bpff
from filters import butter_lowpass_filtfilt as lpff
from filters import butter_highpass_filtfilt as hpff


# =============================================================================
# Cleaning functions
# =============================================================================

def combine_epochs(td_1, td_2):
    '''
    This function combines the good epochs from 2 dict using the logic AND.

    Parameters
    ----------
    td_1 : dict / list of dict
        trial data with epoch field.
    td_2 : dict / list of dict
        trial data with epoch field.

    Returns
    -------
    td_epoch : dict / list of dict
        Dict aving as only field 'epochs', a binary array.

    '''
    td_epoch = []
    if len(td_1) != len(td_2):
        raise Exception('ERROR: artefacts and segments trials data have different length!')
    for art, seg in zip(td_1, td_2):
        if len(art['epochs']) != len(seg['epochs']):
            raise Exception('ERROR: artefacts[epochs] and segments [epochs] have different length!')
        td_epoch.append({'epochs': np.logical_and(np.array(art['epochs']), np.array(seg['epochs'])).astype('int')})
    
    return td_epoch

def segment_data(_td, td_epoch, Fs, **kwargs):
    '''
    This function segments the trial data in several blocks

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    td_epoch : dict / list of dict
        Array of good epochs
    Fs : int / float
        Frequency of the signal
    invert_epoch: str, optional
        set whether invert the epoch values or not. The default value is False. 

    Returns
    -------
    td_out : dict / list of dict
        Trial data.

    '''
    
    invert_epoch = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'invert_epoch':
            invert_epoch = value
    
    td_out = []
    
    td = _td.copy()
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    
    if len(td) != len(td_epoch):
        raise Exception('ERROR: td and td_segment must have the same length!')
    
    if type(Fs) is str:
        if '/' in Fs:
            Fs = td_subfield(td[0],Fs)['fs']
        else:
            Fs = td[0][Fs]
    
    for td_tmp, td_epo in zip(td, td_epoch):
        if invert_epoch:
            td_epo['epochs'] = np.logical_not(td_epo['epochs']).astype('int')
        epochs = get_epochs(td_epo['epochs'], Fs)
        for epoch in epochs:
            td_out_tmp = td_tmp.copy()
            for k,v in td_out_tmp.items():
                if k != 'params':
                    td_out_tmp[k] = np.array(v)[epoch]
                elif 'time' in k:
                    td_out_tmp[k] = np.array(v)[epoch] - np.array(v)[epoch][0]
            # Append new dict
            td_out.append(td_out_tmp)
    
    return td_out

def identify_artefacts(_td, **kwargs):
    '''
    This function identifies the artefacts in the trial data dict(s)

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    fields : list of str
        Fields over which finding the artefacts
    Fs : int
        Sample frequency of the signals
    method : str, optional
        Method for removing the artefacts. The default is amplitude.
    signal_n : int
        Number of signal on which to look for artefacts
    
    Returns
    -------
    td_artefacts : dict / list of dict
        Dictionary of the identified artefacts.

    '''
    
    method = None
    signal_n = None
    threshold = None
    fields = None
    Fs = None
    params = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'fields':
            fields = value
        elif key == 'Fs':
            Fs = value
        elif key == 'method':
            method = value
        elif key == 'signal_n':
            signal_n = value
        elif key == 'threshold':
            threshold = value
        elif key == 'params':
            params = value
        else:
            print('WARNING: key "{}" not recognised by the identify_artefacts function...'.format(key))

    # Get a temporary copy of td
    td = _td.copy()

    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    if params != None:
        subset = td_subfield(td[0],params)
        if 'signals' in subset and 'fs' in subset:
            fields = subset['signals']
            Fs = subset['fs']
        else:
            raise Exception('ERROR: signals and/or fs are missing from the input subset "{}"'.format(subset))
        
    # check string input variable
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if Fs == None:
        raise Exception('ERROR: fields must be assigned!')
        
    if type(fields) is str:
        fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
            
    if signal_n == None:
        print('WARNING: Number of signals with artefacts not specified. Setting to len(fields)-1 ...')
        signal_n = len(fields)-1
    
    if method == None:
        method = 'amplitude'
        print('WARNING: Method for removing artifacts not specified: Selectd method is {}'.format(method))
        
    td_artefacts = []
    for td_tmp in td:        
        data = group_fields(td_tmp,fields)
        td_artefacts.append({'epochs':artefacts_removal(data, Fs, method, signal_n, threshold)})
    
    return td_artefacts

def convert_fields_to_numeric_array(_td, _fields, _vector_target_field, inplace = True):
    '''
    This function converts the content of fields in a vector

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    _fields : str / list of str
        Fields from which collect the data to convert.
    _vector_target_field : str
        Field with the name of the array in td to compare to _fields vectors
    inplace : bool, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''
    
    if inplace:
        td = _td
    else:
        td = _td.copy()

    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # check string input variable
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    if type(_vector_target_field) is not str:
        raise Exception('ERROR: _vector_target_field must be a string!')
    
    # Check that _signals are in the dictionary
    if not is_field(td,_fields):
        raise Exception('ERROR: Selected fields are not in the dict')
    
    for td_tmp in td:
        vector_compare = np.array(td_tmp[_vector_target_field])
        for field in _fields:
            points = np.array(td_tmp[field])
            td_tmp[field] = convert_points_to_target_vector(points, vector_compare)
        
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td
    

# =============================================================================
# Preprocessing functions
# =============================================================================
    
def compute_multitaper(_td, **kwargs):
    '''
    This function compute the multitapers spectal analysis for the td signals

    Parameters
    ----------
    _td : dict / list of dict
        trial data with epoch field.
    **kwargs : dict
        Additional information for computing multitaper.

    Returns
    -------
    td : dict / list of dict
        trial data with signals analysed based on the multitaper parameters.

    '''
    # Multitaper information
    window_size_sec = 0.25 # in seconds
    window_step_sec = 0.01 # in seconds
    freq_min = 10
    freq_max = 100
    NW = 4
    inplace = True
    verbose = False

    # Check input variables
    for key,value in kwargs.items():
        if key == 'wind_size':
            window_size_sec = value
        elif key == 'wind_step':
            window_step_sec = value
        elif key == 'freq_start':
            freq_min = value
        elif key == 'freq_stop':
            freq_max = value
        elif key == 'NW':
            NW = value
        elif key == 'inplace':
            inplace = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    # check dict input variable
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # Compute number of tapers
    tapers_n = np.floor(2*NW)-1
    # Get frequency range
    freq_range = [freq_min , freq_max]
    
    for td_tmp in td:
        Fs = td_tmp['params']['Fs']
        
        # Get window's info in samples
        window_size_smp = round(window_size_sec * Fs)
        window_step_smp = round(window_step_sec * Fs)
        
        for iSgl, signal in enumerate(td_tmp['params']['signals']):
            if verbose:
                print('Processing signal {}/{}'.format(iSgl+1, len(td_tmp['signal_names'])))
            td_tmp[signal], sfreqs, stimes = moving_pmtm(td_tmp[signal], Fs, window_size_smp, window_step_smp, freq_range, NW=NW, NFFT=None, verbose=verbose)
        
        # Update frequency info
        td_tmp['params']['Fs'] = 1/(stimes[1] - stimes[0])
        
        # Re-map target dataset
        if 'target' in set(td_tmp.keys()):
            target_new = np.zeros(stimes.shape)
            for ev in np.unique(td_tmp['target'])[1:]:
                target_new += np.histogram(np.where(td_tmp['target'] == ev), bins = stimes.shape[0], range = (0, len(td_tmp['target'])))[0]
            if (target_new > np.unique(td_tmp['target'])[-1]).any():
                raise Exception('WARNING: Multiple classes are falling in the mutitaping binning!')
            td_tmp['target'] = target_new
            
        td_tmp[td_tmp['params']['time']] = stimes
        td_tmp['params']['freq'] = 'freq'
        td_tmp['freq'] = sfreqs
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td

def compute_filter(_td, **kwargs):
    # Filtering information
    kind = None
    f_min = None
    f_max = None
    order = 5
    inplace = True
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'kind':
            kind = value
        elif key == 'f_min':
            f_min = value
        elif key == 'f_max':
            f_max = value
        elif key == 'order':
            order = value
        elif key == 'verbose':
            verbose = value
        elif key == 'inplace':
            inplace = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    # check dict input variable
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    for td_tmp in td:
        Fs = td_tmp['params']['Fs']
        
        for iSgl, signal in enumerate(td_tmp['params']['signals']):
            if verbose:
                print('Filtering signal {}/{}'.format(iSgl+1, len(td_tmp['signal_names'])))
            if kind == 'bandpass':
                td_tmp[signal] = bpff(data = td_tmp[signal], lowcut = f_min, highcut = f_max, fs = Fs, order=order)
            elif kind == 'lowpass':
                td_tmp[signal] = lpff(data = td_tmp[signal], lowcut = f_min, fs = Fs, order=order)
            elif kind == 'highpass':
                td_tmp[signal] = hpff(data = td_tmp[signal], highcut = f_max, fs = Fs, order=order)
            else:
                raise Exception('ERROR: wrong kind of filter applied! Kind given is : {}'.format(kind))
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td

# =============================================================================
# Pipelines
# =============================================================================

def load_pipeline(**kwargs):
    '''
    This funtion loads and organises data from 

    Parameters
    ----------
    data_path : str / list of str
        Path(s) of the folder(s) with the data.
    data_files : int / list of int
        Number of the files to load.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict
        Trial data organised based on input requests.

    '''
        
    # Input variables
    load_data_from_folder_dict = None
    target_load_dict = None
    remove_fields_dict = None
    remove_all_fields_but_dict = None
    convert_fields_to_numeric_array_dict = None
    params = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'load':
            load_data_from_folder_dict = value
        elif key == 'trigger_file':
            target_load_dict = value
        elif key == 'remove_fields':
            remove_fields_dict = value
        elif key == 'remove_all_fields_but':
            remove_all_fields_but_dict = value
        elif key == 'convert_fields_to_numeric_array':
            convert_fields_to_numeric_array_dict = value
        elif key == 'params':
            params = value
            
    
    # Load data
    if load_data_from_folder_dict == None:
        raise Exception('ERROR: Loading function not assigned!')
    td = load_data_from_folder(folders = load_data_from_folder_dict['path'],**load_data_from_folder_dict)
    td = extract_dicts(td, set(td[0].keys()), keep_name = True, all_layers = True)
    
    if remove_fields_dict != None:
        remove_fields(td, remove_fields_dict['fields'], exact_field = False, inplace = True)
    
    if remove_all_fields_but_dict != None:
        remove_all_fields_but(td, remove_all_fields_but_dict['fields'], exact_field = True, inplace = True)
    
    if target_load_dict != None:
        td_target = load_data_from_folder(folders = target_load_dict['path'],**target_load_dict)
        td_target = extract_dicts(td_target, set(td_target[0].keys()), keep_name = True, all_layers = True)
        if 'fields' in target_load_dict.keys():
            remove_all_fields_but(td_target,target_load_dict['fields'],False,True)
    
        # Combine target data with the predictor data
        combine_dicts(td, td_target, inplace = True)
    
    if convert_fields_to_numeric_array_dict != None:
        convert_fields_to_numeric_array(td, _fields = convert_fields_to_numeric_array_dict['fields'], 
                                        _vector_target_field = convert_fields_to_numeric_array_dict['target_vector'],
                                        inplace = True)    
    
    if params != None:
        add_params(td, params)
    
    return td
    

def cleaning_pipeline(td, **kwargs):
    '''
    This function cleans the dataset by:
        combining signals
        removing artefacts
        segmenting the data

    Parameters
    ----------
    td : dict / list of dict
        Trial data.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict / list of dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    combine_fields_dict = None
    remove_artefacts_dict = None
    add_segmentation_dict = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'combine_fields':
            combine_fields_dict = value
        elif key == 'remove_artefacts':
            remove_artefacts_dict = value
        elif key == 'add_segmentation':
            add_segmentation_dict = value
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    if combine_fields_dict != None:
        combine_fields(td, combine_fields_dict['fields'], **combine_fields_dict)
        
    if remove_artefacts_dict != None:
        td_artefacts = identify_artefacts(td, **remove_artefacts_dict)
        
    if add_segmentation_dict != None:
        td_segment = add_segmentation_dict['epochs']
    
    # Combine artefacts and segmentation
    if remove_artefacts_dict != None and add_segmentation_dict != None:
        td_epoch = combine_epochs(td_artefacts, td_segment)
    elif remove_artefacts_dict == None and add_segmentation_dict != None:
        td_epoch = td_segment
    elif remove_artefacts_dict != None and add_segmentation_dict == None:
        td_epoch = td_artefacts
    
    if remove_artefacts_dict != None or add_segmentation_dict != None:
        td = segment_data(td, td_epoch, remove_artefacts_dict['params'], invert_epoch = True)
    
    return td


def preprocess_pipeline(td, **kwargs):
    '''
    This function preprocess the dataset.

    Parameters
    ----------
    td : dict / list of dict
        Trial data.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict / list of dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    operations = []
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'filter':
            operations.append((key,value))
        elif key == 'multitaper':
            operations.append((key,value))
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    for operation in operations:
        if operation[0] == 'filter':
            compute_filter(td, **operation[1])
        elif operation[0] == 'multitaper':
            compute_multitaper(td, **operation[1])
    
    return td


if __name__ == '__main__':
    # Test convert_fields_to_numeric_array
    td_test_cf2n = {'test1': np.arange(10), 'id1': [1,3,5], 'id2': [2,4]}
    td_new = convert_fields_to_numeric_array(td_test_cf2n, ['id1','id2'], 'test1', remove_selected_fields = True)
    if (td_new['target'] - np.array([0,0,1,2,1,2,1,0,0,0])>0.1).any() or is_field(td_new,['id1','id2']):
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test find_first passed!')
    
    
    