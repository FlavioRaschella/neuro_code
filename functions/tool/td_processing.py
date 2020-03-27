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
from td_utils import is_field, combine_fields, combine_dicts, td_subfield, extract_dicts
from td_utils import remove_fields, remove_all_fields_but, add_params

from utils import group_fields, convert_list_to_array, copy_dict, flatten_list, find_values, find_first

# Processing libs
from processing import artefacts_removal, convert_points_to_target_vector, get_epochs

from power_estimation import moving_pmtm

from filters import sgolay_filter, downsample_signal
from filters import butter_bandpass_filtfilt as bpff
from filters import butter_lowpass_filtfilt as lpff
from filters import butter_highpass_filtfilt as hpff

# Decoding utils
from decoder_utils import extract_features


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

def segment_data(_td, td_epoch, fs, **kwargs):
    '''
    This function segments the trial data in several blocks

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    td_epoch : dict / list of dict
        Array of good epochs
    fs : int / float
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
    
    td = copy_dict(_td)
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    
    if len(td) != len(td_epoch):
        raise Exception('ERROR: td and td_segment must have the same length!')
    
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
    
    for td_tmp, td_epo in zip(td, td_epoch):
        if invert_epoch:
            td_epo['epochs'] = np.logical_not(td_epo['epochs']).astype('int')
        epochs = get_epochs(td_epo['epochs'], fs)
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
    fs : int
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
    fs = None
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'fields':
            fields = value
        elif key == 'fs':
            fs = value
        elif key == 'method':
            method = value
        elif key == 'signal_n':
            signal_n = value
        elif key == 'threshold':
            threshold = value
        else:
            print('WARNING: key "{}" not recognised by the identify_artefacts function...'.format(key))

    # Get a temporary copy of td
    td = copy_dict(_td)

    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    # check string input variable
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if fs == None:
        raise Exception('ERROR: fields must be assigned!')
    
    # Check fs 
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
    
    # Check fields        
    if type(fields) is str and '/' in fields:
        fields = td_subfield(td[0],fields)['signals']
    if type(fields) is str:
        fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
        
    if signal_n == None:
        print('WARNING: Number of signals with artefacts not specified. Setting to len(fields)-1 ...')
        signal_n = len(fields)-1
    
    if method == None:
        method = 'amplitude'
        print('WARNING: Method for removing artifacts not specified: Selectd method is {}'.format(method))
        
    td_artefacts = []
    for td_tmp in td:        
        data = group_fields(td_tmp,fields)
        td_artefacts.append({'epochs':artefacts_removal(data, fs, method, signal_n, threshold)})
    
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
        td = copy_dict(_td)

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

def downsample(_td, **kwargs):
    
    fields = None
    fields_string = ''
    field_time = None
    fs = None
    fs_down = None
    inplace = True
    verbose = False
    adjust_target = False

    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'fields':
            fields = value
        elif key == 'fs':
            fs = value
        elif key == 'field_time':
            field_time = value
        elif key == 'fs_down':
            fs_down = value
        elif key == 'inplace':
            inplace = value
        elif key == 'verbose':
            verbose = value
        elif key == 'adjust_target':
            adjust_target = True
            adjust_target_field = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    # Check input values
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if fs == None:
        raise Exception('ERROR: fs must be assigned!')
    if fs_down == None:
        raise Exception('ERROR: fs_down must be assigned!')

    # Check fields        
    if type(fields) is str:
        if '/' in fields:
            fields_string = fields
            fields = td_subfield(td[0],fields)['signals']
        else:
            fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
        
    if not is_field(td,fields):
        raise Exception('ERROR: fields is not in td!')
    
    # Check fs 
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
            
    # Check fs_down 
    if type(fs_down) is str:
        fs_down = td[0][fs_down]
    elif type(fs_down) is int or type(fs_down) is int:
        pass
    else:
        raise Exception('ERROR: fs_down must be a int / float / str!')

    # Check field_time 
    if field_time != None and type(field_time) is str:
        if '/' in field_time:
            field_time = td_subfield(td[0],field_time)['time']
        
    if not is_field(td,field_time):
        raise Exception('ERROR: field_time is not in td!')

    if adjust_target and field_time != None:
        if type(adjust_target_field) is str:
            if '/' in adjust_target_field:
                target_fields = []
                target_subfields = td_subfield(td[0],adjust_target_field)
                for event, value in target_subfields.items():
                    target_fields.append(value['signals'])
            else:
                target_fields = [adjust_target_field]
            target_fields = flatten_list(target_fields)
        if type(target_fields) is not list:
            raise Exception('ERROR: fields must be a list of strings!')
            
        if not is_field(td,target_fields):
            raise Exception('ERROR: target_fields is not in td!')
    else:
        raise Exception('ERROR: "adjust_target" works only if "field_time" is not None!')

    # Downsample signals
    for iTd, td_tmp in enumerate(td):
        # Downsample target dataset as well
        if adjust_target:
            time_down,_ = downsample_signal(td_tmp[field_time], fs, fs_down)
            for iEv, event in enumerate(target_fields):
                if verbose: print('Downsampling event {}/{} in td {}/{}'.format(iEv+1,len(target_fields),iTd+1,len(td)))
                target_new = np.zeros(len(time_down),int)
                events_n = np.unique(td_tmp[event])[1:]
                for ev in events_n:
                    event_time = td_tmp[field_time][find_values(td_tmp[event],ev,'equal')]
                    event_idx = []
                    for ev_idx in event_time:
                        event_idx.append(find_first(ev_idx,time_down))
                
                    target_new[event_idx] = ev
                # Create new event signal
                td_tmp[event] = target_new
        
        # Downsample time field
        if field_time != None:
            td_tmp[field_time],_ = downsample_signal(td_tmp[field_time], fs, fs_down)

        # Downsample fields
        for iFld, field in enumerate(fields):
            if verbose: print('Downsampling field {}/{} in td {}/{}'.format(iFld+1,len(fields),iTd+1,len(td)))
            td_tmp[field],fs_new = downsample_signal(td_tmp[field], fs, fs_down)
            
            # Update frequency info in params
            if fields_string != '':
                subfields = td_subfield(td[0],fields_string)
                subfields['fs'] = fs_new
            
    if input_dict:
        td = td[0]
    
    return td

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
    fields = None
    fs = None
    fs_string = ''
    window_size_sec = 0.25 # in seconds
    window_step_sec = 0.01 # in seconds
    freq_min = 10
    freq_max = 100
    NW = 4
    inplace = True
    verbose = False
    adjust_target = False

    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'wind_size':
            window_size_sec = value
        elif key == 'wind_step':
            window_step_sec = value
        elif key == 'freq_start':
            freq_min = value
        elif key == 'freq_stop':
            freq_max = value
        elif key == 'nw':
            NW = value
        elif key == 'fs':
            fs = value
        elif key == 'fields':
            fields = value
        elif key == 'inplace':
            inplace = value
        elif key == 'adjust_target':
            adjust_target = True
            adjust_target_field = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    # Check input values
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if fs == None:
        raise Exception('ERROR: fs must be assigned!')
    
    # Check fs 
    if type(fs) is str:
        if '/' in fs:
            fs_string = fs
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
            
    # Check fields        
    if type(fields) is str and '/' in fields:
        fields = td_subfield(td[0],fields)['signals']
    if type(fields) is str:
        fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
        
    if adjust_target:
        if type(adjust_target_field) is str:
            if '/' in adjust_target_field:
                target_fields = []
                target_subfields = td_subfield(td[0],adjust_target_field)
                for event, value in target_subfields.items():
                    target_fields.append(value['signals'])
            else:
                target_fields = [adjust_target_field]
            target_fields = flatten_list(target_fields)
        if type(target_fields) is not list:
            raise Exception('ERROR: fields must be a list of strings!')
            
        if not is_field(td,target_fields):
            raise Exception('ERROR: target_fields is not in td!')
        
    # Compute number of tapers
    tapers_n = (np.floor(2*NW)-1).astype('int')
    if verbose:
        print('# of tapes used = {}'.format(tapers_n))
    # Get frequency range
    freq_range = [freq_min , freq_max]
    
    for td_tmp in td:
        # Get window's info in samples
        window_size_smp = round(window_size_sec * fs)
        window_step_smp = round(window_step_sec * fs)
        
        for iFld, field in enumerate(fields):
            if verbose:
                print('Processing signal {}/{}'.format(iFld+1, len(fields)))
            td_tmp[field], sfreqs, stimes = moving_pmtm(td_tmp[field], fs, window_size_smp, window_step_smp, freq_range, NW=NW, NFFT=None, verbose=verbose)
        
        # Update frequency info
        if fs_string == '': # Not using params
            td_tmp['freq'] = sfreqs
            td_tmp['time'] = stimes
        else: # Using params
            subfields = td_subfield(td[0],fs_string)
            subfields['fs'] = 1/(stimes[1] - stimes[0])
            td_tmp[subfields['time']] = stimes
            subfields['freq'] = 'freq'
            td_tmp[subfields['freq']] = sfreqs
        
        # Re-map target dataset
        if adjust_target:
            for event in target_fields:
                target_new = np.zeros(stimes.shape)
                for ev in np.unique(td_tmp[event])[1:]:
                    target_new += np.histogram(np.where(td_tmp[event] == ev), bins = stimes.shape[0], range = (0, len(td_tmp[event])))[0]
                if (target_new > np.unique(td_tmp[event])[-1]).any():
                    raise Exception('ERROR: Multiple classes are falling in the mutitaping binning!')
                td_tmp[event] = target_new
    
    if input_dict:
        td = td[0]
    
    return td

def compute_filter(_td, **kwargs):
    # Filtering information
    fields = None
    fs = None
    kind = None
    f_min = None
    f_max = None
    win_len = None
    order = 5
    override_fields = True
    save_to_params = False
    inplace = True
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'kind':
            kind = value
        elif key == 'fields':
            fields = value
        elif key == 'fs':
            fs = value
        elif key == 'f_min':
            f_min = value
        elif key == 'f_max':
            f_max = value
        elif key == 'win_len':
            win_len = value
        elif key == 'order':
            order = value
        elif key == 'inplace':
            inplace = value
        elif key == 'verbose':
            verbose = value
        elif key == 'override_fields':
            override_fields = value
        elif key == 'save_to_params':
            save_to_params = True
            save_to_params_field = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if kind not in ['bandpass','lowpass','highpass','sgolay']:
        raise Exception('ERROR: kind specified "{}" is not implemented!'.format(kind))
    
    if inplace:
        td = _td
    else:
        td = copy_dict(_td)
    
    # Check input values
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if fs == None:
        raise Exception('ERROR: fs must be assigned!')
    
    # Check fs 
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
            
    # Check fields        
    if type(fields) is str and '/' in fields:
        fields = td_subfield(td[0],fields)['signals']
    if type(fields) is str:
        fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
    
    if win_len != None:
        if type(win_len) is str:
            win_len = fs*int(win_len[:win_len.lower().find('fs')])            
            if win_len%2 == 0:
                win_len += 1
                print('WARNING: 1 added to win_len because win_len is even. It must be odd for sgolay!')
    
    # Compute filters
    for td_tmp in td:
        signals_name = []
        for iFld, field in enumerate(fields):
            if verbose:
                print('Filtering signal {}/{}'.format(iFld+1, len(fields)))
            if kind == 'bandpass':
                if not override_fields:
                    signal_name = field + '_bp_{}_{}'.format(f_min,f_max)
                else:
                    signal_name = field
                td_tmp[signal_name] = bpff(data = td_tmp[field], lowcut = f_min, highcut = f_max, fs = fs, order=order)
            
            elif kind == 'lowpass':
                if not override_fields:
                    signal_name = field + '_lp_{}'.format(f_min)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = lpff(data = td_tmp[field], lowcut = f_min, fs = fs, order=order)
            
            elif kind == 'highpass':
                if not override_fields:
                    signal_name = field + '_hp_{}'.format(f_max)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = hpff(data = td_tmp[field], highcut = f_max, fs = fs, order=order)
            
            elif kind == 'sgolay':
                if not override_fields:
                    signal_name = field + '_sg_{}'.format(win_len)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = sgolay_filter(data = td_tmp[field], win_len = win_len, order=order)
            
            else:
                raise Exception('ERROR: wrong kind of filter applied! Kind given is : {}'.format(kind))
            signals_name.append(signal_name)
    
        if not override_fields and save_to_params:
            subfield = td_subfield(td_tmp,save_to_params_field)
            subfield['signals'].extend(signals_name)
    
    if input_dict:
        td = td[0]
    
    return td
        


# =============================================================================
# Features
# =============================================================================
        
def extract_features_instant(td, **kwargs):
        
    fields = None
    event_fields = None
    fs = None
    time_n = [10]
    feature_win_sec = [0.5]
    dead_win_sec = [0.02]
    no_event_sec = [1]
    verbose = True
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'fields':
            fields = value
        elif key == 'fs':
            fs = value
        elif key == 'event_fields':
            event_fields = value
        elif key == 'time_n':
            time_n = value
        elif key == 'feature_win_sec':
            feature_win_sec = value
        elif key == 'dead_win_sec':
            dead_win_sec = value
        elif key == 'no_event_sec':
            no_event_sec = value
        elif key == 'verbose':
            verbose = value    
    
    # Check input variables
    if type(td) is dict:
        td = [td]
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    if fields == None:
        raise Exception('ERROR: fields must be assigned!')
    if event_fields == None:
        raise Exception('ERROR: event_fields must be assigned!')
    if fs == None:
        raise Exception('ERROR: fs must be assigned!')
                
    # Check fields        
    if type(fields) is str:
        if '/' in fields:
            fields = td_subfield(td[0],fields)['signals']
        else:
            fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
        
    if not is_field(td,fields):
        raise Exception('ERROR: fields is not in td!')
        
    # Check event_fields        
    if type(event_fields) is str:
        if '/' in event_fields:
            target_fields = []
            event_subfields = td_subfield(td[0],event_fields)
            for event, value in event_subfields.items():
                target_fields.append(value['signals'])
        else:
            event_fields = [event_fields]
        target_fields = flatten_list(target_fields)
    if type(target_fields) is not list:
        raise Exception('ERROR: event_fields must be a list of strings!')
            
    if not is_field(td,target_fields):
        raise Exception('ERROR: target_fields is not in td!')
    
    # Check fs 
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            fs = td[0][fs]
    
    if type(time_n) is int or type(time_n) is float:
        time_n = [int(time_n)]
    if type(time_n) is not list:
        raise Exception('ERROR: time_n must be a list of int!')  
        
    if type(feature_win_sec) is int or type(feature_win_sec) is float:
        feature_win_sec = [feature_win_sec]
    if type(feature_win_sec) is not list:
        raise Exception('ERROR: feature_win_sec must be a list of int!')
        
    if type(dead_win_sec) is int or type(dead_win_sec) is float:
        dead_win_sec = [dead_win_sec]
    if type(dead_win_sec) is not list:
        raise Exception('ERROR: feature_win_sec must be a list of int!')
        
    if type(no_event_sec) is int or type(no_event_sec) is float:
        no_event_sec = [no_event_sec]
    if type(no_event_sec) is not list:
        raise Exception('ERROR: feature_win_sec must be a list of int!')
    
    # Extract the features
    td_features = []
    
    feature_win_smp = np.round(np.array(feature_win_sec) * fs).astype('int')
    dead_win_smp = np.round(np.array(dead_win_sec) * fs).astype('int')
    no_event_smp = np.round(np.array(no_event_sec) * fs).astype('int')

    # Loop over the features
    loop_n = len(time_n) * len(feature_win_smp) * len(dead_win_smp) * len(no_event_smp)
    loop_count = 1
    for time in time_n:
        for feature_win in feature_win_smp:
            for dead_win in dead_win_smp:
                for no_event in no_event_smp:
                    print('Extract features loop {}/{}'.format(loop_count,loop_n))
                    # Collect signals for features in one 2darray
                    features = []
                    labels = []
                    
                    # Actual number of no_event to use for each td
                    no_event_td = np.round(no_event/len(td)).astype('int')
                    
                    for td_tmp in td:
                        data_fields = [td_tmp[field] for field in fields]
                        
                        # Check dimension in data
                        data_fields_features = []
                        for feat in data_fields:
                            if feat.ndim == 1:
                                data_fields_features.append(1)
                            else:
                                data_fields_features.append(feat.shape[1])
                        if (np.diff(data_fields_features)>0.1).any():
                            raise Exception('ERROR: Data in fields have different [1] dimension!')
                        else:
                            data_fields_features = data_fields_features[0]
                        
                        # Concatenate data to feed into the feature extraction
                        data = convert_list_to_array(data_fields, axis = 1)
                        event_data = group_fields(td_tmp, target_fields)
                        
                        # Get features
                        features_n = len(fields) * time * data_fields_features
                        template = np.round(np.linspace(0,-feature_win,time)).astype('int')

                        features_tmp, labels_tmp = extract_features(data, event_data, dead_win, no_event_td, features_n, template)
                        
                        # Append features
                        features.append(features_tmp)
                        labels.append(labels_tmp)
                    
                    features = np.concatenate(features, axis = 0)
                    labels = np.concatenate(labels, axis = 0)
                    params = {'sample_rate': fs,
                              'events_name': target_fields + ['no_event'],
                              'time_n': time,
                              'channels_n': len(fields),
                              'channel_dim': data_fields_features,
                              'no_event_smp': no_event,
                              'dead_win_smp': dead_win,
                              'feature_win_smp': feature_win,
                              'features_n': features_n,
                              'template': template}
                    
                    td_features.append({'features': features,'labels': labels, 'params': params})
                    
                    # Improve loop counter
                    loop_count += 1
                    
    return td_features

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
        key = key.lower()
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
        key = key.lower()
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
        td = segment_data(td, td_epoch, remove_artefacts_dict['fields'], invert_epoch = True)
    
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
        key = key.lower()
        if key == 'filter':
            operations.append((key,value))
        elif key == 'multitaper':
            operations.append((key,value))
        elif key == 'downsample':
            operations.append((key,value))
        else:
            print('WARNING: key "{}" not recognised by the preprocess pipeline...'.format(key))
    
    for operation in operations:
        if operation[0] == 'filter':
            td = compute_filter(td, **operation[1])
        elif operation[0] == 'multitaper':
            td = compute_multitaper(td, **operation[1])
        elif operation[0] == 'downsample':
            td = downsample(td, **operation[1])
    
    return td


def features_pipeline(td, **kwargs):
    '''
    This function extracts the features from the dataset.

    Parameters
    ----------
    td : dict / list of dict
        Trial data.
    **kwargs : dict
        Additional information for extracting the features

    Returns
    -------
    td : dict / list of dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    operations = []
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'event_instant':
            operations.append((key,value))
        elif key == 'event_surround':
            raise Exception('ERROR: this feature extraction process has not been implemented yet')
            operations.append((key,value))
        else:
            raise Exception('ERROR: key "{}" not recognised by the feature pipeline...'.format(key))
    
    for operation in operations:
        if operation[0] == 'event_instant':
            td_features = extract_features_instant(td, **operation[1])
        elif operation[0] == 'event_surround':
            pass
    
    return td_features


if __name__ == '__main__':
    # Test convert_fields_to_numeric_array
    td_test_cf2n = {'test1': np.arange(10), 'id1': [1,3,5], 'id2': [2,4]}
    td_new = convert_fields_to_numeric_array(td_test_cf2n, ['id1','id2'], 'test1', remove_selected_fields = True)
    if (td_new['target'] - np.array([0,0,1,2,1,2,1,0,0,0])>0.1).any() or is_field(td_new,['id1','id2']):
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test find_first passed!')
    
    
    