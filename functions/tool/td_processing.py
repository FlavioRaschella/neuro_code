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
# Td utils libs
from td_utils import is_field, combine_fields, combine_dicts, td_subfield, extract_dicts
from td_utils import remove_fields, remove_all_fields_but, add_params
from utils import group_fields, convert_list_to_array, copy_dict, flatten_list, find_values, find_first, transpose
# Import processing libs
from processing import artefacts_removal, convert_points_to_target_vector, get_epochs, convert_time_samples
# Import power estimation libs
from power_estimation import moving_pmtm, moving_pmtm_trigger
# Import filer libs
from filters import sgolay_filter, downsample_signal, moving_average, envelope
from filters import butter_bandpass_filtfilt as bpff
from filters import butter_lowpass_filtfilt as lpff
from filters import butter_highpass_filtfilt as hpff
# Import decoding utils
from decoder_utils import extract_features
# Import plotting libs
import matplotlib.pyplot as plt

# =============================================================================
# Cleaning functions
# =============================================================================

def combine_epochs(td_1, td_2):
    '''
    This function combines the good epochs from 2 dict using the logic AND.

    Parameters
    ----------
    td_1 : dict / list of dict, len (n_td)
        Trial data with epoch field.
        
    td_2 : dict / list of dict, len (n_td)
        Trial data with epoch field.

    Returns
    -------
    td_epoch : dict / list of dict, len (n_td)
        Dict aving as only field 'epochs', a binary array.

    '''
    td_epoch = []
    if len(td_1) != len(td_2):
        raise Exception('ERROR: artefacts and segments trials data have different length!')
    for art, seg in zip(td_1, td_2):
        if len(art['epochs']) != len(seg['epochs']):
            raise Exception('ERROR: artefacts[epochs] and segments [epochs] have different length!')
        td_epoch.append({'epochs': np.logical_or(np.array(art['epochs']), np.array(seg['epochs'])).astype('int')})
    
    return td_epoch

def segment_data(_td, _td_epoch, fs, **kwargs):
    '''
    This function segments the trial data in several blocks.

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    td_epoch : dict / list of dict, len (n_td)
        Array of epochs to keep.
        
    fs : int / float
        Sampling frequency.
        
    invert_epoch: str, optional
        Set whether invert the epoch values or not. The default value is False. 

    Returns
    -------
    td_out : dict / list of dict, len (n_td)
        Trial data.

    '''
    # Input data
    invert_epoch = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'invert_epoch':
            invert_epoch = value
    
    td_out = []
    
    td = copy_dict(_td)
    td_epoch = copy_dict(_td_epoch)
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    
    if type(td_epoch) is dict:
        td_epoch = [td_epoch]
    
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
                if k != 'params' and 'time' not in k:
                    td_out_tmp[k] = np.array(v)[epoch]
                elif 'time' in k:
                    td_out_tmp[k] = np.array(v)[epoch] - np.array(v)[epoch][0]
            # Append new dict
            td_out.append(td_out_tmp)
    
    return td_out

def identify_artefacts(_td, **kwargs):
    '''
    This function identifies the artefacts for the signals in td.

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : list of str
        Fields over which finding the artefacts.
        
    fs : int
        Sampling frequency of the signals in fields.
        
    method : str, optional
        Method for removing the artefacts. The default is amplitude.
        
    signal_n : int, optional
        Number of signal on which to look for artefacts. The default is len(fields)-1.
    
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

def convert_fields_to_numeric_array(_td, fields, vector_target_field, kind = 'time', inplace = True):
    '''
    This function converts the points in a vector to a target vector.
    [1,4,6] --> [0 1 0 0 1 0 1]

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, , len (n_fields)
        Fields in td from which collect the data to convert. If Kind == 'time'
        the data are expected in a time format, otherwise in a sample format.
        
    vector_target_field : str
        Time array in td to compare to fields vectors.
        
    kind : str, optional
        Set whether data are in 'time' or 'samples' format. The default is 'time'.
        If 'samples', the Time array is converted using np.arange.
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.

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
    if type(fields) is str:
        fields = [fields]
        
    if type(fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    if type(vector_target_field) is not str:
        raise Exception('ERROR: vector_target_field must be a string!')
    
    # Check that _signals are in the dictionary
    if not is_field(td,fields):
        raise Exception('ERROR: Selected fields are not in the dict')
    
    if kind not in ['time','samples']:
        raise Exception('ERROR: kind can be either "time" or "samples". It is "{}"'.format(kind))
    
    for td_tmp in td:
        if kind == 'time':
            vector_compare = np.array(td_tmp[vector_target_field])
        else:
            vector_compare = np.arange(len(td_tmp[vector_target_field]))
        for field in fields:
            points = np.array(td_tmp[field])
            td_tmp[field] = convert_points_to_target_vector(points, vector_compare)
        
    if input_dict:
        td = td[0]
    
    return td
    

# =============================================================================
# Preprocessing functions
# =============================================================================

def downsample(_td, **kwargs):
    '''
    This function downsamples signals in td (e.g. td[fields]).

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data to convert.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    fs_down : int / float
        New sampling frequency used for downsampling.
        
    field_time : str, optional
        Fields in td containing the time information of the signals.
        It can either be one key in td or the path in td where to find the time 
        signal in td (e.g. params/data/data).
        
    adjust_target : str, optional
        Fields in td containing the target signals (binary signals).
        It can either be one key in td or the path in td where to find the fs 
        (e.g. params/data/data).
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        Trial data.

    '''
    fields = None
    fields_string = ''
    fs = None
    fs_down = None
    field_time = None
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
    if type(fs_down) is not int or type(fs_down) is not int:
        raise Exception('ERROR: fs_down must be an int or a float!')

    # Check field_time 
    if field_time != None and type(field_time) is str:
        if '/' in field_time:
            field_time = td_subfield(td[0],field_time)['time']
        
    if not is_field(td,field_time):
        raise Exception('ERROR: field_time is not in td!')

    if adjust_target:
        if field_time != None:
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
                subfields = td_subfield(td_tmp,fields_string)
                subfields['fs'] = fs_new
            
    if input_dict:
        td = td[0]
    
    return td

def compute_multitaper(_td, **kwargs):
    '''
    This function computes the multitapers spectal analysis for the signals in td (e.g. td[fields]).

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data to convert.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    win_size : int / float, optional
        Size of the sliding window for computing the pmtm. It is in seconds.
        The default is 0.25 seconds.
        
    win_step : int / float, optional
        Step of the sliding window for computing the pmtm. It is in seconds.
        The default is 0.01 seconds.
        
    NW : int, optional
        Time Half-Bandwidth Product for computing the pmtm. The default is 4.
        
    freq_min : int / float, optional
        Minimum frequency in the stored frequency band of the spectogram.
        The default is 10.
         
    freq_max : int / float, optional
        Maximum frequency in the stored frequency band of the spectogram.
        The default is 100.
        
    unity : str, optional
        Unity of the output computed power. It can be 'power' or 'db'. 
        The default is 'db'.
        
    adjust_target : str, optional
        Fields in td containing the target signals (binary signals).
        It can either be one key in td or the path in td where to find the 
        target signal (e.g. params/data/data).
        
    kind : str, optional
        Select the type of spectrogram computation: 'chronux' or 'milekovic'. 
        The default is 'chronux'.
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        trial data with signals analysed based on the multitaper parameters.

    '''
    # Multitaper information
    fields = None
    fs = None
    fs_string = ''
    window_size_sec = 0.25 # in seconds
    window_step_sec = 0.01 # in seconds
    norm = None
    NW = 4
    freq_min = 10
    freq_max = 100
    unit = 'db'
    adjust_target = False
    
    kind = 'chronux'
    
    inplace = True
    verbose = False

    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'win_size':
            window_size_sec = value
        elif key == 'win_step':
            window_step_sec = value
        elif key == 'freq_start':
            freq_min = value
        elif key == 'freq_stop':
            freq_max = value
        elif key == 'unit':
            unit = value
        elif key == 'norm':
            norm = value
        elif key == 'nw':
            NW = value
        elif key == 'fs':
            fs = value
        elif key == 'fields':
            fields = value
        elif key == 'inplace':
            inplace = value
        elif key == 'kind':
            kind = value
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
    
    # Get window's info in samples
    window_size_smp = round(window_size_sec * fs)
    window_step_smp = round(window_step_sec * fs)
    
    # Loop over the data trials
    for iTd, td_tmp in enumerate(td):
        if verbose:
            print('\nProcessing signals in td {}/{}'.format(iTd+1, len(td)))
        
        # Collect data
        data_fields = [td_tmp[field] for field in fields]
        data = convert_list_to_array(data_fields, axis = 1)
        # Compute pmtm
        mt_spectrogram, sfreqs, stimes = moving_pmtm(data, 
                        window_size_smp, window_step_smp, freq_range, kind = kind,
                        norm = norm, NW = NW, Fs = fs, unit = unit, verbose=verbose)
        for iFld, field in enumerate(fields):
            td_tmp[field] = mt_spectrogram[:,:,iFld]
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
                    bins = np.arange(int(window_size_smp/2),len(td_tmp[event])-int(window_size_smp/2)+1,window_step_smp)
                    # bins = np.arange(int(window_size_smp/2),len(td_tmp[event])-int(window_size_smp/2),window_step_smp)
                    target_new_tmp = np.histogram(np.where(td_tmp[event] == ev), bins = bins)[0]
                    if (find_values(td_tmp[event]) == len(td_tmp[event])-int(window_size_smp/2)+window_step_smp).any():
                        target_new_tmp = np.append(target_new_tmp,1)
                    else:
                        target_new_tmp = np.append(target_new_tmp,0)
                    
                    target_new += target_new_tmp
                if (target_new > np.unique(td_tmp[event])[-1]).any():
                    raise Exception('ERROR: Multiple classes are falling in the mutitaping binning!')
                td_tmp[event] = target_new
    
    if input_dict:
        td = td[0]
    
    return td

def compute_multitaper_trigger(_td, **kwargs):
    '''
    This function computes the multitapers spectal analysis for the signals in td (e.g. td[fields]).

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data to convert.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    events : str
        Name of the event field in td. The events must be in samples.
        
    win_size : int / float, optional
        Size of the sliding window for computing the pmtm. It is in seconds.
        The default is 0.25 seconds.
        
    win_step : int / float, optional
        Step of the sliding window for computing the pmtm. It is in seconds.
        The default is 0.01 seconds.
        
    pre_event : int, option
        Samples before the event to use for the pmtm. win_step/2 will be added
        to pre_event to account for the windowing. It is in Fs.
        
    post_event : int, option
        Samples after the event to use for the pmtm. win_step/2 will be added
        to pre_event to account for the windowing. It is in Fs.
        
    NW : int, optional
        Time Half-Bandwidth Product for computing the pmtm. The default is 4.
        
    freq_min : int / float, optional
        Minimum frequency in the stored frequency band of the spectogram.
        The default is 10.
         
    freq_max : int / float, optional
        Maximum frequency in the stored frequency band of the spectogram.
        The default is 100.
        
    unity : str, optional
        Unity of the output computed power. It can be 'power' or 'db'. 
        The default is 'db'.
        
    adjust_target : str, optional
        Fields in td containing the target signals (binary signals).
        It can either be one key in td or the path in td where to find the 
        target signal (e.g. params/data/data).
        
    kind : str, optional
        Select the type of spectrogram computation: 'chronux' or 'milekovic'. 
        The default is 'chronux'.
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        trial data with signals analysed based on the multitaper parameters.

    '''
    # Multitaper information
    fields = None
    fs = None
    fs_string = ''
    window_size_sec = 0.25 # in seconds
    window_step_sec = 0.01 # in seconds
    norm = None
    NW = 4
    freq_min = 10
    freq_max = 100
    unit = 'db'
    
    pre_event = None
    post_event = None
    
    kind = 'chronux'
    events = []
    
    inplace = True
    verbose = False

    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'win_size':
            window_size_sec = value
        elif key == 'win_step':
            window_step_sec = value
        elif key == 'events':
            events = value
        elif key == 'pre_event':
            pre_event = value
        elif key == 'post_event':
            post_event = value
        elif key == 'freq_start':
            freq_min = value
        elif key == 'freq_stop':
            freq_max = value
        elif key == 'unit':
            unit = value
        elif key == 'norm':
            norm = value
        elif key == 'nw':
            NW = value
        elif key == 'fs':
            fs = value
        elif key == 'fields':
            fields = value
        elif key == 'inplace':
            inplace = value
        elif key == 'kind':
            kind = value
        elif key == 'verbose':
            verbose = value
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

    if pre_event is None:   
        pre_event = fs
        
    if post_event is None:   
        post_event = fs
            
    # Check fields        
    if type(fields) is str and '/' in fields:
        fields = td_subfield(td[0],fields)['signals']
    if type(fields) is str:
        fields = [fields]
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
    
    if not is_field(td, events):
        raise Exception('ERROR: event field not in td!')
        
    # Compute number of tapers
    tapers_n = (np.floor(2*NW)-1).astype('int')
    if verbose:
        print('# of tapes used = {}'.format(tapers_n))
    # Get frequency range
    freq_range = [freq_min , freq_max]
    
    # Get window's info in samples
    window_size_smp = round(window_size_sec * fs)
    window_step_smp = round(window_step_sec * fs)
    
    # Loop over the data trials
    for iTd, td_tmp in enumerate(td):
        if verbose:
            print('\nProcessing signals in td {}/{}'.format(iTd+1, len(td)))
        
        # Collect data
        data_fields = [td_tmp[field] for field in fields]
        data = convert_list_to_array(data_fields, axis = 1)
        events_smp = find_values(td_tmp[events],1)
        # Compute pmtm
        mt_spectrogram, sfreqs, stimes = moving_pmtm_trigger(data, 
                        events_smp, window_size_smp, window_step_smp, freq_range, 
                        pre_event = pre_event, post_event = post_event,
                        kind = kind, norm = norm, NW = NW, Fs = fs, unit = unit, verbose=verbose)
        for iFld, field in enumerate(fields):
            td_tmp[field] = mt_spectrogram[:,:,:,iFld]
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
    
    if input_dict:
        td = td[0]
    
    return td



def compute_filter(_td, **kwargs):
    '''
    This function computes filters for the signals in td (e.g. td[fields]).

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data to convert.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    kind : str
        Type of filter to apply to the data. Possible options are: 
        'bandpass', 'lowpass', 'highpass', 'sgolay', 'envelope'.
         
    order : int , optional
        Order of the filter. The default value is 5.
         
    add_operation : str , optional
        Manipulate the data after computing the filter, by 'add' or 'subtract'
        the filtered signal from the original one. The default value is None.
        
    f_low : int / float, optional
        Frequency value for low pass filtering. It is used for 'bandpass',
        'lowpass' and 'highpass'.
         
    f_high : int / float, optional
        Frequency value for high pass filtering. It is used for 'bandpass',
        'lowpass' and 'highpass'.
        
    win_len : int / str, optional
        Length of the window used for the sgolay filter.
        If str, it can describe a fs multiplication (e.g. '3fs'). In this case,
        3 times the fs will be used.
        
    override_fields : str, optional
        Decide wether override the existiong signals or build new ones.
        The default value is True.
        
    save_to_params : str, optional
        Save the name of the new filtered signals in the params structure in td.
        The str in input must be the path to the saving location (e.g. params/data/data).
        The default value is False.
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        trial data with filtered signals.

    '''
    # Filtering information
    fields = None
    fs = None
    kind = None
    f_low = None
    f_high = None
    win_len = None
    order = 5
    add_operation = None
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
        elif key == 'f_low':
            f_low = value
        elif key == 'f_high':
            f_high = value
        elif key == 'win_len':
            win_len = value
        elif key == 'order':
            order = value
        elif key == 'add_operation':
            add_operation = value
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
            print('WARNING: key "{}" not recognised by the compute_filter function...'.format(key))
    
    if kind not in ['bandpass','lowpass','highpass','sgolay','envelope']:
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
    
    if kind == 'sgolay' and win_len == None:
        raise Exception('ERROR: for sgolay filter you must input win_len!')
    
    if add_operation != None :
        if add_operation not in ['add','subtract']:
            raise Exception('ERROR: add_operation can only be "add" or "subtract"! You inputed {}.'.format(add_operation))
        else:
            print('WARNING: add_operation only implemented for sgolay filter!')
    
    # Compute filters
    for td_tmp in td:
        signals_name = []
        for iFld, field in enumerate(fields):
            if verbose:
                print('Filtering signal {}/{}'.format(iFld+1, len(fields)))
            if kind == 'bandpass':
                if not override_fields:
                    signal_name = field + '_bp_{}_{}'.format(f_low,f_high)
                else:
                    signal_name = field
                td_tmp[signal_name] = bpff(data = td_tmp[field], lowcut = f_low, highcut = f_high, fs = fs, order=order)
            
            elif kind == 'lowpass':
                if not override_fields:
                    signal_name = field + '_lp_{}'.format(f_low)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = lpff(data = td_tmp[field], lowcut = f_low, fs = fs, order=order)
            
            elif kind == 'highpass':
                if not override_fields:
                    signal_name = field + '_hp_{}'.format(f_high)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = hpff(data = td_tmp[field], highcut = f_high, fs = fs, order=order)
            
            elif kind == 'sgolay':
                if not override_fields:
                    signal_name = field + '_sg_{}'.format(win_len)
                else:
                    signal_name = field
                # print('field: ' + field + '; win_len: ' + str(win_len) + '; order: ' + str(order))
                data_filt = sgolay_filter(data = td_tmp[field], win_len = int(win_len), order=order)
                if add_operation != None:
                    if add_operation == 'add':
                        td_tmp[signal_name] = td_tmp[field] + data_filt
                    elif add_operation == 'subtract':
                        td_tmp[signal_name] = td_tmp[field] - data_filt
                else:
                    td_tmp[signal_name]  = data_filt
            
            elif kind == 'envelope':
                if not override_fields:
                    signal_name = field + '_env_{}_{}'.format(f_low,f_high)
                else:
                    signal_name = field
                
                td_tmp[signal_name] = envelope(data = td_tmp[field], Fs = fs, lowcut = f_low, highcut = f_high, method = 'squared', order = order)
            
            else:
                raise Exception('ERROR: wrong kind of filter applied! Kind given is : {}'.format(kind))
            signals_name.append(signal_name)
    
        if not override_fields and save_to_params:
            subfield = td_subfield(td_tmp,save_to_params_field)
            subfield['signals'].extend(signals_name)
    
    if input_dict:
        td = td[0]
    
    return td
        

def compute_mav(_td, **kwargs):
    '''
    This function computes the mean average value (mav) for the signals in td 
    (e.g. td[fields]).

    Parameters
    ----------
    _td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data to convert.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    window_size_sec : int / float, optional
        Size of the sliding window for computing the mav. It is in seconds.
        The default is 0.5 seconds.
        
    window_step_sec : int / float, optional
        Step of the sliding window for computing the mav. It is in seconds.
        The default is 0.25 seconds.
        
    adjust_target : str, optional
        Fields in td containing the target signals (binary signals).
        It can either be one key in td or the path in td where to find the  
        target signal (e.g. params/data/data).
        
    inplace : bool, optional
        Perform operation on the input data dict. The default is True.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        trial data with filtered signals.

    '''
    # Input info for computing mav
    fields = None
    fs = None
    window_size_sec = 0.5 # in seconds
    window_step_sec = 0.25 # in seconds
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
        elif key == 'wind_size':
            window_size_sec = value
        elif key == 'wind_step':
            window_step_sec = value
        elif key == 'inplace':
            inplace = value
        elif key == 'verbose':
            verbose = value
        elif key == 'adjust_target':
            adjust_target = True
            adjust_target_field = value
        else:
            print('WARNING: key "{}" not recognised by the compute_mav function...'.format(key))
    
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
        
    # Get window's info in samples
    window_size_smp = round(window_size_sec * fs)
    window_step_smp = round(window_step_sec * fs)
    
    for iTd, td_tmp in enumerate(td):
        for iFld, field in enumerate(fields):
            if verbose: print('\MAV signal {}/{} in td {}/{}'.format(iFld+1, len(fields), iTd+1, len(td)))
            td_tmp[field] = moving_average(td_tmp[field], window_step_smp, window_size_smp)
        
        # Re-map target dataset
        if adjust_target:
            for event in target_fields:
                event_mav = moving_average(td_tmp[event], window_step_smp, window_size_smp)
                td_tmp[event] = np.zeros(len(event_mav))
                td_tmp[event][event_mav>1/window_size_smp] = 1
    if input_dict:
        td = td[0]
    
    return td


# =============================================================================
# Features
# =============================================================================
        
def extract_features_instant(td, **kwargs):
    '''
    This function extracts the features from the signals in td (e.g. td[fields]).

    Parameters
    ----------
    td : dict / list of dict, len (n_td)
        Trial data.
        
    fields : str / list of str, len (n_fields)
        Fields in td from which collect the data.
        If str, it can either be the key in td or the path in td where to find 
        the name of the signals to use in td (e.g. params/data/data).
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    time_n : int / list of int, len (n_times), optional
        Number of additional time stamps to cellect before the event.
        The default is 10.
        
    feature_win_sec : float / list of float, len (n_feature_win_sec), optional
        Length of the window from which collect the time points. It is seconds.
        The default is 0.5 seconds.
        
    dead_win_sec : float / list of float, len (n_dead_win_sec), optional
        Length of the dead time around the event. It is in seconds.
        The default is 0.02 seconds.
        
    no_event_sec : float / list of float, len (n_no_event_sec), optional
        Amount of no events to collect for the no_event class. It is in seconds.
        The default is 1 second.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        trial data with filtered signals.

    '''
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
                    if verbose: print('Extract features loop {}/{}'.format(loop_count,loop_n))
                    # Collect signals for features in one 2darray
                    features = []
                    labels = []
                    
                    # Actual number of no_event to use for each td
                    no_event_td = np.round(no_event/len(td)).astype('int')
                    
                    for td_tmp in td:
                        data_fields = [transpose(td_tmp[field],'column') for field in fields]
                        
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
            # Take only positive values
            for td_tmp in td_target:
                for field in target_load_dict['fields']:
                    td_tmp[field] = td_tmp[field][td_tmp[field]>0]
            
            if 'convert_to' in target_load_dict.keys() and 'fs' in target_load_dict.keys():
                for td_tmp in td_target:
                    for field in target_load_dict['fields']:
                        td_tmp[field] = convert_time_samples(td_tmp[field], fs = target_load_dict['fs'], convert_to = target_load_dict['convert_to'])
    
        # Combine target data with the predictor data
        combine_dicts((td, td_target), inplace = True)
    
    if convert_fields_to_numeric_array_dict != None:
        td = convert_fields_to_numeric_array(td, fields = convert_fields_to_numeric_array_dict['fields'], 
                                             vector_target_field = convert_fields_to_numeric_array_dict['target_vector'])    
    
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
    return_epochs = False
    td_epochs = []
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'combine_fields':
            combine_fields_dict = value
        elif key == 'remove_artefacts':
            remove_artefacts_dict = value
        elif key == 'add_segmentation':
            add_segmentation_dict = value
        elif key == 'return_epochs':
            return_epochs = value
    
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
        td_segment = add_segmentation_dict['td_segment']
    
    # Combine artefacts and segmentation
    if remove_artefacts_dict != None and add_segmentation_dict != None:
        td_epochs = combine_epochs(td_artefacts, td_segment)
        if (is_field(remove_artefacts_dict, 'plot') and remove_artefacts_dict['plot'] == True) or\
            (is_field(add_segmentation_dict, 'plot') and add_segmentation_dict['plot'] == True):
                for td_artefacts_tmp, td_segment_tmp in zip(td_artefacts,td_segment):
                    plt.figure(); plt.plot(td_artefacts_tmp['epochs'],'--b'); plt.plot(td_segment_tmp['epochs'],'--r');
                    plt.title('Artefacts: blue; Epochs: Red.')
        td = segment_data(td, td_epochs, remove_artefacts_dict['fs'], invert_epoch = True)
    elif remove_artefacts_dict == None and add_segmentation_dict != None:
        if (is_field(add_segmentation_dict, 'plot') and add_segmentation_dict['plot'] == True):
            for td_tmp in td_segment:
                plt.figure(); plt.plot(td_tmp['epochs']) 
        td_epochs = td_segment
        td = segment_data(td, td_epochs, add_segmentation_dict['fs'], invert_epoch = True)
    elif remove_artefacts_dict != None and add_segmentation_dict == None:
        if (is_field(remove_artefacts_dict, 'plot') and remove_artefacts_dict['plot'] == True):
            for td_tmp in td_epochs:
                plt.figure(); plt.plot(td_artefacts['epochs']) 
        td_epochs = td_artefacts
        td = segment_data(td, td_epochs, remove_artefacts_dict['fs'], invert_epoch = True)
    
    if return_epochs == True:
        return td, td_epochs
    else:
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
        if 'filter' in key:
            operations.append((key,value))
        elif key == 'multitaper':
            operations.append((key,value))
        elif key == 'downsample':
            operations.append((key,value))
        else:
            print('WARNING: key "{}" not recognised by the preprocess pipeline...'.format(key))
    
    for operation in operations:
        if  'filter' in operation[0]:
            td = compute_filter(td, **operation[1])
        elif operation[0] == 'multitaper':
            td = compute_multitaper(td, **operation[1])
        elif operation[0] == 'downsample':
            td = downsample(td, **operation[1])
        elif operation[0] == 'mav':
            td = compute_mav(td, **operation[1])
    
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
    
    
    