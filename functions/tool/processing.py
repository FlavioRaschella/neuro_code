#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:07:29 2020

@author: raschell
"""

# Import numpy lib
import numpy as np
# Import processing libs
from scipy import interpolate
# Import utils libs
from utils import find_first, transpose, find_values
# Impot plotting libs
import matplotlib.pyplot as plt

# =============================================================================
# Signal processing
# =============================================================================
def get_baseline(data):
    
    if type(data) is np.ndarray:
        data = [data]
    if type(data) is not list:
        raise Exception('ERROR: data is not list!')
    
    # Combine data
    data_comb = []
    for dt in data:
        data_comb.append(transpose(dt,'column'))
    data_comb = np.concatenate(data_comb, axis = 0)
    
    # Compute stats
    n_data = data_comb.shape[0]
    data_mean = np.mean(data_comb, axis = 0)
    data_std = np.std(data_comb, axis = 0)
    
    return data_comb, data_mean, data_std, n_data
    

def get_trigger_data(data, events, before_event, after_event, mean_norm = [], std_norm = []):
    
    if type(data) is np.ndarray:
        data = [data]
    if type(data) is not list:
        raise Exception('ERROR: data is not list!')
        
    if type(events) is np.ndarray:
        events = [events]
    if type(events) is not list:
        raise Exception('ERROR: events is not list!')
        
    if len(events) != len(data):
        raise Exception('ERROR: len(events) {} != len(data) {}.'.format(len(events), len(data)))
    
    # Check n_channles and n_events
    n_channles = []
    n_events = []
    for dt, event in zip(data, events):
        if dt.ndim == 1:
            n_channles.append(1)
            dt = np.expand_dims(dt,1)
        else:
            n_channles.append(dt.shape[1])
        if event.ndim == 1:
            n_events.append(1)
            event = np.expand_dims(event,1)
        else:
            n_events.append(event.shape[1])
    if (np.diff(n_channles)>0.1).any():
        raise Exception('ERROR: data does not have a fixed n_channles.')
    else:
        n_channles = n_channles[0]
    if (np.diff(n_events)>0.1).any():
        raise Exception('ERROR: events does not have a fixed n_events.')
    else:
        n_events = n_events[0]
    
    if type(before_event) is not int and type(before_event) is not float:
        raise Exception('ERROR: before_event is not int or float.')
    else:
        before_event = int(before_event)
        
    if type(after_event) is not int and type(after_event) is not float:
        raise Exception('ERROR: after_event is not int or float.')
    else:
        after_event = int(after_event)
    
    # Combine data
    data_events = [np.array([], dtype=np.int64).reshape(0,after_event+before_event,n_channles)] * n_events
    for iDt, (dt, event) in enumerate(zip(data, events)):
        n_data = dt.shape[0]
        for iEv in range(event.shape[1]):
            data_event = []
            ev = find_values(event[:,iEv],1,'equal')
            in_borders = np.logical_and(ev-before_event > 0, ev+after_event < n_data)
            # if np.sum(in_borders == False)>0:
            #     print('Trigger {}: {}/{} events out of border'.format(iEv,np.sum(in_borders == False),len(ev)))
            ev = ev[in_borders]
            if (ev == False).all():
                continue
            for ev_sgl in ev:
                data_event.append(dt[range(ev_sgl-before_event,ev_sgl+after_event),:])
            data_events[iEv] = np.concatenate((data_events[iEv], np.array(data_event)), axis = 0)
        
    # Compute stats
    data_mean = []
    data_std = []
    data_sem = []
    data_snr = []
    n_data = []
    for data_event in data_events:
        data_mean.append(np.mean(data_event, axis = 0))
        data_std.append(np.std(data_event, axis = 0))
        n_data.append(data_event.shape[0])
        data_sem.append(np.std(data_event, axis = 0) / data_event.shape[0])
        
        if len(mean_norm) != 0 and len(std_norm) != 0:
            data_snr.append(np.abs(np.mean(data_event, axis = 0)-np.tile(mean_norm,(after_event+before_event,1))) / \
                            (np.std(data_event, axis = 0)+np.tile(std_norm,(after_event+before_event,1))) )
    
    return data_events, data_mean, data_std, n_data, data_sem, data_snr



# =============================================================================
# Epoching
# =============================================================================
def get_epochs(binary_array, Fs, verbose = False):
    
    if binary_array.ndim > 1:
        raise Exception('ERROR: Input array has more that 1 dimension!')
    
    signal_len = binary_array.shape[0]
    
    # Separate dataset in good epochs
    good_start = np.where(np.logical_and(binary_array[:-1]==False ,binary_array[1:]==True))[0]
    if binary_array[0] == 1 and good_start.size == 0:
        good_start = np.insert(good_start,0,0)
    if binary_array[0] == 1 and good_start[0] != 0:
        good_start = np.insert(good_start,0,0)
        
    good_stop = np.where(np.logical_and(binary_array[:-1]==True ,binary_array[1:]==False))[0]
    if binary_array[-1] == 1 and good_stop.size == 0:
        good_stop = np.append(good_stop,signal_len)
    if binary_array[-1] == 1 and good_stop[-1] != signal_len:
        good_stop = np.append(good_stop,signal_len)
    
    # Check that good_start and good_stop have same length
    if len(good_start) != len(good_stop):
        raise Exception('Start and Stop epochs have different length!')
    
    good_epoch = np.where((good_stop - good_start + 1) > 2*Fs)
    good_start = good_start[good_epoch]
    good_stop = good_stop[good_epoch]
    if verbose:
        print('Start: {}; Stop: {}'.format(good_start, good_stop))
    
    epochs = []
    for epoch_start, epoch_end in zip(good_start, good_stop):
        epochs.append(np.arange(epoch_start, epoch_end))
    
    return epochs


def epochs_separation(data, good_epochs, Fs, print_figure = False):
    '''
    Parameters
    ----------
    data : np.ndarray
        Signal array from which remove the artefacts
    Fs : int
        Data sampling frequency
    good_epochs : np.ndarray
        Binary vector of good epochs
    print_figure : bool, optional
        Print original data and epochs data. The default is False.

    Returns
    -------
    epochs : list
        Data blocks for the good epochs

    '''
    
    if type(data) is not np.ndarray:
        raise Exception('ERROR: data in input must have a np.array format!')
        
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
        
    # Convert array to column array
    if data.shape[0]<data.shape[1]:
        data = data.T
    
    if type(good_epochs) is list:
        print('good_epochs is a list --> converting to np.array')
        good_epochs = np.array(good_epochs)
        
    if good_epochs.ndim == 2:
        if 1 in good_epochs.shape:
            good_epochs = np.squeeze(good_epochs)
        else:
            raise Exception('ERROR: good_epochs has 2 dimensions both > 1!')
    elif good_epochs.ndim > 2:
        raise Exception('Dimension of good_epochs is > 2')
    
    # Data length
    data_n = data.shape[0]
    if data_n != len(good_epochs):
        raise Exception('Data len and good epochs len are different!')
    
    # Separate dataset in good epochs
    good_start = np.where(np.logical_and(good_epochs[:-1]==False ,good_epochs[1:]==True))[0]+1
    if good_epochs[0] == 1:
        good_start = np.insert(good_start,0,0)
        
    good_stop = np.where(np.logical_and(good_epochs[:-1]==True ,good_epochs[1:]==False))[0]+1
    if good_epochs[-1] == 1:
        good_stop = np.append(good_stop,data_n)
    
    # Check that good_start and good_stop have same length
    if len(good_start) != len(good_stop):
        raise Exception('Start and Stop epochs have different length!')
    
    # Take epochs that are longer that 2s
    good_epoch = np.where((good_stop - good_start + 1) > 2*Fs)[0]
    good_start = good_start[good_epoch]
    good_stop = good_stop[good_epoch]
    print('Epochs: Start {}; Stop {}'.format(good_start, good_stop))
    
    if print_figure:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(data)
        ax[0].set_title('All dataset')
        ax[1].plot(good_epochs)
        ax[1].set_title('Good epochs')
        fig.tight_layout()
    
    # Divide data in ephocs
    epochs = []
    for idx_ep, (ep_start, ep_stop) in enumerate(zip(good_start,good_stop)):
        take_idx = np.arange(ep_start, ep_stop)
        epochs.append(data[take_idx,:])
        if print_figure:
            fig, ax = plt.subplots(1,1)
            ax.plot(data[take_idx,:])
            ax.set_title('Epoch {}/{}'.format(idx_ep+1,len(good_start)))
            fig.tight_layout()
    
    return epochs
    

# =============================================================================
# Artefactcs
# =============================================================================
def artefacts_removal(data, Fs, method = 'amplitude', n = 1, threshold = None):
    '''
    Parameters
    ----------
    data : np.ndarray
        Signal array from which remove the artefacts
    Fs : int
        Data sampling frequency
    method: str
        Method to find the artefacts
    n : int, optional
        Minimum number of signal with artefact. The default is 1.
    threshold : int, optional
        Signal threshold for considering an artefact. The default is 300.

    Returns
    -------
    good_idx : signals indexes with no artefacts

    '''
    
    if type(data) is not np.ndarray:
        raise Exception('ERROR: data in input must have a np.array format!')
        
    if data.ndim == 1:
        sig_n = 1
        data = np.expand_dims(data, axis=0)
    else:
        sig_n = min(data.shape)
    
    if n > sig_n:
        print('Selected number of signal for artefacts removal ({0}) > the actual number of signals ({1}).\nSetting n = {1}'.format(n,sig_n))
    
    # Convert array to column array
    data = transpose(data,'column')
    
    # Set junk offset
    junk_offset = np.ceil(Fs/2).astype('int')
    
    if threshold == None:
        threshold = np.percentile(data,95, axis = 0).mean()
        print('Threshold not specified. Setting threshold = 95 percentile of the data...')
    
    if method == 'amplitude':
        # Set an array for bad indexes
        idx = np.logical_or(data < -threshold, data > threshold).astype('int')
        # Bad indexes
        bad_idx = (np.sum(idx, axis=1) >= n).astype('int')
    else:
        raise Exception('No other method for finding artefacts implemented!')
    
    # Remove junk period
    junk_init = np.where((bad_idx[:-1]==0) & (bad_idx[1:]==1))[0]+1
    junk_end = np.where((bad_idx[:-1]==1) & (bad_idx[1:]==0))[0]+1
    for ji in junk_init:
        bad_idx[ji-junk_offset:ji] = 1
    
    for je in junk_end:
        bad_idx[je:je+junk_offset] = 1
    
    # # Return good indexes
    # return np.logical_not(bad_idx).astype('int')
    # Return good indexes
    return bad_idx
            

# =============================================================================
# Conversion
# =============================================================================

def convert_points_to_target_vector(points, vector):
    '''
    This function converts an array of points into a target vector.
    [1,4,6] --> [0 1 0 0 1 0 1]
    
    Parameters
    ----------
    points : list of np.array
        Points to be remapped into a vector.
    vector : np.array
        vector for remapping of the points.

    Returns
    -------
    vector_target : np.array
        vector with pointes remapped.

    '''    
    if type(points) is np.ndarray:
        points = [points]
    
    if type(points) is list and type(points[0]) is not np.ndarray:
        raise Exception('ERROR: points input must be a list of np.array!')
    
    if type(vector) is list:
        vector = np.array(vector)
    vector_target = np.zeros(len(vector)).astype('int')
    
    for iPnt, points_list in enumerate(points):
        for point in points_list:
            vector_target[find_first(point,vector)] = iPnt+1
    
    return vector_target
    

def convert_time_samples(points, fs, convert_to = 'time'):
    '''
    This function converts an array of points from time-based to sample-based
    and viceversa.
    
    Parameters
    ----------
    points : list / np.ndarray, shape (n_points,)
        Points to be converted.
        
    fs : int / float
        Sampling frequency.
        
    convert_to : str, optional
        Set to what unit points must be converted: "samples", "time".
        The default is 'time'.

    Returns
    -------
    points : list / np.ndarray
        Converted points.

    '''
    
    if type(points) is not list and type(points) is not np.ndarray:
        raise Exception('ERROR: points input must be either a list or a np.array! It is a "{}"'.format(type(points)))
    
    if type(points) is np.ndarray:
        points = points.astype('float')
    
    if type(fs) is not int and type(fs) is not float:
        raise Exception('ERROR: fs input must be either a int or a float! It is a "{}"'.format(type(fs)))
    
    if convert_to not in ['time', 'samples']:
        raise Exception('ERROR: convert_to input must be either "time" or "samples". It is "{}"!'.format(convert_to))
    
    for iPt, point in enumerate(points):
        if convert_to == 'time':
            points[iPt] = point/fs
        else:
            points[iPt] = point*fs
    
    return points

# =============================================================================
# Interpolation
# =============================================================================
def interpolate1D(array, length_new, kind = 'linear'):
    '''
    This function interpolates the given array to set a different lenght

    Parameters
    ----------
    array : numpy.ndarray, shape (n_array,)
        Array to interpolate.
        
    length_new : int/float
        New lenght of the array.
        
    kind : str, optional
        Kind of interpolation to apply to the data.

    Returns
    -------
    Array_new.

    '''
    f = interpolate.interp1d(np.arange(len(array)), array, kind = kind, fill_value = 'extrapolate')
    return f(np.linspace(0,len(array),length_new))
    
# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    data = np.random.rand(100,10)
    data[12:21,1] = 10
    data[18:25,3] = 10
    good_test = np.ones(100).astype('int'); good_test[18:21] = 0;
    good_idx = artefacts_removal(data, Fs = 0, method = 'amplitude', n = 2, threshold = 9)
    if (good_test!=good_idx).any():
        raise Exception('ERROR: Test artefacts_removal NOT passed.')
    else:
        print('Test artefacts_removal passed!')

    vector = np.arange(11)
    points = np.array([3.1, 6.9, 9.1])
    if (convert_points_to_target_vector(points, vector) - np.array([0,0,0,0,1,0,0,1,0,0,1])>0.1).any():
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test find_first passed!')
    
    print('All implemented tests passed!')
    
    