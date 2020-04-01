#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:07:29 2020

@author: raschell
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

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
    if data.shape[0]<data.shape[1]:
        data = data.T
    
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
    from utils import find_first
    
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
    

def interpolate1D(array, length_new, kind = 'linear'):
    '''
    This function interpolates the given array to set a different lenght

    Parameters
    ----------
    array : numpy.ndarray
        Array to interpolate.
    length_new : int/float
        New lenght of the array.
    kind : str
        Kind of interpolation to apply to the data

    Returns
    -------
    Array_new.

    '''
    f = interpolate.interp1d(np.arange(len(array)), array, kind = kind, fill_value = 'extrapolate')
    return f(np.linspace(0,len(array),length_new))
    

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
    
    