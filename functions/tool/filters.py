#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:29:32 2020

@author: raschell

This library contains filter for signal processing
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, detrend, hilbert, decimate, savgol_filter
from utils import transpose

# Design filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a

def butter_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a

# Apply unidirectional filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Apply bidirectional filter
def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filtfilt(data, lowcut, fs, order=5):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filtfilt(data, highcut, fs, order=5):
    b, a = butter_highpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def sgolay_filter(data, win_len, order=5):
    return data - savgol_filter(x = data, window_length = win_len, polyorder = order)


# Moving average
def average(data, periods):
    '''
    Compute the average for a defined period.

    Parameters
    ----------
    data : numpy array
        1D dataset over which computing the moving average.
    periods : int
        Period in samples over which computing the moving average.

    Returns
    -------
    data_out : numpy array
        Signal average.

    '''
    if type(data) == np.ndarray and len(data.shape) == 2:
        data_out = np.empty(shape = (data.shape[0], data.shape[1]-periods+1))
        for idx, dt in enumerate(data):
            weights = np.ones(periods) / periods
            data_out[idx,:] = np.convolve(dt, weights, mode='valid')
    else:
        weights = np.ones(periods) / periods
        data_out = np.convolve(data, weights, mode='valid')
    return data_out

def moving_average(data, periods):
    return print('To be implemented!')
    

# Envelope
def envelope(data, Fs, lowcut = 0, highcut = 0, method = 'squared', order = 4):
    '''
    The function computes the envelope of the signal by two different methods.
    First Method: By USing Low Pass Filter. The data is Squared, Passed
    through LPF and then taken square root.
    Second Method: Using Hilbert Transform. Hilbert Transform is taken using
    the inbuilt function in Matlab

    Parameters
    ----------
    signal : numpy array
        Signal over which compute the envelope.
    Fs : float
        Sampling frequency [Hz].
    lowcut : float
        low pass frequency [Hz].
    highcut : float
        high pass frequency [Hz].
    method : TYPE
        Type of method for computing the envelope: 'abs','squared', hilbert.

    Returns
    -------
    Envelope of the signal.

    '''
    
    if type(data) == list:
        data = np.array(data)
    
    if data.ndim != 1:
        raise Exception('ERROR: data dimension different from 1!')
        
    # Band Pass
    if highcut != 0:
        data_filt = detrend(butter_bandpass_filtfilt(data, highcut, 450, Fs, order))
    else:
        data_filt = detrend(data);
    
    # Rectification
    if method == 'squared':
        # Squaring for rectifing
        # gain of 2 for maintianing the same energy in the output
        data_filt_rect = 2 * data_filt * data_filt;
    elif method == 'abs':
        data_filt_rect = np.abs(data_filt)
    elif method == 'hilber':
        data_filt_rect = np.abs(hilbert(data_filt))
    
    # Low Pass
    if method == 'squared' or method == 'abs':
        data_filt = butter_lowpass_filtfilt(data_filt_rect, lowcut, Fs, order)
    
    if method == 'squared':
        envelope = (np.abs(data_filt))**(0.5)
    else:
        envelope = data_filt;
    
    return envelope

# Downsample dataset
def downsample_signal(data, fs, target_fs):
    '''
    This function downsamples the data in input

    Parameters
    ----------
    data : np.ndarray
        Data to downsample.
    fs : int / float
        Sampling frequency of the data.
    target_fs : int / float
        Target sampling frequency of the data.

    Returns
    -------
    y : np.ndarray
        Downsampled data.
    actual_fs : int
        Actual sample frequency.

    '''
    
    decimation_ratio = np.round(fs / target_fs).astype('int')
        
    data = transpose(data,'row')
    if fs < target_fs:
        raise ValueError("ERROR: fs < target_fs")
    else:
        try:
            y = decimate(data, decimation_ratio, 3, zero_phase=True)
        except:
            y = decimate(data, decimation_ratio, 3)
        actual_fs = fs / decimation_ratio
                
    return transpose(y,'column'), actual_fs 
