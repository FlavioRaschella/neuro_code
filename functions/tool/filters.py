#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 14:29:32 2020

@author: raschell

This library contains filter for signal processing
"""

import numpy as np
from scipy.signal import butter, lfilter, filtfilt, detrend, hilbert, decimate, savgol_filter, iirfilter
from utils import transpose
from mne.filter import notch_filter

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

def notch_filtfilt(data, fs, fs_cut, fs_band, verbose = False):
    # Fp1 = fs_cut - fs_band / 2 and Fs2 = fs_cut + fs_band / 2 
    return notch_filter(x = data, Fs = fs, freqs = fs_cut, trans_bandwidth = fs_band, verbose = verbose)

def sgolay_filter(data, win_len, order=5):
    return data - savgol_filter(x = data, window_length = win_len, polyorder = order)

# Average
def average(data):
    '''
    Compute the average over the data.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Data over which computing the average.

    Returns
    -------
    data_out : np.ndarray, shape (n_channels,)
        Average of the data.

    '''
    if type(data) != np.ndarray:
        raise Exception('ERROR: data in input must be a np.ndarray! It is a {}'.format(type(data)))
    
    if data.ndim == 1:
        data = np.expand_dims(data,axis = 1)
    
    return np.ma.average(data,axis=0).data

# Moving average
def moving_average(data, win_step, win_size):
    '''
    Compute the moving average over the data.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Data over which computing the moving average.
    win_step : int
        Step of the sliding window. win_step is in samples.
    win_size : int
        Size of the sliding window. win_size is in samples.

    Returns
    -------
    data_out : np.ndarray, shape (n_windows, n_channels).
        Signal average. n_windows = round((n_samples - win_size)/win_step)
        
    '''
    if type(data) != np.ndarray:
        raise Exception('ERROR: data in input must be a np.ndarray! It is a {}'.format(type(data)))
    
    if data.ndim == 1:
        data = np.expand_dims(data,axis = 1)
    
    n_samples = data.shape[0]
    wins_start = np.arange(0,n_samples-win_size,win_step).astype('int')
    
    data_out = []
    for win_start in wins_start:
        data_out.append(average(data[win_start : win_start+ win_size, :]))
    
    return np.array(data_out)

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
