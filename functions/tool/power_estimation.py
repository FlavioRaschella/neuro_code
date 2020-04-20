#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:47:14 2020

@author: raschell
"""

from scipy.signal import hilbert
from scipy.signal.windows import dpss
import math
import numpy as np
from utils import transpose
import matplotlib.pyplot as plt

# Extract signal power from Hilber transformation
def hilbert_transform(data):
    return  np.square(np.abs(hilbert(data)))

def process_spectrogram_params(window_start, window_size_smp, freq_range, Fs = None, NFFT = None):
    # set the NFFT
    if NFFT==None:
        NFFT = np.max([256, 2**nextpow2(window_size_smp)])
    
    if Fs == None:
        Fs = 2*np.pi
    
    # Frequency info
    df = Fs/NFFT;
    sfreqs = np.arange(0,Fs/2+df,df) # all possible frequencies
    freq_idx = np.where(np.logical_and(sfreqs>=freq_range[0],sfreqs<=freq_range[1]))[0]
    sfreqs = sfreqs[freq_idx]
    # Time info
    window_middle_times = window_start + round(window_size_smp/2)
    stimes = window_middle_times/Fs
    
    return df, sfreqs, stimes, freq_idx

def nextpow2(x):
    return math.ceil(math.log2(np.abs(x)))

def pow2db(data):
    '''
    This function converts power amplitude to db.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Input data vector.
        
    Returns
    -------
    data : np.ndarray, shape (n_windows, n_channels)
        Input data vector in db.
        
    '''
    return (10.*np.log10(data)+300)-300;
    

def display_spectrogram_params(NW, window_size_smp, window_step_smp, df, Fs= None):
    
    # Correct input values
    tapers_n = (np.floor(2*NW)-1).astype('int')
    if Fs == None:
        Fs = 2*np.pi
    
    # Display spectrogram properties
    print(' ');
    print('Multitaper Spectrogram Properties:');
    print('Spectral Resolution: {}Hz'.format(df) );
    print('Window Length: {}s'.format( np.round(1000*window_size_smp/Fs)/1000) );
    print('Window Step: {}s'.format( np.round(1000*window_step_smp/Fs)/1000) );
    print('Time Half-Bandwidth Product: {}'.format(NW) );
    print('Number of Tapers: {}'.format(tapers_n) );

def pmtm_params(Fs, NFFT):
    # Frequency info
    df = Fs/NFFT;
    sfreqs = np.arange(0,Fs,df) # all possible frequencies
    return sfreqs


def pmtm(data, NW = 4, Fs = None, NFFT = None, kind = 'chronux'):
    '''
    Compute the power spectrum via Multitapering.
    If the number of tapers == 1, it is a stft (short-time fourier transform)
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Input data vector.
        
    NW : int / float, optional
        Time Half-Bandwidth Product. The default is 4.
        
    Fs : int / float, optional
        Sampling frequency. The default is 2*np.pi.
        
    NFFT : int, optional
        Length of the signal for the FFT analisys. The default value is
        max(256, 2**nextpow2(n_samples)).
        
    kind : str, optional
        Type of computation. It can be "milekovic" or "chronux".
        The default value is chronux.

    Returns
    -------
    Sk : np.ndarray, shape (n_windows, n_channels)
        Power spectrum computed via MTM.

    '''
    
    if kind not in ['milekovic','chronux']:
        raise Exception('ERROR: kinds can only be "milekovic" or "chronux". You inputed "{}"!'.format(kind))
        
    # Number of channels
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    else:
        data = transpose(data, 'column')
    
    # Data length
    N = data.shape[0]
    n_channels = data.shape[1]
    
    if Fs == None:
        Fs = 2*np.pi
    
    # set the NFFT
    if NFFT==None:
        NFFT = max(256, 2**nextpow2(N))
    
    w = pmtm_params(Fs, NFFT)
    
    # Compute tapers
    if kind in ['chronux']:
        if NW == 1:
            tapers = np.expand_dims(np.hamming(N),1) * np.sqrt(Fs)
        else:
            tapers = dpss(N, NW, Kmax=2*NW-1) * np.sqrt(Fs)
    elif kind in ['milekovic']:
        if NW == 1:
            tapers = np.expand_dims(np.hamming(N)/np.linalg.norm(np.hamming(N)),1)
        else:
            tapers = dpss(N, NW, Kmax=2*NW-1)
    tapers = transpose(tapers,'column')
    n_tapers = tapers.shape[1]
    
    # Add channel indices to tapers
    tapers = np.tile(np.expand_dims(tapers,2),(1,1,n_channels))
    # Add taper indices to data
    data = np.tile(np.expand_dims(data,1),(1,n_tapers,1))
    
    data_proj = data*tapers
    
    if kind in ['chronux']:
        Sk_complex = (np.fft.fft(data_proj.T, NFFT)/Fs).T
    elif kind in ['milekovic']:
        Sk_complex = (np.fft.fft(data_proj.T, NFFT)).T
    
    # Sk = np.mean(abs(Sk_complex), axis = 1)
    
    # Sk = np.empty((NFFT, n_channels))
    # Sk[:] = np.NaN
    
    # for channel in range(n_channels):
    #     # Compute the FFT
    #     Sk_complex = np.fft.fft(np.multiply(tapers.transpose(), data[:,channel]), NFFT)/Fs
    #     # Compute the whole power spectrum [Power]
    #     # Sk[:,channel] = np.mean(abs(Sk_complex)**2, axis = 0)
    #     Sk[:,channel] = np.mean(abs(Sk_complex), axis = 0)

    return Sk_complex, w, NFFT


def compute_psd(Sk_complex, w, NFFT, Fs = None, kind = 'chronux', unit = 'power'):
    '''
    Compute the 1-sided PSD [Power/freq].
    Also, compute the corresponding freq vector & freq units.

    Parameters
    ----------
    Sk : np.ndarray, shape (n_samples, n_freq)
        Whole power spectrum [Power]; it can be a vector or a matrix.
        For matrices the operation is applied to each column.
        
    w : np.ndarray, shape (n_freq, )
        Frequency vector in rad/sample or in Hz.
        
    NFFT : int
        Length of the signal for the FFT analisys.
        
    Fs : int / float, optional
        Sampling Frequency. The default is 2*np.pi.
        
    kind : str, optional
        Type of computation. It can be "milekovic" or "chronux".
        The default value is chronux.
        
    unit : str, optional
        Output unit of the psd. The default is 'power'.
        
    Returns
    -------
    psd : np.ndarray, shape (n_samples, n_freq)
        One-sided PSD
        
    w : np.ndarray, shape (n_freq, )
        One-sided frequency vector.
        
    '''
        
    if kind not in ['milekovic','chronux']:
        raise Exception('ERROR: kinds can only be "milekovic" or "chronux". You inputed "{}"!'.format(kind))
        
    # Generate the one-sided spectrum [Power]
    if NFFT % 2 == 0:
        select = np.arange(int(NFFT/2 +1)) # EVEN
        Sk_unscaled = Sk_complex[select,:,:] # Take only [0,pi] or [0,pi)
        # Sk_unscaled[1:-1,:] = 2*Sk_unscaled[1:-1,:] # Don't double unique Nyquist point
    else:
        select = np.arange(int((NFFT+1)/2)) # ODD
        Sk_unscaled = Sk_complex[select,:,:] # Take only [0,pi] or [0,pi)
        # Sk_unscaled[1:,:] = 2*Sk_unscaled[1:,:] # Only DC is a unique point and doesn't get doubled
    
    w = w[select]
    
    # Compute the PSD [Power/freq]
    if Fs != None:
        # psd = Sk_unscaled/Fs # Scale by the sampling frequency to obtain the psd 
        # psd = Sk_unscaled/np.sqrt(Fs) # Scale by the sampling frequency to obtain the psd 
        units = 'Power/Frequency (Power Amplitude/Hz)'
    else:
        Fs = 2*np.pi
        # psd = Sk_unscaled/(2*np.pi) # Scale the power spectrum by 2*pi to obtain the psd
        units = 'rad/sample'
    
    if kind in ['chronux']:
        psd = np.mean(abs(np.conj(Sk_unscaled)*Sk_unscaled), axis = 1)
    elif kind in ['milekovic']:
        psd = np.mean(abs(Sk_unscaled), axis = 1) / np.sqrt(Fs)
    
    if unit == 'db':
        psd = pow2db(psd)
        if Fs != None:
            units = 'Power/Frequency (dB/Hz)'
        else:
            units = 'Power/Frequency (dB/(rad/sample))'
    
    return psd, w, units

# Multitaper power estimation
def moving_pmtm(data, win_size, win_step, freq_range,
                norm = None, NW = 4, Fs = None, NFFT=None, kind = 'chronux', unit = 'power', verbose=False):
    '''
    Compute the power spectrum via Multitapering.
    If NW == 1, it is a stft (short-time fourier transform). In fact tapers_n = 2*NW-1.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Input data vector.
        
    win_size : int
        Size of the sliding window for computing the pmtm. It is in samples.
        
    win_step : int
        Step of the sliding window for computing the pmtm. It is in samples.
        
    freq_range : list, len (2)
        Min and max frequencies of the spectogram.
        
    norm : np.ndarray, optional
        Normalisation values for each channel. The default is None.
        
    NW : int, shape (n_freq, n_channels), optional
        Time Half-Bandwidth Product for computing the pmtm. The default is 4.
        
    Fs : int / float, optional
        Sampling frequency. The default is 2*np.pi.
        
    NFFT : int, optional
        Length of the signal for the FFT analisys. The default value is
        max(256, 2**nextpow2(n_samples)).
        
    kind : str, optional
        Type of computation. It can be "milekovic" or "chronux".
        The default value is chronux.
        
    unity : str, optional
        Unity of the output computed power. It can be 'power' or 'db'. 
        The default is 'power'.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.
    
    Returns
    -------
    mt_spectrogram : np.ndarray, shape (n_windows, n_freq, n_channels)
        Power spectrum computed via MTM over the channels.

    sfreqs : np.ndarray, shape (n_freq,)
        Array of frequencies of the spectograms.

    stimes : np.ndarray, shape (n_windows,)
        Array of time instants of the spectograms.

    '''
    # Check input variables
    if unit not in ['power','db']:
        raise Exception('ERROR: wrong unit assigned!')
    
    if type(data) is list:
        data = np.array(data)
    
    if kind not in ['milekovic','chronux']:
        raise Exception('ERROR: kinds can only be "milekovic" or "chronux". You inputed "{}"!'.format(kind))
    
    # Number of channels
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    else:
        data = transpose(data, 'column')
        
    N = data.shape[0]
    n_channels = data.shape[1]
    if verbose:
        print('Data length: {}. Number of channels: {}.'.format(N,n_channels))

    # Set win_size in int format
    win_size = int(win_size)
    win_step = int(win_step)

    # Compute pmtm features
    win_start = np.arange(0,N-win_size+1,win_step).astype('int')
    df, sfreqs, stimes, freq_idx = process_spectrogram_params(win_start, win_size, freq_range, Fs)
    
    # Compute spectrogram
    mt_spectrogram = np.zeros((len(win_start),len(freq_idx),n_channels))
    
    # If normalisation values are assigned, check that the dimensions fits with
    # the ones of the spectograms
    if type(norm) == np.ndarray:
        norm = transpose(norm,'column')
        if norm.shape[0] != len(sfreqs):
            raise Exception('ERROR: normalisation array 1st dimensions "{}" != from spectogram freq dimension {}!'.format(norm.shape[0],len(sfreqs)))
        if norm.ndim == 2 and norm.shape[1] != n_channels:
            raise Exception('ERROR: normalisation array 2nd dimensions "{}" != from the number of channels {}!'.format(norm.shape[1],n_channels))
    
    for iWin, win_idx in enumerate(win_start):
        Sk_complex, w, NFFT = pmtm(data[win_idx:win_idx+win_size,:], NW, Fs, kind = kind)
        psd, w, _ = compute_psd(Sk_complex, w, NFFT, Fs, unit = unit, kind = kind)
        # Save values
        mt_spectrogram[iWin,:,:] = psd[freq_idx]
        
    if type(norm) == np.ndarray:
        norm_mat = np.tile(norm, (len(stimes),1,1))
        mt_spectrogram = mt_spectrogram / norm_mat
        
    # Display spectrogram info
    if verbose:
        display_spectrogram_params(NW, win_size, win_step, df, Fs)
        
    if n_channels == 1:
        mt_spectrogram = np.squeeze(mt_spectrogram)
    
    return mt_spectrogram, sfreqs, stimes


# Multitaper power estimation around an event
def moving_pmtm_trigger(data, events, win_size, win_step, freq_range, pre_event, post_event,
                        norm = None, NW = 4, Fs = None, NFFT=None, kind = 'chronux', unit = 'power', verbose=False):
    '''
    Compute the power spectrum via Multitapering.
    If NW == 1, it is a stft (short-time fourier transform). In fact tapers_n = 2*NW-1.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Input data vector.
        
    events : list of int, len (n_events)
        List of events around which to compute the pmtm. It is in samples.
        
    win_size : int
        Size of the sliding window for computing the pmtm. It is in samples.
        
    win_step : int
        Step of the sliding window for computing the pmtm. It is in samples.
        
    freq_range : list of float/int, len (2)
        Min and max frequencies of the spectogram.
        
    pre_event : int
        Samples before the event to use for the pmtm. win_step/2 will be added
        to pre_event to account for the windowing.
        
    post_event : int
        Samples after the event to use for the pmtm. win_step/2 will be added
        to pre_event to account for the windowing.
        
    norm : np.ndarray, shape (n_freq, n_channels), optional
        Normalisation values for each channel. The default is None.
        
    NW : int, optional
        Time Half-Bandwidth Product for computing the pmtm. The default is 4.
        
    Fs : int / float, optional
        Sampling frequency. The default is 2*np.pi.
        
    NFFT : int, optional
        Length of the signal for the FFT analisys. The default value is
        max(256, 2**nextpow2(n_samples)).
        
    kind : str, optional
        Type of computation. It can be "milekovic" or "chronux".
        The default value is chronux.
        
    unity : str, optional
        Unity of the output computed power. It can be 'power' or 'db'. 
        The default is 'power'.
        
    verbose : bool, optional
        Narrate the several operations in this method. The default is False.
    
    Returns
    -------
    mt_spectrogram : np.ndarray, shape (n_windows, n_freq, n_events, n_channels)
        Power spectrum computed via MTM around each event for each channel.

    sfreqs : np.ndarray, shape (n_freq,)
        Array of frequencies of the spectograms.

    stimes : np.ndarray, shape (n_windows,)
        Array of time instants of the spectograms.

    '''
    # Check input variables
    if unit not in ['power','db']:
        raise Exception('ERROR: wrong unit assigned!')
    
    if type(data) is list:
        data = np.array(data)
    
    if type(events) is int or type(events) is float:
        events = [events]
        
    if type(events) is np.ndarray:
        events = events.tolist()
        
    if type(events) is not list:
        raise Exception('ERROR: events must be a list! You inputed a "{}"'.format(type(events)))
    
    if (type(pre_event) is not int and type(pre_event) is not float) or \
        (type(post_event) is not int and type(post_event) is not float):
        raise Exception('ERROR: pre_event and post_event must be either an int or a float! You inputed: pre_event "{}", post_event "{}"'.format(type(pre_event),type(post_event)))
    
    if kind not in ['milekovic','chronux']:
        raise Exception('ERROR: kinds can only be "milekovic" or "chronux". You inputed "{}"!'.format(kind))
    
    # Number of channels
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)
    else:
        data = transpose(data, 'column')
        
    n_channels = data.shape[1]
    n_events = len(events)
    if verbose:
        print('Number of events: {}. Number of channels: {}.'.format(n_events, n_channels))

    # Set win_size in int format
    win_size = int(win_size)
    win_step = int(win_step)
    pre_event = int(pre_event + win_size/2)
    post_event = int(post_event + win_size/2)
    
    # Compute pmtm features
    win_start = np.arange(-pre_event,post_event-win_size+win_step,win_step).astype('int')
    df, sfreqs, stimes, freq_idx = process_spectrogram_params(win_start, win_size, freq_range, Fs)
        
    # Compute spectrogram
    mt_spectrogram = np.zeros((len(win_start),len(freq_idx),n_events,n_channels))
    
    # If normalisation values are assigned, check that the dimensions fits with
    # the ones of the spectograms
    if type(norm) == np.ndarray:
        norm = transpose(norm,'column')
        if norm.shape[0] != len(sfreqs):
            raise Exception('ERROR: normalisation array 1st dimension "{}" != from spectogram freq dimension {}!'.format(norm.shape[0],len(sfreqs)))
        if norm.ndim == 2 and norm.shape[1] != n_channels:
            raise Exception('ERROR: normalisation array 2nd dimension "{}" != from the number of channels {}!'.format(norm.shape[1],n_channels))
    
    for iEv, event in enumerate(events):
        for iWin, win_idx in enumerate(win_start):
            Sk_complex, w, NFFT = pmtm(data[event+win_idx:event+win_idx+win_size,:], NW, Fs, kind = kind)
            psd, w, _ = compute_psd(Sk_complex, w, NFFT, Fs, unit = unit, kind = kind)
            # Save values
            mt_spectrogram[iWin,:,iEv,:] = psd[freq_idx]
    
    if type(norm) == np.ndarray:
        if norm.ndim == 1:
            norm = np.expand_dims(np.expand_dims(norm, axis=1), axis=1)
        else:
            norm = np.expand_dims(norm, axis=1)
        norm_mat = np.tile(norm, (len(stimes),1,n_events,1))
        mt_spectrogram = mt_spectrogram / norm_mat
        
    # Display spectrogram info
    if verbose:
        display_spectrogram_params(NW, win_size, win_step, df, Fs)
        
    if n_channels == 1:
        mt_spectrogram = np.squeeze(mt_spectrogram)
    
    return mt_spectrogram, sfreqs, stimes


def get_informative_bands(mt_spectrogram, sfreqs = [], plot = None):
    '''
    This function computes the most informative bands.

    Parameters
    ----------
    mt_spectrogram : np.ndarray, shape (n_samples, n_freq)
        Spectogram if the signal.
        
    sfreqs : np.ndarray, shape (n_freq, ), optional
        Frequency band over which the mt_spectrogram was computed.
        
    Returns
    -------
    spect_band_info : np.ndarray
        Show the bands which contains the most information
    '''

    spect_mean = np.mean(mt_spectrogram, axis = 0)
    spect_std = np.std(mt_spectrogram, axis = 0)
    spect_info_band = spect_std/spect_mean
    
    if plot and sfreqs == []:
        plt.plot(spect_info_band)
    elif plot:
        plt.plot(sfreqs, spect_info_band)
        
    return spect_info_band, spect_mean

# EOF
    