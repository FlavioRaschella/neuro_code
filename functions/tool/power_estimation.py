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

# Extract signal power from Hilber transformation
def hilbert_transform(data):
    return  np.square(np.abs(hilbert(data)))

def process_spectrogram_params(Fs, window_start, window_size_smp, freq_range, NFFT = None):
    
    # set the NFFT
    if NFFT==None:
        NFFT = max(256, 2**nextpow2(window_size_smp))
    
    # frequency
    df = Fs/NFFT;
    sfreqs = np.arange(0,Fs,df) # all possible frequencies
    freq_idx = np.where(np.logical_and(sfreqs>=freq_range[0],sfreqs<=freq_range[1]))[0]
    sfreqs = sfreqs[freq_idx]
    # time
    window_middle_times = window_start + round(window_size_smp/2);
    stimes = window_middle_times/Fs;
    
    return df, sfreqs, stimes, freq_idx

def nextpow2(x):
    return math.ceil(math.log2(np.abs(x)))

def pow2db(data):
    """
    ydB = 10*log10(y);
    ydB = db(y,'power');
    We want to guarantee that the result is an integer
    if y is a negative power of 10.  To do so, we force
    some rounding of precision by adding 300-300.
    """
    return (10.*np.log10(data)+300)-300;
    

def display_spectrogram_params(NW, window_size_smp, window_step_smp, tapers_n, df, Fs):
    # Display spectrogram properties
    print(' ');
    print('Multitaper Spectrogram Properties:');
    print('Spectral Resolution: {}Hz'.format(df) );
    print('Window Length: {}s'.format( np.round(1000*window_size_smp/Fs)/1000) );
    print('Window Step: {}s'.format( np.round(1000*window_step_smp/Fs)/1000) );
    print('Time Half-Bandwidth Product: {}'.format(NW) );
    print('Number of Tapers: {}'.format(tapers_n) );

def pmtm(data, Fs, NW=None, NFFT=None, v=None):
    """
    Multitapering spectral estimation
        
    """
    N = len(data)

    # if dpss not provided, compute them
    if v is None:
        if NW is not None:
            tapers = dpss(N, NW, Kmax=2*NW).T*math.sqtr(Fs)
        else:
            raise ValueError("NW must be provided (e.g. 2.5, 3, 3.5, 4")
    elif v is not None:
        tapers = v[:]
    else:
        raise ValueError("if e provided, v must be provided as well and viceversa.")

    # set the NFFT
    if NFFT==None:
        NFFT = max(256, 2**nextpow2(N))

    Sk_complex = np.fft.fft(np.multiply(tapers.transpose(), data), NFFT)/Fs
    Sk = np.mean(abs(Sk_complex)**2 , axis=0)

    return Sk_complex, Sk


def moving_pmtm(data, Fs, win_size, win_step, freq_range, NW=None, NFFT=None, verbose=False):
    
    N = np.max(data.shape)
    win_start = np.arange(0,N-win_size,win_step)
    
    # Compute pmtm features
    df, sfreqs, stimes, freq_idx = process_spectrogram_params(Fs, NFFT, win_start, win_size,freq_range)
    
    tapers = dpss(M = win_size, NW = NW, Kmax = NW*2).T*math.sqrt(Fs)
    tapers_n = np.min(tapers.shape)
    
    # Compute spectrogram
    mt_spectrogram = np.zeros((len(win_start),len(freq_idx)))
    
    counter = 0
    for idx in win_start:
        Sk_complex, Sk = pmtm(data[idx:idx+win_size], Fs, v=tapers)
        mt_spectrogram[counter,:] = Sk[freq_idx]
        counter += 1
        
    # Display spectrogram info
    if verbose:
        display_spectrogram_params(NW, win_size, win_step, tapers_n, df, Fs)
        
    return pow2db(np.flip(mt_spectrogram.T)), sfreqs, stimes
    
    
    
    