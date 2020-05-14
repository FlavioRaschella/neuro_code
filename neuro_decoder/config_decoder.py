#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:16:01 2020

@author: raschell
"""

#%% LIBRARIES

# Import data loading, processing
from td_processing import load_pipeline, cleaning_pipeline, preprocess_pipeline
from loading_data import load_data_from_folder

# Numpy lib
import numpy as np

# Plotting libs
from td_utils import td_plot
import matplotlib.pyplot as plt

# Import utils lib
from utils import find_first
import copy

# Import decoding function
from decoder_utils import prepare_cv_uniform, grid_search_cv, bootstrap_MInorm_simple, build_decoder

#%% ===========================================================================
# LOAD DATA
# =============================================================================
DATA_PATH = '../data_test/decoder'
DATA_FILE = 'data_decoder'
DATA_NAME = ['data_1','data_2','data_3','data_4']
TARGET_FILE = 'events_decoder'
TARGET_NAME = ['event_1','event_2']

params = {'data': {'data': {'signals': DATA_NAME, 'time': 'time', 'fs': 'FS'}},
          'event': {'trigger' : {'signals': TARGET_NAME, 'kind': 'array'}}}

LOAD_OPT = {'load': {'path': DATA_PATH, 'files_name': DATA_FILE},
            'trigger_file': {'path': DATA_PATH, 'files_name': TARGET_FILE, 'files_format': '.mat', 'fields': TARGET_NAME},
            'convert_fields_to_numeric_array': {'fields': TARGET_NAME, 'target_vector': 'time'},
            'params': params }

td = load_pipeline(**LOAD_OPT)[0]

#%% ===========================================================================
# CLEAN DATA
# =============================================================================

# Plot raw signals
fig, ax = td_plot(td, y = ['data_1','data_2','data_3','data_4'], x = 'time', events = ['event_1'], sharex = True)

# sgolay filter
FS = td['params']['data']['data']['fs']
FILTER_HP = {'filter': {'kind': 'sgolay', 'fields': 'params/data/data', 'fs': 'params/data/data', 'order': 2, 'win_len': round(3*FS)+1 , 'add_operation' : 'subtract'}}
FILTER_LP = {'filter': {'kind': 'sgolay', 'fields': 'params/data/data', 'fs': 'params/data/data', 'order': 2, 'win_len': round(FS/6)*2+1  }}
td = preprocess_pipeline(td, **FILTER_HP)
td = preprocess_pipeline(td, **FILTER_LP)

# Plot processed signals
td_plot(td, y = ['data_1','data_2','data_3','data_4'], x = 'time', axs = ax)

#%% Get epochs of interest

# Remove artefacts. Get a threshold
thresholds = []
for field in td['params']['data']['data']['signals']:
    thresholds.append( 5*np.std(td[field]) )

fig, axs = plt.subplots(len(td['params']['data']['data']['signals']),1, sharex = True)
for field, th, ax in zip(td['params']['data']['data']['signals'], thresholds, axs):
    ax.plot(td[field],'-b')
    ax.axhline(th,color = 'r')
    ax.axhline(-th,color = 'r')

remove_method = 'amplitude'
remove_threshold = np.mean(thresholds)
remove_signal_n = 2

# Collect additional epochs to remove
EPOCHS_FILE = 'epochs_decoder'
td_epochs = load_data_from_folder(folders = DATA_PATH, files_name = EPOCHS_FILE)[0]

epochs_on  = np.round(FS*(td_epochs['epochs_on'])).astype('int')
epochs_off = np.round(FS*(td_epochs['epochs_off'])).astype('int')
epochs = np.zeros(td[DATA_NAME[0]].shape)

if epochs_off[0] < epochs_on[0]:
    epochs[:epochs_off[0]] = 1
for epoch_on in epochs_on:
    if find_first(epoch_on,epochs_off) != None:
        epoch_off_next = epochs_off[find_first(epoch_on,epochs_off)]
        epochs[epoch_on:epoch_off_next] = 1
if epochs_on[-1] > epochs_off[-1]:
    epochs[epochs_on[-1]:] = 1

# Plot epochs
fig, axs = plt.subplots(len(td['params']['data']['data']['signals']),1, sharex = True)
for field, th, ax in zip(td['params']['data']['data']['signals'], thresholds, axs):
    ax.plot(td[field],'-b')
    ax.axhline(th,color = 'r')
    ax.axhline(-th,color = 'r')
    ax.plot((epochs-0.5)*0.1,'-c', linewidth=0.5)
    ax.set_title(field)

# Separate epochs periods
td_epochs = [{'epochs': epochs, 'fs': FS}]
CLEANING_OPT = {'remove_artefacts': {'fields': 'params/data/data', 'fs': 'params/data/data', 'method': remove_method, 'threshold': remove_threshold, 'signal_n': remove_signal_n},
                'add_segmentation': {'td_segment': td_epochs, 'plot': True}}
td = cleaning_pipeline(td,**CLEANING_OPT)

#%% ===========================================================================
# Prepare data for decoding
# =============================================================================

data = []
events = []
for td_tmp in td:
    data.append(np.array([td_tmp[field] for field in td_tmp['params']['data']['data']['signals']]).T)
    events.append(np.array([td_tmp[field] for field in td_tmp['params']['event']['trigger']['signals']]).T)

# Separate data for crossvalidation
cv_blocks, data, events, n_blocks, _, _ = prepare_cv_uniform(data, events, deviation = 0.5, cv_division = 4)

# Get a copy of the events supposing a shift of [0, 0]
shifts = [0, 0]
events_shifted = copy.deepcopy(events)

#%% ===========================================================================
# Decode events
# =============================================================================

# Build features
time_n = np.array([5, 7, 10]) # Number of decoding taps
feature_win = np.array([0.5, 0.7, 0.8]) * FS # Length of the decoding window
dead_win = np.array([0.01]) * FS # 0.01 seconds
no_event = np.array([200]) * FS # 200 seconds
# Tolerance windows
win_tol = np.arange(0.05,0.45,0.05)*FS
# Estimator
estimator = 'rLDA'

regularization_coeff = np.concatenate([[0], np.arange(0.1,1,0.1), [0.99]])
threshold_detect = 0.8
refractory_period = 0.5*FS

params_clf = {'regularization_coeff': regularization_coeff,
              'threshold_detect': threshold_detect,
              'refractory_period': refractory_period}

params_data = {'time_n': time_n,
              'feature_win': feature_win,
              'dead_win': dead_win,
              'no_event': no_event,
              'shifts': shifts}

# Compute the grid_search
models, best_model = grid_search_cv(estimator, data, events, events_shifted, cv_blocks, win_tol, params_clf, params_data)

# Get mutual information for the best decoder
plt.figure();
plt.plot(best_model['params']['win_tol'],best_model['score'])
plt.xlabel('Tolerance window [ms]')
plt.ylabel('MI')
plt.title('Mutual information by tolerance window')

#%% ===========================================================================
# Evaluate best decoder performance
# =============================================================================

params_clf = {'regularization_coeff': best_model['params']['regularization_coeff'],
              'threshold_detect': best_model['params']['threshold_detect'],
              'refractory_period': best_model['params']['refractory_period']}

params_data = {'time_n': best_model['params']['time_n'],
               'feature_win': best_model['params']['feature_win'],
               'dead_win': best_model['params']['dead_win'],
               'no_event': best_model['params']['no_event'],
               'shifts': shifts}

model = build_decoder(estimator, data, events, events_shifted, cv_blocks, params_clf, params_data, win_tol, plot = True)[0]
# MI for tollerances
plt.figure();
plt.plot(model['params']['win_tol'],model['score'])
plt.xlabel('Tolerance window [ms]')
plt.ylabel('MI')
plt.title('Mutual information by tolerance window')

# Confusion matrix
conf_matrix = np.round(model['conf_matrix'][-1,:,:]).astype('int')
bootMI = bootstrap_MInorm_simple(conf_matrix)

# EOF