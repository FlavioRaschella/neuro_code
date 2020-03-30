#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:16:01 2020

@author: raschell
"""

#%% Libraries
# Import data loading, processing
from td_processing import load_pipeline, cleaning_pipeline, preprocess_pipeline, features_pipeline
from td_utils import td_plot

#%% Load data
DATA_PATH = '/Volumes/MK_EPIOS/PD/Initiation/Data/Patient6_May2019/PostSx_Day2'
DATA_FILE = [3,4]
TARGET_NAME = ['MANUAL_EV_RFS_time','MANUAL_EV_RFO_time','MANUAL_EV_LFS_time','MANUAL_EV_LFO_time']
# TARGET_NAME = 'MANUAL_EV_RFS_time'

params = {'data': {'data': {'signals': ['LFP_BIP7','LFP_BIP8','LFP_BIP9','LFP_BIP10','LFP_BIP11','LFP_BIP12'], 'time': 'LFP_time', 'fs': 'LFP_Fs'}},
          'event': {'foot' : {'signals': TARGET_NAME, 'kind': 'array'}}}

LOAD_OPT = {'load': {'path': DATA_PATH, 'files_number': DATA_FILE},
            'trigger_file': {'path': DATA_PATH, 'files_number': DATA_FILE, 'files_format': '.mat', 'fields': TARGET_NAME, 'pre_ext': '_B33_MANUAL_gaitEvents'},
            'convert_fields_to_numeric_array': {'fields': TARGET_NAME,'target_vector': 'LFP_time'},
            'params': params }
td = load_pipeline(**LOAD_OPT)

#%% CLEANING DATA
CHANNELS = [['LFP_BIP7','LFP_BIP8'],['LFP_BIP8','LFP_BIP9'],['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]

CLEANING_OPT = {'combine_fields': {'fields': CHANNELS, 'method': 'subtract', 'remove_fields': True, 'save_to_params': 'params/data/data'},
                'remove_artefacts': {'fields': 'params/data/data', 'fs': 'params/data/data', 'method': 'amplitude', 'threshold': 300, 'signal_n': 1}}

td = cleaning_pipeline(td,**CLEANING_OPT)

#%% PREPROCESS DATA
# PREPROCESS_OPT = {'multitaper': {'fields': 'params/data/data', 'fs': 'params/data/data', 'wind_size': 0.25,'wind_step': 0.01, 'freq_start': 10, 'freq_stop': 100, 'NW': 4, 'adjust_target': 'params/event', 'inplace': False}}
# PREPROCESS_OPT = {'filter': {'kind': 'bandpass', 'fields': 'params/data/data', 'fs': 'params/data/data', 'f_min' : 50, 'f_max': 100, 'order':3}}
# PREPROCESS_OPT = {'filter': {'kind': 'sgolay', 'fields': 'params/data/data', 'fs': 'params/data/data', 'win_len': '3fs'}}

PREPROCESS_OPT = {'downsample': {'fields': 'params/data/data', 'fs': 'params/data/data', 
                                 'field_time': 'params/data/data', 'fs_down': 2000, 'adjust_target': 'params/event',
                                 'inplace': False, 'verbose': True}}

td_out = preprocess_pipeline(td,**PREPROCESS_OPT)

#%% Extract features
fields = 'params/data/data'
event_fields = 'params/event'
fs = 'params/data/data'
time_n = [7, 10] #10
feature_win_sec = [0.3, 0.5] # 0.5 seconds
dead_win_sec = [0.02] # 0.02 seconds
no_event_sec = [1] # 10 seconds

FEATURES_OPT = {'event_instant': {'fields': fields, 'event_fields': event_fields, 'fs': fs, 'time_n': time_n,
                                  'feature_win_sec': feature_win_sec, 'dead_win_sec': dead_win_sec,
                                  'no_event_sec': no_event_sec}}

td_features = features_pipeline(td_out,**FEATURES_OPT)

#%%
X = td_features[0]['features']
y = td_features[0]['labels']

#%% DECODER
CLASSIFIER =    {'selected': 'RF', \
                'GB': dict(trees=1000, learning_rate=0.01, depth=3, seed=333), \
                'RF': dict(trees=1000, depth=5, seed=333), \
                'rLDA': dict(r_coeff=0.3), \
                'LDA': dict()}
EXPORT_CLS = True

#%% Cross Validation
CV_PERFORM =   {'selected':'StratifiedShuffleSplit', \
                'False':None, \
                'StratifiedShuffleSplit': dict(test_ratio=0.2, folds=8, seed=0, export_result=True), \
                'LeaveOneOut': dict(export_result=False)}

#%% Test decoder


#%% Doceder metrics

