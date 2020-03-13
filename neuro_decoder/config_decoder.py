#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:16:01 2020

@author: raschell
"""

#%% Libraries
# Import data loading, processing
from td_utils import load_pipeline, cleaning_pipeline, preprocess_pipeline, td_plot

#%% Load data
DATA_PATH = ['/Volumes/MK_EPIOS/PD/from ED']
# DATA_FILE = [[3,4,5]]
DATA_FILE = [[3,4]]
FORMAT = '.mat'
# TARGET_NAME = ['MANUAL_EV_RFO_time','MANUAL_EV_LFS_time','MANUAL_EV_LFO_time']
TARGET_NAME = 'MANUAL_EV_RFS_time'

LOAD_OPT = {'remove_all_fields_but': {'fields': ['LFP_time','LFP_BIP7','LFP_BIP8','LFP_BIP9','LFP_BIP10','LFP_BIP11','LFP_BIP12']},
            'trigger_file': {'path': DATA_PATH, 'files': DATA_FILE, 'file_format': '.mat', 'fields': TARGET_NAME, 'pre_ext': '_B33_MANUAL_gaitEvents'},
            'convert_fields_to_numeric_array': {'fields': TARGET_NAME,'target_vector': 'LFP_time'},
            'params': {'signals': ['LFP_BIP7','LFP_BIP8','LFP_BIP9','LFP_BIP10','LFP_BIP11','LFP_BIP12'], 'time': 'LFP_time', 'Fs': 8192} }
td = load_pipeline(DATA_PATH, DATA_FILE, FORMAT, **LOAD_OPT)

#%% CLEANING DATA
CHANNELS = [['LFP_BIP7','LFP_BIP8'],['LFP_BIP8','LFP_BIP9'],['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]

CLEANING_OPT = {'combine_fields': {'fields': CHANNELS, 'method': 'subtract', 'remove_selected_fields': True, 'save_name_to_params': True},
                'remove_artefacts': {'method': 'amplitude', 'threshold': 300}}

td = cleaning_pipeline(td,**CLEANING_OPT)

#%% PREPROCESS DATA
PREPROCESS_OPT = {'multitaper': {'wind_size': 0.25,'wind_step': 0.01, 'freq_start': 10, 'freq_stop': 100, 'NW': 4}}
# PREPROCESS_OPT = {'filter': {'kind': 'bandpass', 'f_min' : 50, 'f_max': 100, 'order':3}}

td = preprocess_pipeline(td,**PREPROCESS_OPT)

#%% Extract features
FEATURES = {}

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

