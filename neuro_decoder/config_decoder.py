#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:16:01 2020

@author: raschell
"""

#%% Libraries
# Import data management libraries
import numpy as np
# Import loading functions
from loading_data import load_data_from_folder
# Import data processing
from td_utils import remove_all_fields_but, combine_fields, convert_fields_to_numeric_array, load_and_organise_data

#%% Load data
DATA_PATH = ['/Volumes/MK_EPIOS/HUMANS/from ED']
DATA_FILE = [[3,4,5]]
FORMAT = '.mat'
TARGET_NAME = ['MANUAL_EV_RFO_time','MANUAL_EV_LFS_time','MANUAL_EV_LFO_time']
TARGET_NAME = 'MANUAL_EV_RFS_time'

LOAD_OPT = {'remove_all_fields_but': {'field': 'LFP'}, 'trigger_file': {'path': DATA_PATH, 'files': DATA_FILE, 'field': TARGET_NAME, 'file_format': '.mat', 'pre_ext': '_B33_MANUAL_gaitEvents'} }

td = load_and_organise_data(DATA_PATH, DATA_FILE, FORMAT, LOAD_OPT)

# Load
td_predic = load_data_from_folder(folder = DATA_PATH,file_num = DATA_FILE,file_format = '.mat')
# Remove fields from td
remove_all_fields_but(td_predic,['LFP'], exact_field = False, inplace = True)
# Load gait events
td_target = load_data_from_folder(folder = DATA_PATH,file_num = DATA_FILE,file_format = '.mat', pre_ext = '_B33_MANUAL_gaitEvents', fields = TARGET_NAME)
td_target = load_data_from_folder(folder = DATA_PATH,file_num = DATA_FILE,file_format = '.mat', pre_ext = '_B33_MANUAL_gaitEvents')
# Remove fields from td
remove_all_fields_but(td_target, TARGET_NAME, exact_field = False, inplace = True)
# Combine fields
combine_fields(td_predic, td_target, inplace = True)
# Convert fields to numeric array
convert_fields_to_numeric_array(td_predic, _fields = TARGET_NAME,
                                _vector_target_field = 'LFP_time', remove_selected_fields = True, inplace = True)

#%% Divide data in blocks
CHANNELS = [['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]



#%% PREPROCESS DATA
FILTERS = {}


#%% Extract features
FEATURES = {'selected':'PSD','PSD':dict(fmin=1, fmax=250, wlen=0.5, wstep=82, decim=4)}

#%% DECODER
CLASSIFIER =    {'selected': 'RF', \
                'GB': dict(trees=1000, learning_rate=0.01, depth=3, seed=666), \
                'RF': dict(trees=1000, depth=5, seed=666), \
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

