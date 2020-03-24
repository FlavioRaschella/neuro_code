#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:21:54 2020

@author: raschell

This function analyses the signals at initiation of gait in PD patients
"""

# Import data management libraries
import numpy as np
# Import utils libs
from td_utils import remove_fields, is_field, remove_all_fields_but, combine_dicts, td_plot
from utils import flatten_list

# Saving lib
import pickle

# Processing libs
from processing import interpolate1D

def get_initiation(td, signals, events_side, **kwargs):
    '''
    This function helps selecting the gait events manually by displaying some preselected signals.
    This function saves a file with marked gait events.
    
    Parameters
    ----------
    td : dict / list of dict
        trialData structure containing the relevant signals to display.
    signals : str / list of str
        Signals to display for marking the events.
    events_side : str / list of str, optional
        Side of the initiation events to analise. The default value is ['Right','Left'].

    '''
    # Input variables
    events_side = ['Right','Left']
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'events_side':
            events_side = value
        elif key == 'verbose':
            verbose = value

    events = []
    if 'Right' in events_side:
        events += ['RFS','RFO']
    if 'Left' in events_side:
        events += ['LFS','LFO']
    
    if not is_field(td, signals) or not is_field(td, events):
        raise Exception('Missing fields in td list!')
        
    #%% Get intervals of interest
    
    # Set offset for expanding the intevals
    offset_a_sec = 1 # seconds
    
    td_inteval = []
    for iTd, td_tmp in enumerate(td):
        print('Preparing intervals in file {}: {}/{}'.format(td_tmp['File'], iTd+1, len(td)))
        td_init_tmp = dict()
        # Set time interval
        for side in events_side:
            if side == 'Right':
                if len(td_tmp['RFO']) != len(td_tmp['RFS']):
                    raise('RFS and RFO arrays have different length!')
                td_tmp['RFO'] = sorted(td_tmp['RFO'])
                td_tmp['RFS'] = sorted(td_tmp['RFS'])
                if td_tmp['RFO'][-1] > 5000: # events are in samples
                    Fs = 1/(td_tmp['KIN_time'][1] - td_tmp['KIN_time'][0])
                    offset_a_smp = np.round((offset_a_sec*Fs)).astype('int')
                    
                    intervals_a = np.hstack([np.array(td_tmp['RFO']-offset_a_smp).reshape(len(td_tmp['RFO']),1),
                                           np.array(td_tmp['RFO']).reshape(len(td_tmp['RFS']),1)])/Fs
                    intervals = np.hstack([np.array(td_tmp['RFO']).reshape(len(td_tmp['RFO']),1),
                                           np.array(td_tmp['RFS']).reshape(len(td_tmp['RFS']),1)])/Fs
                else: # events are in seconds
                    intervals_a = np.hstack([np.array(td_tmp['RFO']-offset_a_sec).reshape(len(td_tmp['RFO']),1),
                                           np.array(td_tmp['RFO']).reshape(len(td_tmp['RFS']),1)])
                    intervals = np.hstack([np.array(td_tmp['RFO']).reshape(len(td_tmp['RFO']),1),
                                           np.array(td_tmp['RFS']).reshape(len(td_tmp['RFS']),1)])
            else:
                if len(td_tmp['LFO']) != len(td_tmp['LFS']):
                    raise('LFS and LFO arrays have different length!')
                td_tmp['LFO'] = sorted(td_tmp['LFO'])
                td_tmp['LFS'] = sorted(td_tmp['LFS'])
                if td_tmp['RFO'][-1] > 5000: # events are in samples
                    Fs = 1/(td_tmp['KIN_time'][1] - td_tmp['KIN_time'][0])
                    offset_a_smp = np.round((offset_a_sec*Fs)).astype('int')
                    
                    intervals_a = np.hstack([np.array(td_tmp['LFO']-offset_a_smp).reshape(len(td_tmp['LFO']),1),
                                           np.array(td_tmp['LFO']).reshape(len(td_tmp['LFS']),1)])/Fs
                    intervals = np.hstack([np.array(td_tmp['LFO']).reshape(len(td_tmp['LFO']),1),
                                           np.array(td_tmp['LFS']).reshape(len(td_tmp['LFS']),1)])/Fs
                else: # events are in seconds
                    intervals_a = np.hstack([np.array(td_tmp['LFO']-offset_a_sec).reshape(len(td_tmp['LFO']),1),
                                           np.array(td_tmp['LFO']).reshape(len(td_tmp['LFS']),1)])
                    intervals = np.hstack([np.array(td_tmp['LFO']).reshape(len(td_tmp['LFO']),1),
                                           np.array(td_tmp['LFS']).reshape(len(td_tmp['LFS']),1)])
        
            # Store intervals
            td_init_tmp[side] = {'intervals': intervals, 'intervals_a': intervals_a}
        td_inteval.append(td_init_tmp)
            
    combine_dicts(td_init, td_inteval, inplace = True)
    
    #%% Extract the data for each interval
    
    td_rt = []
    td_lt = []
    
    data_info = [('KIN','KIN'),
                 ('EMG','EMG'),
                 ('LFP','LFP_lbp'),
                 ('LFP','LFP_hbp')]
    
    info2copy = ['KIN_name','EMG_name','LFP_lbp_name','LFP_hbp_name',
                 'KIN_time','EMG_time','LFP_time',
                 'KIN_Fs','EMG_Fs','LFP_Fs',
                 'File','Folder']
    # Loop over the files
    for iTd, td_tmp in enumerate(td_init):
        # break
        print('Preparing file {}: {}/{}'.format(td_tmp['File'], iTd+1, len(td_init)))
        
        # Set time interval
        for side in events_side:
            # break
            td_init_tmp = dict()
            # Store general data
            for info in info2copy:
                td_init_tmp[info] = td_tmp[info]
            
            # Collect kinematic data
            for info in data_info:
                # After event
                interval_data = [np.where(np.logical_and(np.array(td_tmp[info[0] + '_time']) >= interval[0], np.array(td_tmp[info[0] + '_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals']]
                td_init_tmp[info[0] + '_interval'] = interval_data
                for signal in td_tmp[info[1] + '_name']:
                    td_init_tmp[signal] = []
                    for interval in interval_data:
                        td_init_tmp[signal].append(np.array(td_tmp[signal])[interval])
                
                # Before event
                interval_data = [np.where(np.logical_and(np.array(td_tmp[info[0] + '_time']) >= interval[0], np.array(td_tmp[info[0] + '_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals_a']]
                td_init_tmp[info[0] + '_interval_a'] = interval_data
                signal_name_a = []
                for signal in td_tmp[info[1] + '_name']:
                    signal_name = signal+'_a'
                    signal_name_a.append(signal_name)
                    td_init_tmp[signal_name] = []
                    for interval in interval_data:
                        td_init_tmp[signal_name].append(np.array(td_tmp[signal])[interval])
                td_init_tmp[info[1] + '_name_a'] = signal_name_a
            
            if side == 'Right':
                td_rt.append(td_init_tmp)
            else:
                td_lt.append(td_init_tmp)
        
    #%% Normalise data
    
    data_info = [('KIN','KIN_interval'),
                 ('EMG','EMG_interval'),
                 ('LFP','LFP_interval'),
                 ('KIN_a','KIN_interval_a'),
                 ('EMG_a','EMG_interval_a'),
                 ('LFP_a','LFP_interval_a')]
    
    td_interval_rt = dict()
    td_interval_lt = dict()
    for info in data_info:
        td_interval_rt[info[0]] = []
        td_interval_lt[info[0]] = []
    
    # Set intervals
    for td_rt_tmp, td_lt_tmp in zip(td_rt,td_lt):
        for info in data_info:
            td_interval_rt[info[0]].extend(np.array([len(interval) for interval in td_rt_tmp[info[1]]]))
            td_interval_lt[info[0]].extend(np.array([len(interval) for interval in td_lt_tmp[info[1]]]))
    
    for info in data_info:
        td_interval_rt[info[0]] = np.array(td_interval_rt[info[0]]).mean().round().astype('int')
        td_interval_lt[info[0]] = np.array(td_interval_lt[info[0]]).mean().round().astype('int')
    
    # Normalise dataset
    for td_rt_tmp, td_lt_tmp in zip(td_rt,td_lt):
        
        data_info = [('KIN_name', 'KIN'),
                     ('EMG_name', 'EMG'),
                     ('LFP_lbp_name', 'LFP'),
                     ('LFP_hbp_name', 'LFP'),
                     ('KIN_name_a','KIN_a'),
                     ('EMG_name_a','EMG_a'),
                     ('LFP_lbp_name_a','LFP_a'),
                     ('LFP_hbp_name_a','LFP_a')]
        
        # Right
        for info in data_info:
            signal_name = []
            for signal in td_rt_tmp[info[0]]:
                signal_name.append(signal + '_nor')
                signal_new= []
                for sig in td_rt_tmp[signal]:
                    signal_new.append(interpolate1D(sig, td_interval_rt[info[1]]))
                td_rt_tmp[signal + '_nor'] = signal_new
            td_rt_tmp[info[0] + '_nor'] = signal_name
        
        # Left
        for info in data_info:
            signal_name = []
            for signal in td_lt_tmp[info[0]]:
                signal_name.append(signal + '_nor')
                signal_new= []
                for sig in td_lt_tmp[signal]:
                    signal_new.append(interpolate1D(sig, td_interval_lt[info[1]]))
                td_lt_tmp[signal + '_nor'] = signal_new
            td_lt_tmp[info[0] + '_nor'] = signal_name
    
    #%% Save data
    pickle_out = open(save_name + '.pickle','wb')
    pickle.dump([td, td_initiation, td_init, td_lt, td_rt], pickle_out)
    pickle_out.close()
    
    n_r_tot = 0
    n_l_tot = 0
    string_to_save = []
    string_to_save = ''
    for td_lt_tmp, td_rt_tmp in zip(td_lt, td_rt):
        filename = td_lt_tmp['File']
        n_r = len(td_lt_tmp['KIN_RightFoot_P_y_a_nor'])
        n_l = len(td_rt_tmp['KIN_RightFoot_P_y_a_nor'])
        n_r_tot += n_r
        n_l_tot += n_l
        
        string_to_save += 'File: {}\nRight events: {}\nLeft events: {}\n\n'.format(filename, n_r, n_l)
    string_to_save += 'All togehter\nRight events: {}\nLeft events: {}\n\n'.format(n_r_tot, n_l_tot)
    
    file1 = open(save_name + '.txt','w') 
    file1.write(string_to_save)
    file1.close()
    
    # Load data
    # pickle_in = open(save_name + '.pickle',"rb")
    # td, td_initiation, td_init, td_lt, td_rt = pickle.load(pickle_in)

# EOF