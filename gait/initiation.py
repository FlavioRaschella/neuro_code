#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:21:54 2020

@author: raschell

This function analyses the signals at initiation of gait in PD patients
"""

# Import data management libraries
import numpy as np

# Import loading functions
from loading_data import load_data_from_folder
# Import data processing
from td_utils import remove_fields, is_field, remove_all_fields_but, combine_dicts
from utils import flatten_list
from os import path

folder = ['/Volumes/MK_EPIOS/PD/from ED']
file_num = [[3,4,5]]
# file_num = [[6,7]]

file_format = '.mat'

signal_kin_time = 'KIN_time'
signal_kin = ['KIN_RightShoulder_P_y','KIN_RightShoulder_P_z','KIN_RightShoulder_P_x',
              'KIN_LeftShoulder_P_y','KIN_LeftShoulder_P_z','KIN_LeftShoulder_P_x',
              'KIN_RightUpLeg_P_y','KIN_RightUpLeg_P_z','KIN_RightUpLeg_P_x',
              'KIN_LeftUpLeg_P_y','KIN_LeftUpLeg_P_z','KIN_LeftUpLeg_P_x',
              'KIN_RightHand_P_y','KIN_RightHand_P_z','KIN_RightHand_P_x',
              'KIN_LeftHand_P_y','KIN_LeftHand_P_z','KIN_LeftHand_P_x',
              'KIN_RightFoot_P_y','KIN_RightFoot_P_z','KIN_RightFoot_P_x',
              'KIN_LeftFoot_P_y','KIN_LeftFoot_P_z','KIN_LeftFoot_P_x']
signal_emg_time = 'EMG_time'
signal_emg = ['EMG_LTA', 'EMG_LMG', 'EMG_LLG', 'EMG_LVL', 'EMG_LRF', 'EMG_RTA', 'EMG_RMG', 'EMG_RLG', 'EMG_RVL', 'EMG_RRF']
signal_lfp_time = 'LFP_time'
signal_lfp = [['LFP_BIP7','LFP_BIP8'],['LFP_BIP8','LFP_BIP9'],['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]

signal_to_use = [signal_kin_time] + [signal_emg_time] + [signal_lfp_time] + signal_kin + signal_emg + signal_lfp + ['File','Folder']
events_side = ['Right','Left']

#%% Load data  
# Load LFP
td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format)

# Remove fields from td
remove_all_fields_but(td, flatten_list(signal_to_use), exact_field =  True, inplace = True)

# Load gait events
td_gait = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format, pre_ext = '_gait_events_initation')

combine_dicts(td, td_gait, inplace = True)

#%% Plot EMG
import matplotlib.pyplot as plt

# Loop over the files
for td_tmp in td:
    fig, ax = plt.subplots(5,1)
    
    ax[0].plot(td_tmp['EMG_time'], td_tmp['EMG_LRF'],'k')
    ax[0].plot(td_tmp['EMG_time'], td_tmp['EMG_RRF'],'b')
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].set_title('EMG RF. L:k, R:b ')
    
    ax[1].plot(td_tmp['EMG_time'], td_tmp['EMG_LVL'],'k')
    ax[1].plot(td_tmp['EMG_time'], td_tmp['EMG_RVL'],'b')
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].set_title('EMG VL. L:k, R:b ')
    
    ax[2].plot(td_tmp['EMG_time'], td_tmp['EMG_LTA'],'k')
    ax[2].plot(td_tmp['EMG_time'], td_tmp['EMG_RTA'],'b')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].set_title('EMG TA. L:k, R:b ')
    
    ax[3].plot(td_tmp['EMG_time'], td_tmp['EMG_LMG'],'k')
    ax[3].plot(td_tmp['EMG_time'], td_tmp['EMG_RMG'],'b')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[3].set_title('EMG MG. L:k, R:b ')
    
    ax[4].plot(td_tmp['EMG_time'], td_tmp['EMG_LLG'],'k')
    ax[4].plot(td_tmp['EMG_time'], td_tmp['EMG_RLG'],'b')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[4].set_title('EMG LG. L:k, R:b ')
    
    plt.tight_layout()
    
    fig.savefig('{}_raw_EMG.pdf'.format(path.join(td_tmp['Folder'],td_tmp['File'])), bbox_inches='tight')
    
#%% Extract and process data

from filters import butter_lowpass_filter as lpf
from filters import butter_bandpass_filtfilt as bpf
from filters import envelope as env
from filters import downsample_signal
from power_estimation import hilbert_transform

dataset_len = len(td)

events_to_use = []
if 'Right' in events_side:
    events_to_use += ['RFS','RFO']
if 'Left' in events_side:
    events_to_use += ['LFS','LFO']

signal_n = len(signal_to_use)
if not is_field(td, signal_to_use) or not is_field(td, events_to_use):
    raise Exception('Missing fields in td list!')

# Decoder list
td_init = []

# Loop over the files
for iTd, td_tmp in enumerate(td):
    print('Preparing data in file {}: {}/{}'.format(td_tmp['File'], iTd+1, dataset_len))
    
    td_init_tmp = dict()
    
    td_init_tmp['File'] = td_tmp['File']
    # Collect kinematic data
    td_init_tmp[signal_kin_time] = td_tmp[signal_kin_time]
    td_init_tmp['KIN_Fs'] = np.round(1/(td_tmp[signal_kin_time][1]-td_tmp[signal_kin_time][0])).astype('int')
    for iSig, signal in enumerate(signal_kin):
        print('Processing KIN signal {}/{}'.format(iSig+1,len(signal_kin)))
        td_init_tmp[signal] = lpf(td_tmp[signal],10,td_init_tmp['KIN_Fs'])
    td_init_tmp['KIN_name'] = signal_kin
    td_init_tmp['KIN_time'] = td_tmp['KIN_time']
    
    # Collect emg data
    td_init_tmp[signal_emg_time] = td_tmp[signal_emg_time]
    td_init_tmp['EMG_Fs'] =  np.round(1/(td_tmp[signal_emg_time][1]-td_tmp[signal_emg_time][0])).astype('int')
    for iSig, signal in enumerate(signal_emg):
        print('Processing EMG signal {}/{}'.format(iSig+1,len(signal_emg)))
        td_init_tmp[signal] = env(td_tmp[signal],td_init_tmp['EMG_Fs'], lowcut = 4, highcut = 50, method = 'squared', order = 3)
    td_init_tmp['EMG_name'] = signal_emg
    td_init_tmp['EMG_time'] = td_tmp['EMG_time']
    
    # Collect lfp data LOW BETA
    signal_names = []
    td_init_tmp[signal_lfp_time] = td_tmp[signal_lfp_time]
    td_init_tmp['LFP_Fs'] = np.round(1/(td_tmp[signal_lfp_time][1]-td_tmp[signal_lfp_time][0])).astype('int')
    for iSig, signal in enumerate(signal_lfp):
        print('Processing LFP low beta signal {}/{}'.format(iSig+1,len(signal_lfp)))
        if type(signal) == list:
            # signal_tmp = bpf(np.array(td_tmp[signal[0]]) - np.array(td_tmp[signal[1]]), lowcut = 10, highcut = 20, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal[0]]) - np.array(td_tmp[signal[1]]), lowcut = 13, highcut = 23, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}-{}_lowBetaPow'.format(signal[0],signal[1]))
        else:
            # signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 10, highcut = 20, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 13, highcut = 23, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}_lowBetaPow'.format(signal))
        signal_tmp, Fs = downsample_signal(signal_tmp,td_init_tmp['LFP_Fs'],2000)
        td_init_tmp[signal_names[iSig]] = hilbert_transform(signal_tmp)
        
    td_init_tmp['LFP_lbp_name'] = signal_names
    
    # Collect lfp data HIGH BETA
    signal_names = []
    for iSig, signal in enumerate(signal_lfp):
        print('Processing LFP high beta signal {}/{}'.format(iSig+1,len(signal_lfp)))
        if type(signal) == list:
            # signal_tmp = bpf(np.array(td_tmp[signal[0]]) - np.array(td_tmp[signal[1]]), lowcut = 20, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal[0]]) - np.array(td_tmp[signal[1]]), lowcut = 28, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}-{}_highBetaPow'.format(signal[0],signal[1]))
        else:
            # signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 20, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 28, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}_highBetaPow'.format(signal))
        signal_tmp, Fs = downsample_signal(signal_tmp,td_init_tmp['LFP_Fs'],2000)
        td_init_tmp[signal_names[iSig]] = hilbert_transform(signal_tmp)
    
    td_init_tmp['LFP_hbp_name'] = signal_names
    td_init_tmp['LFP_Fs'] = Fs
    td_init_tmp['LFP_time'] = np.arange(0,td_init_tmp[signal_names[iSig]].shape[0]/Fs,1/Fs)
    td_tmp['LFP_time'][-1]
    
    td_init.append(td_init_tmp)    

#%% Save data
import pickle

pickle_out = open('/Volumes/MK_EPIOS/PD/from ED/td_3_4_5.pickle','wb')
pickle.dump([td, td_init], pickle_out)
pickle_out.close()

#%% Load data
import pickle

pickle_in = open('/Volumes/MK_EPIOS/PD/from ED/td_3_4_5.pickle',"rb")
td, td_init = pickle.load(pickle_in)
#%% Add Angles
from kinematics import compute_angle_3d

signal_angles = ['KIN_angle_leg_right','KIN_angle_leg_left',
                 'KIN_angle_trunk_right','KIN_angle_trunk_left',
                 'KIN_angle_arm_right','KIN_angle_arm_left']
signal_displace = ['KIN_Left_UpLeg-Foot_P_x','KIN_Left_UpLeg-Foot_P_z']


for iTd, td_tmp in enumerate(td_init):
    vect_foot_right = np.array([np.array(td_tmp['KIN_RightFoot_P_z'])-np.array(td_tmp['KIN_RightUpLeg_P_z']),
                                np.array(td_tmp['KIN_RightFoot_P_y'])-np.array(td_tmp['KIN_RightUpLeg_P_y']),
                                np.array(td_tmp['KIN_RightFoot_P_x'])-np.array(td_tmp['KIN_RightUpLeg_P_x'])]).T
    td_tmp[signal_angles[0]] = compute_angle_3d(vect_foot_right, np.tile([1,0,0],(vect_foot_right.shape[0],1)),'acos')-90

    vect_foot_left = np.array([np.array(td_tmp['KIN_LeftFoot_P_z'])-np.array(td_tmp['KIN_LeftUpLeg_P_z']),
                                np.array(td_tmp['KIN_LeftFoot_P_y'])-np.array(td_tmp['KIN_LeftUpLeg_P_y']),
                                np.array(td_tmp['KIN_LeftFoot_P_x'])-np.array(td_tmp['KIN_LeftUpLeg_P_x'])]).T
    td_tmp[signal_angles[1]] = compute_angle_3d(vect_foot_left, np.tile([1,0,0],(vect_foot_left.shape[0],1)),'acos')-90

    vect_trunk_right = np.array([np.array(td_tmp['KIN_RightShoulder_P_z'])-np.array(td_tmp['KIN_RightUpLeg_P_z']),
                                np.array(td_tmp['KIN_RightShoulder_P_y'])-np.array(td_tmp['KIN_RightUpLeg_P_y']),
                                np.array(td_tmp['KIN_RightShoulder_P_x'])-np.array(td_tmp['KIN_RightUpLeg_P_x'])]).T
    td_tmp[signal_angles[2]] = compute_angle_3d(vect_trunk_right, np.tile([1,0,0],(vect_trunk_right.shape[0],1)),'acos')-90

    vect_trunk_left = np.array([np.array(td_tmp['KIN_LeftShoulder_P_z'])-np.array(td_tmp['KIN_LeftUpLeg_P_z']),
                                np.array(td_tmp['KIN_LeftShoulder_P_y'])-np.array(td_tmp['KIN_LeftUpLeg_P_y']),
                                np.array(td_tmp['KIN_LeftShoulder_P_x'])-np.array(td_tmp['KIN_LeftUpLeg_P_x'])]).T
    td_tmp[signal_angles[3]] = compute_angle_3d(vect_trunk_left, np.tile([1,0,0],(vect_trunk_left.shape[0],1)),'acos')-90
    
    vect_arm_right = np.array([np.array(td_tmp['KIN_RightHand_P_z'])-np.array(td_tmp['KIN_RightShoulder_P_z']),
                               np.array(td_tmp['KIN_RightHand_P_y'])-np.array(td_tmp['KIN_RightShoulder_P_y']),
                               np.array(td_tmp['KIN_RightHand_P_x'])-np.array(td_tmp['KIN_RightShoulder_P_x'])]).T
    td_tmp[signal_angles[4]] = compute_angle_3d(vect_arm_right, np.tile([1,0,0],(vect_arm_right.shape[0],1)),'acos')-90

    vect_arm_left = np.array([np.array(td_tmp['KIN_LeftHand_P_z'])-np.array(td_tmp['KIN_LeftShoulder_P_z']),
                              np.array(td_tmp['KIN_LeftHand_P_y'])-np.array(td_tmp['KIN_LeftShoulder_P_y']),
                              np.array(td_tmp['KIN_LeftHand_P_x'])-np.array(td_tmp['KIN_LeftShoulder_P_x'])]).T
    td_tmp[signal_angles[5]] = compute_angle_3d(vect_arm_left, np.tile([1,0,0],(vect_arm_left.shape[0],1)),'acos')-90
    
    
    td_tmp[signal_displace[0]] = np.array(td_tmp['KIN_LeftUpLeg_P_x'])-np.array(td_tmp['KIN_LeftFoot_P_x'])
    td_tmp[signal_displace[1]] = np.array(td_tmp['KIN_LeftUpLeg_P_z'])-np.array(td_tmp['KIN_LeftFoot_P_z'])
    
    td_tmp['KIN_name'] = signal_angles + signal_displace + ['KIN_RightFoot_P_y','KIN_LeftFoot_P_y']


str2rem = ['KIN_RightShoulder_P_y','KIN_RightShoulder_P_z','KIN_RightShoulder_P_x',
          'KIN_LeftShoulder_P_y','KIN_LeftShoulder_P_z','KIN_LeftShoulder_P_x',
          'KIN_RightUpLeg_P_y','KIN_RightUpLeg_P_z','KIN_RightUpLeg_P_x',
          'KIN_LeftUpLeg_P_y','KIN_LeftUpLeg_P_z','KIN_LeftUpLeg_P_x',
          'KIN_RightHand_P_y','KIN_RightHand_P_z','KIN_RightHand_P_x',
          'KIN_LeftHand_P_y','KIN_LeftHand_P_z','KIN_LeftHand_P_x',
          'KIN_RightFoot_P_z','KIN_RightFoot_P_x',
          'KIN_LeftFoot_P_z','KIN_LeftFoot_P_x']
remove_fields(td_init, str2rem, exact_field = True, inplace = True)


#%% Get intervals of interest

# Set offset for expanding the intevals
offset_pre_sec = 2 # seconds

td_inteval = []
for iTd, td_tmp in enumerate(td):
    print('Preparing intervals in file {}: {}/{}'.format(td_tmp['File'], iTd+1, dataset_len))
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
                offset_pre_smp = np.round((offset_pre_sec*Fs)).astype('int')
                
                intervals_pre = np.hstack([np.array(td_tmp['RFO']-offset_pre_smp).reshape(len(td_tmp['RFO']),1),
                                       np.array(td_tmp['RFO']).reshape(len(td_tmp['RFS']),1)])/Fs
                intervals = np.hstack([np.array(td_tmp['RFO']).reshape(len(td_tmp['RFO']),1),
                                       np.array(td_tmp['RFS']).reshape(len(td_tmp['RFS']),1)])/Fs
            else: # events are in seconds
                intervals_pre = np.hstack([np.array(td_tmp['RFO']-offset_pre_sec).reshape(len(td_tmp['RFO']),1),
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
                offset_pre_smp = np.round((offset_pre_sec*Fs)).astype('int')
                
                intervals_pre = np.hstack([np.array(td_tmp['LFO']-offset_pre_smp).reshape(len(td_tmp['LFO']),1),
                                       np.array(td_tmp['LFO']).reshape(len(td_tmp['LFS']),1)])/Fs
                intervals = np.hstack([np.array(td_tmp['LFO']).reshape(len(td_tmp['LFO']),1),
                                       np.array(td_tmp['LFS']).reshape(len(td_tmp['LFS']),1)])/Fs
            else: # events are in seconds
                intervals_pre = np.hstack([np.array(td_tmp['LFO']-offset_pre_sec).reshape(len(td_tmp['LFO']),1),
                                       np.array(td_tmp['LFO']).reshape(len(td_tmp['LFS']),1)])
                intervals = np.hstack([np.array(td_tmp['LFO']).reshape(len(td_tmp['LFO']),1),
                                       np.array(td_tmp['LFS']).reshape(len(td_tmp['LFS']),1)])
    
        # Store intervals
        intervals_pre
        td_init_tmp[side] = dict()
        td_init_tmp[side]['intervals'] = intervals
        td_init_tmp[side]['intervals_pre'] = intervals_pre
    td_inteval.append(td_init_tmp)
        
combine_dicts(td_init, td_inteval, inplace = True)

#%% Extract the data for each interval

td_rt = []
td_lt = []
dataset_len = len(td_init)
# Loop over the files
for iTd, td_tmp in enumerate(td_init):
    print('Preparing file {}: {}/{}'.format(td_tmp['File'], iTd+1, dataset_len))
    
    # Set time interval
    for side in events_side:
        td_init_tmp = dict()
        # Store general data
        td_init_tmp['KIN_name'] = td_tmp['KIN_name']
        td_init_tmp['EMG_name'] = td_tmp['EMG_name']
        td_init_tmp['LFP_lbp_name'] = td_tmp['LFP_lbp_name']
        td_init_tmp['LFP_hbp_name'] = td_tmp['LFP_hbp_name']
        td_init_tmp['KIN_time'] = td_tmp['KIN_time']
        td_init_tmp['EMG_time'] = td_tmp['EMG_time']
        td_init_tmp['LFP_time'] = td_tmp['LFP_time']
        td_init_tmp['KIN_Fs'] = td_tmp['KIN_Fs']
        td_init_tmp['EMG_Fs'] = td_tmp['EMG_Fs']
        td_init_tmp['LFP_Fs'] = td_tmp['LFP_Fs']
        td_init_tmp['File'] = td_tmp['File']
        td_init_tmp['Folder'] = td_tmp['Folder']
        
        # Collect kinematic data
        interval_kin = [np.where(np.logical_and(np.array(td_tmp['KIN_time']) >= interval[0], np.array(td_tmp['KIN_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals']]
        td_init_tmp['KIN_interval'] = interval_kin
        for signal in td_tmp['KIN_name']:
            td_init_tmp[signal] = []
            for interval in interval_kin:
                td_init_tmp[signal].append(np.array(td_tmp[signal])[interval])
        
        # Collect kinematic data pre
        interval_kin = [np.where(np.logical_and(np.array(td_tmp['KIN_time']) >= interval[0], np.array(td_tmp['KIN_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals_pre']]
        td_init_tmp['KIN_interval_pre'] = interval_kin
        signal_name_pre = []
        for signal in td_tmp['KIN_name']:
            signal_name = signal+'_pre'
            signal_name_pre.append(signal_name)
            td_init_tmp[signal_name] = []
            for interval in interval_kin:
                td_init_tmp[signal_name].append(np.array(td_tmp[signal])[interval])
        td_init_tmp['KIN_name_pre'] = signal_name_pre
        
        # Collect emg data
        interval_emg = [np.where(np.logical_and(np.array(td_tmp['EMG_time']) >= interval[0], np.array(td_tmp['EMG_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals']]
        td_init_tmp['EMG_interval'] = interval_emg
        for signal in td_tmp['EMG_name']:
            td_init_tmp[signal] = []
            for interval in interval_emg:
                td_init_tmp[signal].append(np.array(td_tmp[signal])[interval])
           
        # Collect emg data pre
        interval_emg = [np.where(np.logical_and(np.array(td_tmp['EMG_time']) >= interval[0], np.array(td_tmp['EMG_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals_pre']]
        td_init_tmp['EMG_interval_pre'] = interval_emg
        signal_name_pre = []
        for signal in td_tmp['EMG_name']:
            signal_name = signal+'_pre'
            signal_name_pre.append(signal_name)
            td_init_tmp[signal_name] = []
            for interval in interval_emg:
                td_init_tmp[signal_name].append(np.array(td_tmp[signal])[interval])
        td_init_tmp['EMG_name_pre'] = signal_name_pre
        
        # Collect lfp data
        interval_lfp = [np.where(np.logical_and(np.array(td_tmp['LFP_time']) >= interval[0], np.array(td_tmp['LFP_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals']]
        td_init_tmp['LFP_interval'] = interval_lfp
        for signal in td_tmp['LFP_lbp_name']:
            td_init_tmp[signal] = []
            for interval in interval_lfp:
                td_init_tmp[signal].append(np.array(td_tmp[signal])[interval])
        
        # Collect hfp data
        for signal in td_tmp['LFP_hbp_name']:
            td_init_tmp[signal] = []
            for interval in interval_lfp:
                td_init_tmp[signal].append(np.array(td_tmp[signal])[interval])
                
        # Collect lfp data pre
        interval_lfp = [np.where(np.logical_and(np.array(td_tmp['LFP_time']) >= interval[0], np.array(td_tmp['LFP_time']) <= interval[1]))[0] for interval in td_tmp[side]['intervals_pre']]
        td_init_tmp['LFP_interval_pre'] = interval_lfp
        signal_name_pre = []
        for signal in td_tmp['LFP_lbp_name']:
            signal_name = signal+'_pre'
            signal_name_pre.append(signal_name)
            td_init_tmp[signal_name] = []
            for interval in interval_lfp:
                td_init_tmp[signal_name].append(np.array(td_tmp[signal])[interval])
        td_init_tmp['LFP_lbp_name_pre'] = signal_name_pre
        
        # Collect hfp data pre
        signal_name_pre = []
        for signal in td_tmp['LFP_hbp_name']:
            signal_name = signal+'_pre'
            signal_name_pre.append(signal_name)
            td_init_tmp[signal_name] = []
            for interval in interval_lfp:
                td_init_tmp[signal_name].append(np.array(td_tmp[signal])[interval])
        td_init_tmp['LFP_hbp_name_pre'] = signal_name_pre
        
        if side == 'Right':
            td_rt.append(td_init_tmp)
        else:
            td_lt.append(td_init_tmp)
    
#%% Plot data pre normalization
import matplotlib.pyplot as plt

def col_scale(n,shade = 'gray'):
    if shade == 'gray':
        col = np.array([np.linspace(0,0.8,n), np.linspace(0,0.8,n), np.linspace(0,0.8,n)]).T
    elif shade == 'r':
        col = np.array([np.linspace(0,0.8,n), np.linspace(0,0,n), np.linspace(0,0,n)]).T
    elif shade == 'g':
        col = np.array([np.linspace(0,0,n), np.linspace(0,0.8,n), np.linspace(0,0,n)]).T
    elif shade == 'b':
        col = np.array([np.linspace(0,0,n), np.linspace(0,0,n), np.linspace(0,0.8,n)]).T
    return col

def join_lists(list1,list2):
    if len(list1) != len(list2):
        raise Exception('ERROR: lists have different length.')
    
    lists = []
    for list1_el, list2_el in zip(list1,list2):
        lists.append(np.concatenate((list1_el,list2_el), axis = 0))
    return lists

lfp_lb_name = td_lt[0]['LFP_lbp_name']
lfp_lb_pre_name = td_lt[0]['LFP_lbp_name_pre']
lfp_hb_name = td_lt[0]['LFP_hbp_name']
lfp_hb_pre_name = td_lt[0]['LFP_hbp_name_pre']

# Left side
for lfp_l_name, lfp_h_name, lfp_l_pre_name, lfp_h_pre_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_pre_name,lfp_hb_pre_name):
    # break
    singal_KIN_foot = []
    singal_KIN_arm = []
    singal_KIN_trunk = []
    singal_KIN_leg = []
    
    singal_EMG_lg = []
    singal_EMG_vl = []
    singal_EMG_ta = []
    
    singal_LFP_lb = []
    singal_LFP_hb = []
    
    for td_lt_tmp in td_lt:
        # break
        # KIN
        singal_KIN_foot.extend( join_lists(td_lt_tmp['KIN_LeftFoot_P_y_pre'],td_lt_tmp['KIN_LeftFoot_P_y']) )
        singal_KIN_arm.extend(  join_lists(td_lt_tmp['KIN_angle_arm_left_pre'],td_lt_tmp['KIN_angle_arm_left']) )
        singal_KIN_trunk.extend(join_lists(td_lt_tmp['KIN_angle_trunk_left_pre'],td_lt_tmp['KIN_angle_trunk_left']) )
        singal_KIN_leg.extend(  join_lists(td_lt_tmp['KIN_angle_leg_left_pre'],td_lt_tmp['KIN_angle_leg_left']) )
        
        # EMG
        singal_EMG_lg.extend( join_lists(td_lt_tmp['EMG_LLG_pre'],td_lt_tmp['EMG_LLG']) )
        singal_EMG_vl.extend( join_lists(td_lt_tmp['EMG_LVL_pre'],td_lt_tmp['EMG_LVL']) )
        singal_EMG_ta.extend( join_lists(td_lt_tmp['EMG_LTA_pre'],td_lt_tmp['EMG_LTA']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_lt_tmp[lfp_l_pre_name],td_lt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_lt_tmp[lfp_h_pre_name],td_lt_tmp[lfp_h_name]) )
    
    col = col_scale(len(singal_KIN_foot))
    col_r = col_scale(len(singal_KIN_foot),'r')
    col_g = col_scale(len(singal_KIN_foot),'g')
    col_b = col_scale(len(singal_KIN_foot),'b')
    fig, ax = plt.subplots(5,1)
    
    for s, c in zip(singal_KIN_leg,col):
        ax[4].plot(s,color = c)
    ax[4].set_title('KIN ANGLE LEG')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_EMG_vl,col_r):
        ax[3].plot(s,color = c)
    for s, c in zip(singal_EMG_ta,col_g):
        ax[3].plot(s,color = c)
    for s, c in zip(singal_EMG_lg,col):
        ax[3].plot(s,color = c)
    ax[3].set_title('EMG')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_KIN_foot,col):
        ax[2].plot(s,color = c)
    ax[2].set_title('KIN Foot')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_LFP_lb,col):
        ax[1].plot(s,color = c)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_LFP_hb,col):
        ax[0].plot(s,color = c)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.tight_layout()


# Right side
for lfp_l_name, lfp_h_name, lfp_l_pre_name, lfp_h_pre_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_pre_name,lfp_hb_pre_name):
    # break
    singal_KIN_foot = []
    singal_KIN_arm = []
    singal_KIN_trunk = []
    singal_KIN_leg = []
    
    singal_EMG_lg = []
    singal_EMG_vl = []
    singal_EMG_ta = []
    
    singal_LFP_lb = []
    singal_LFP_hb = []
    
    for td_rt_tmp in td_rt:
        # break
        # KIN
        singal_KIN_foot.extend( join_lists(td_rt_tmp['KIN_RightFoot_P_y_pre'],td_rt_tmp['KIN_RightFoot_P_y']) )
        singal_KIN_arm.extend(  join_lists(td_rt_tmp['KIN_angle_arm_right_pre'],td_rt_tmp['KIN_angle_arm_right']) )
        singal_KIN_trunk.extend(join_lists(td_rt_tmp['KIN_angle_trunk_right_pre'],td_rt_tmp['KIN_angle_trunk_right']) )
        singal_KIN_leg.extend(  join_lists(td_rt_tmp['KIN_angle_leg_right_pre'],td_rt_tmp['KIN_angle_leg_right']) )
        
        # EMG
        singal_EMG_lg.extend( join_lists(td_rt_tmp['EMG_RLG_pre'],td_rt_tmp['EMG_RLG']) )
        singal_EMG_vl.extend( join_lists(td_rt_tmp['EMG_RVL_pre'],td_rt_tmp['EMG_RVL']) )
        singal_EMG_ta.extend( join_lists(td_lt_tmp['EMG_RTA_pre'],td_lt_tmp['EMG_RTA']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_rt_tmp[lfp_l_pre_name],td_rt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_rt_tmp[lfp_h_pre_name],td_rt_tmp[lfp_h_name]) )
    
    col = col_scale(len(singal_KIN_foot))
    fig, ax = plt.subplots(5,1)
    
    for s, c in zip(singal_KIN_leg,col):
        ax[4].plot(s,color = c)
    ax[4].set_title('KIN ANGLE LEG')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_EMG_vl,col_r):
        ax[3].plot(s,color = c)
    for s, c in zip(singal_EMG_ta,col_g):
        ax[3].plot(s,color = c)
    for s, c in zip(singal_EMG_lg,col):
        ax[3].plot(s,color = c)
    ax[3].set_title('EMG')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_KIN_foot,col):
        ax[2].plot(s,color = c)
    ax[2].set_title('KIN Foot')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_LFP_lb,col):
        ax[1].plot(s,color = c)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    for s, c in zip(singal_LFP_hb,col):
        ax[0].plot(s,color = c)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout()

#%% Normalise data
from scipy import interpolate

# Collect average init length
td_interval_rt = dict()
td_interval_lt = dict()
td_interval_rt['KIN'] = []
td_interval_rt['EMG'] = []
td_interval_rt['LFP'] = []
td_interval_lt['KIN'] = []
td_interval_lt['EMG'] = []
td_interval_lt['LFP'] = []
td_interval_rt['KIN_pre'] = []
td_interval_rt['EMG_pre'] = []
td_interval_rt['LFP_pre'] = []
td_interval_lt['KIN_pre'] = []
td_interval_lt['EMG_pre'] = []
td_interval_lt['LFP_pre'] = []

    
# Set intervals
for td_rt_tmp, td_lt_tmp in zip(td_rt,td_lt):
    # Get KIN average interval
    td_interval_rt['KIN'].extend(np.array([len(interval) for interval in td_rt_tmp['KIN_interval']]))
    td_interval_lt['KIN'].extend(np.array([len(interval) for interval in td_lt_tmp['KIN_interval']]))
    # Get EMG average interval
    td_interval_rt['EMG'].extend(np.array([len(interval) for interval in td_rt_tmp['EMG_interval']]))
    td_interval_lt['EMG'].extend(np.array([len(interval) for interval in td_lt_tmp['EMG_interval']]))
    # Get LFP average interval
    td_interval_rt['LFP'].extend(np.array([len(interval) for interval in td_rt_tmp['LFP_interval']]))
    td_interval_lt['LFP'].extend(np.array([len(interval) for interval in td_lt_tmp['LFP_interval']]))
    
    # Get KIN average interval pre
    td_interval_rt['KIN_pre'].extend(np.array([len(interval) for interval in td_rt_tmp['KIN_interval_pre']]))
    td_interval_lt['KIN_pre'].extend(np.array([len(interval) for interval in td_lt_tmp['KIN_interval_pre']]))
    # Get EMG average interval pre
    td_interval_rt['EMG_pre'].extend(np.array([len(interval) for interval in td_rt_tmp['EMG_interval_pre']]))
    td_interval_lt['EMG_pre'].extend(np.array([len(interval) for interval in td_lt_tmp['EMG_interval_pre']]))
    # Get LFP average interval pre
    td_interval_rt['LFP_pre'].extend(np.array([len(interval) for interval in td_rt_tmp['LFP_interval_pre']]))
    td_interval_lt['LFP_pre'].extend(np.array([len(interval) for interval in td_lt_tmp['LFP_interval_pre']]))
    
td_interval_rt['KIN'] = np.array(td_interval_rt['KIN']).mean().round().astype('int')
td_interval_rt['EMG'] = np.array(td_interval_rt['EMG']).mean().round().astype('int')
td_interval_rt['LFP'] = np.array(td_interval_rt['LFP']).mean().round().astype('int')
td_interval_rt['KIN_pre'] = np.array(td_interval_rt['KIN_pre']).mean().round().astype('int')
td_interval_rt['EMG_pre'] = np.array(td_interval_rt['EMG_pre']).mean().round().astype('int')
td_interval_rt['LFP_pre'] = np.array(td_interval_rt['LFP_pre']).mean().round().astype('int')

td_interval_lt['KIN'] = np.array(td_interval_lt['KIN']).mean().round().astype('int')
td_interval_lt['EMG'] = np.array(td_interval_lt['EMG']).mean().round().astype('int')
td_interval_lt['LFP'] = np.array(td_interval_lt['LFP']).mean().round().astype('int')
td_interval_lt['KIN_pre'] = np.array(td_interval_lt['KIN_pre']).mean().round().astype('int')
td_interval_lt['EMG_pre'] = np.array(td_interval_lt['EMG_pre']).mean().round().astype('int')
td_interval_lt['LFP_pre'] = np.array(td_interval_lt['LFP_pre']).mean().round().astype('int')    

# Normalise dataset
for td_rt_tmp, td_lt_tmp in zip(td_rt,td_lt):
    # Norm KIN
    signal_KIN_name = []
    for signal in td_rt_tmp['KIN_name']:
        signal_KIN_name.append(signal+'_inter')
        signal_KIN = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_KIN.append(f(np.linspace(0,len(sig),td_interval_rt['KIN'])))
        td_rt_tmp[signal+'_inter'] = signal_KIN
    td_rt_tmp['KIN_name_inter'] = signal_KIN_name
    
    signal_KIN_name = []
    for signal in td_lt_tmp['KIN_name']:
        signal_KIN_name.append(signal+'_inter')
        signal_KIN = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_KIN.append(f(np.linspace(0,len(sig),td_interval_lt['KIN'])))
        td_lt_tmp[signal+'_inter'] = signal_KIN
    td_lt_tmp['KIN_name_inter'] = signal_KIN_name
    
    # Norm EMG
    signal_EMG_name = []
    for signal in td_rt_tmp['EMG_name']:
        signal_EMG_name.append(signal+'_inter')
        signal_EMG = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_EMG.append(f(np.linspace(0,len(sig),td_interval_rt['EMG'])))
        td_rt_tmp[signal+'_inter'] = signal_EMG
    td_rt_tmp['EMG_name_inter'] = signal_EMG_name
    
    signal_EMG_name = []
    for signal in td_lt_tmp['EMG_name']:
        signal_EMG_name.append(signal+'_inter')
        signal_EMG = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_EMG.append(f(np.linspace(0,len(sig),td_interval_lt['EMG'])))
        td_lt_tmp[signal+'_inter'] = signal_EMG
    td_lt_tmp['EMG_name_inter'] = signal_EMG_name
    
    # Norm LFP
    signal_LFP_name = []
    for signal in td_rt_tmp['LFP_lbp_name']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_rt['LFP'])))
        td_rt_tmp[signal+'_inter'] = signal_LFP
    td_rt_tmp['LFP_lbp_name_inter'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_rt_tmp['LFP_hbp_name']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_rt['LFP'])))
        td_rt_tmp[signal+'_inter'] = signal_LFP
    td_rt_tmp['LFP_hbp_name_inter'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_lt_tmp['LFP_lbp_name']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_lt['LFP'])))
        td_lt_tmp[signal+'_inter'] = signal_LFP
    td_lt_tmp['LFP_lbp_name_inter'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_lt_tmp['LFP_hbp_name']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_lt['LFP'])))
        td_lt_tmp[signal+'_inter'] = signal_LFP
    td_lt_tmp['LFP_hbp_name_inter'] = signal_LFP_name
    
    
    
# Normalise dataset PRE
for td_rt_tmp, td_lt_tmp in zip(td_rt,td_lt):
    # Norm KIN
    signal_KIN_name = []
    for signal in td_rt_tmp['KIN_name_pre']:
        signal_KIN_name.append(signal+'_inter')
        signal_KIN = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_KIN.append(f(np.linspace(0,len(sig),td_interval_rt['KIN_pre'])))
        td_rt_tmp[signal+'_inter'] = signal_KIN
    td_rt_tmp['KIN_name_inter_pre'] = signal_KIN_name
    
    signal_KIN_name = []
    for signal in td_lt_tmp['KIN_name_pre']:
        signal_KIN_name.append(signal+'_inter')
        signal_KIN = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_KIN.append(f(np.linspace(0,len(sig),td_interval_lt['KIN_pre'])))
        td_lt_tmp[signal+'_inter'] = signal_KIN
    td_lt_tmp['KIN_name_inter_pre'] = signal_KIN_name
    
    # Norm EMG
    signal_EMG_name = []
    for signal in td_rt_tmp['EMG_name_pre']:
        signal_EMG_name.append(signal+'_inter')
        signal_EMG = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_EMG.append(f(np.linspace(0,len(sig),td_interval_rt['EMG_pre'])))
        td_rt_tmp[signal+'_inter'] = signal_EMG
    td_rt_tmp['EMG_name_inter_pre'] = signal_EMG_name
    
    signal_EMG_name = []
    for signal in td_lt_tmp['EMG_name_pre']:
        signal_EMG_name.append(signal+'_inter')
        signal_EMG = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_EMG.append(f(np.linspace(0,len(sig),td_interval_lt['EMG_pre'])))
        td_lt_tmp[signal+'_inter'] = signal_EMG
    td_lt_tmp['EMG_name_inter_pre'] = signal_EMG_name
    
    # Norm LFP
    signal_LFP_name = []
    for signal in td_rt_tmp['LFP_lbp_name_pre']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_rt['LFP_pre'])))
        td_rt_tmp[signal+'_inter'] = signal_LFP
    td_rt_tmp['LFP_lbp_name_inter_pre'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_rt_tmp['LFP_hbp_name_pre']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_rt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_rt['LFP_pre'])))
        td_rt_tmp[signal+'_inter'] = signal_LFP
    td_rt_tmp['LFP_hbp_name_inter_pre'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_lt_tmp['LFP_lbp_name_pre']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_lt['LFP_pre'])))
        td_lt_tmp[signal+'_inter'] = signal_LFP
    td_lt_tmp['LFP_lbp_name_inter_pre'] = signal_LFP_name
    
    signal_LFP_name = []
    for signal in td_lt_tmp['LFP_hbp_name_pre']:
        signal_LFP_name.append(signal+'_inter')
        signal_LFP = []
        for sig in td_lt_tmp[signal]:
            f = interpolate.interp1d(np.arange(len(sig)), sig, kind = 'linear', fill_value = 'extrapolate')
            signal_LFP.append(f(np.linspace(0,len(sig),td_interval_lt['LFP_pre'])))
        td_lt_tmp[signal+'_inter'] = signal_LFP
    td_lt_tmp['LFP_hbp_name_inter_pre'] = signal_LFP_name
    
#%% Plot data
from stats import confidence_interval

lfp_lb_name = td_lt[0]['LFP_lbp_name_inter']
lfp_lb_pre_name = td_lt[0]['LFP_lbp_name_inter_pre']
lfp_hb_name = td_lt[0]['LFP_hbp_name_inter']
lfp_hb_pre_name = td_lt[0]['LFP_hbp_name_inter_pre']

# Left side
for lfp_l_name, lfp_h_name, lfp_l_pre_name, lfp_h_pre_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_pre_name,lfp_hb_pre_name):
    # break
    singal_KIN_foot_r  = []
    singal_KIN_arm_r   = []
    singal_KIN_trunk_r = []
    singal_KIN_leg_r   = []
    singal_KIN_foot_l  = []
    singal_KIN_arm_l   = []
    singal_KIN_trunk_l = []
    singal_KIN_leg_l   = []
    
    singal_EMG_lg_r = []
    singal_EMG_vl_r = []
    singal_EMG_ta_r = []
    singal_EMG_rf_r = []
    singal_EMG_lg_l = []
    singal_EMG_vl_l = []
    singal_EMG_ta_l = []
    singal_EMG_rf_l = []
    
    singal_LFP_lb = []
    singal_LFP_hb = []
    
    for td_lt_tmp in td_lt:
        # break
        # KIN
        singal_KIN_foot_r.extend( join_lists(td_lt_tmp['KIN_RightFoot_P_y_pre_inter'],td_lt_tmp['KIN_RightFoot_P_y_inter']) )
        singal_KIN_arm_r.extend(  join_lists(td_lt_tmp['KIN_angle_arm_right_pre_inter'],td_lt_tmp['KIN_angle_arm_right_inter']) )
        singal_KIN_trunk_r.extend(join_lists(td_lt_tmp['KIN_angle_trunk_right_pre_inter'],td_lt_tmp['KIN_angle_trunk_right_inter']) )
        singal_KIN_leg_r.extend(  join_lists(td_lt_tmp['KIN_angle_leg_right_pre_inter'],td_lt_tmp['KIN_angle_leg_right_inter']) )
        
        singal_KIN_foot_l.extend( join_lists(td_lt_tmp['KIN_LeftFoot_P_y_pre_inter'],td_lt_tmp['KIN_LeftFoot_P_y_inter']) )
        singal_KIN_arm_l.extend(  join_lists(td_lt_tmp['KIN_angle_arm_left_pre_inter'],td_lt_tmp['KIN_angle_arm_left_inter']) )
        singal_KIN_trunk_l.extend(join_lists(td_lt_tmp['KIN_angle_trunk_left_pre_inter'],td_lt_tmp['KIN_angle_trunk_left_inter']) )
        singal_KIN_leg_l.extend(  join_lists(td_lt_tmp['KIN_angle_leg_left_pre_inter'],td_lt_tmp['KIN_angle_leg_left_inter']) )
        
        # EMG
        singal_EMG_lg_r.extend( join_lists(td_lt_tmp['EMG_RLG_pre_inter'],td_lt_tmp['EMG_RLG_inter']) )
        singal_EMG_vl_r.extend( join_lists(td_lt_tmp['EMG_RVL_pre_inter'],td_lt_tmp['EMG_RVL_inter']) )
        singal_EMG_ta_r.extend( join_lists(td_lt_tmp['EMG_RTA_pre_inter'],td_lt_tmp['EMG_RTA_inter']) )
        singal_EMG_rf_r.extend( join_lists(td_lt_tmp['EMG_RRF_pre_inter'],td_lt_tmp['EMG_RRF_inter']) )
        
        singal_EMG_lg_l.extend( join_lists(td_lt_tmp['EMG_LLG_pre_inter'],td_lt_tmp['EMG_LLG_inter']) )
        singal_EMG_vl_l.extend( join_lists(td_lt_tmp['EMG_LVL_pre_inter'],td_lt_tmp['EMG_LVL_inter']) )
        singal_EMG_ta_l.extend( join_lists(td_lt_tmp['EMG_LTA_pre_inter'],td_lt_tmp['EMG_LTA_inter']) )
        singal_EMG_rf_l.extend( join_lists(td_lt_tmp['EMG_LRF_pre_inter'],td_lt_tmp['EMG_LRF_inter']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_lt_tmp[lfp_l_pre_name],td_lt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_lt_tmp[lfp_h_pre_name],td_lt_tmp[lfp_h_name]) )
        
    
    col = col_scale(len(singal_KIN_foot))
    col_r = col_scale(len(singal_KIN_foot),'r')
    col_g = col_scale(len(singal_KIN_foot),'g')
    col_b = col_scale(len(singal_KIN_foot),'b')
    
    # LFP & EMGs
    fig, ax = plt.subplots(6,1)
    plt.suptitle('LEFT INIT: LFP & EMGs')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_hb).T)
    ax[0].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[0].plot(m, color='r', linewidth=2)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].vlines(td_interval_lt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_lb).T)
    ax[1].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[1].plot(m, color='r', linewidth=2)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].vlines(td_interval_lt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_vl_l).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[2].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_vl_r).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[2].plot(m, color='b', linewidth=2)
    ax[2].set_title('EMG LVL : k; EMG RVL : b')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].vlines(td_interval_lt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_rf_l).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[3].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_rf_r).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[3].plot(m, color='b', linewidth=2)
    ax[3].set_title('EMG LRF : k; EMG RRF : b')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[3].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_lg_l).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[4].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_lg_r).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[4].plot(m, color='b', linewidth=2)
    ax[4].set_title('EMG LLG : k; EMG RLG : b')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[4].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_ta_l).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[5].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_ta_r).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[5].plot(m, color='b', linewidth=2)
    ax[5].set_title('EMG LTA : k; EMG RTA : b')
    ax[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[5].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    plt.tight_layout()
    fig.savefig('{}_{}_KIN_LEFT.pdf'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File']), lfp_h_name[0:19]), bbox_inches='tight')
    
    # LFP & KIN
    fig, ax = plt.subplots(6,1)
    plt.suptitle('LEFT INIT: LFP & KIN')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_hb).T)
    ax[0].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[0].plot(m, color='r', linewidth=2)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].vlines(td_interval_lt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_lb).T)
    ax[1].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[1].plot(m, color='r', linewidth=2)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].vlines(td_interval_lt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_arm_l).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[2].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_arm_r).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[2].plot(m, color='b', linewidth=2)
    ax[2].set_title('LEFT ANG ARM: k; RIGHT ANG ARM: b')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].vlines(td_interval_lt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_trunk_l).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[3].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_trunk_r).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[3].plot(m, color='b', linewidth=2)
    ax[3].set_title('LEFT ANG TRUNK: k; RIGHT ANG TRUNK: b')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[3].vlines(td_interval_lt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_leg_l).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[4].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_leg_r).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[4].plot(m, color='b', linewidth=2)
    ax[4].set_title('LEFT ANG LEG: k; RIGHT ANG LEG: b')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[4].vlines(td_interval_lt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_foot_l).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[5].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_foot_r).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[5].plot(m, color='b', linewidth=2)
    ax[5].set_title('LEFT ANG FOOT: k; RIGHT ANG FOOT: b')
    ax[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[5].vlines(td_interval_lt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')

    plt.tight_layout()
    fig.savefig('{}_{}_EMG_LEFT.pdf'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File']), lfp_h_name[0:19]), bbox_inches='tight')


# Right side    
for lfp_l_name, lfp_h_name, lfp_l_pre_name, lfp_h_pre_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_pre_name,lfp_hb_pre_name):
    # break
    singal_KIN_foot_r  = []
    singal_KIN_arm_r   = []
    singal_KIN_trunk_r = []
    singal_KIN_leg_r   = []
    singal_KIN_foot_l  = []
    singal_KIN_arm_l   = []
    singal_KIN_trunk_l = []
    singal_KIN_leg_l   = []
    
    singal_EMG_lg_r = []
    singal_EMG_vl_r = []
    singal_EMG_ta_r = []
    singal_EMG_rf_r = []
    singal_EMG_lg_l = []
    singal_EMG_vl_l = []
    singal_EMG_ta_l = []
    singal_EMG_rf_l = []
    
    singal_LFP_lb = []
    singal_LFP_hb = []
    
    for td_rt_tmp in td_rt:
        # break
        # KIN
        singal_KIN_foot_r.extend( join_lists(td_rt_tmp['KIN_RightFoot_P_y_pre_inter'],td_rt_tmp['KIN_RightFoot_P_y_inter']) )
        singal_KIN_arm_r.extend(  join_lists(td_rt_tmp['KIN_angle_arm_right_pre_inter'],td_rt_tmp['KIN_angle_arm_right_inter']) )
        singal_KIN_trunk_r.extend(join_lists(td_rt_tmp['KIN_angle_trunk_right_pre_inter'],td_rt_tmp['KIN_angle_trunk_right_inter']) )
        singal_KIN_leg_r.extend(  join_lists(td_rt_tmp['KIN_angle_leg_right_pre_inter'],td_rt_tmp['KIN_angle_leg_right_inter']) )
        
        singal_KIN_foot_l.extend( join_lists(td_rt_tmp['KIN_LeftFoot_P_y_pre_inter'],td_rt_tmp['KIN_LeftFoot_P_y_inter']) )
        singal_KIN_arm_l.extend(  join_lists(td_rt_tmp['KIN_angle_arm_left_pre_inter'],td_rt_tmp['KIN_angle_arm_left_inter']) )
        singal_KIN_trunk_l.extend(join_lists(td_rt_tmp['KIN_angle_trunk_left_pre_inter'],td_rt_tmp['KIN_angle_trunk_left_inter']) )
        singal_KIN_leg_l.extend(  join_lists(td_rt_tmp['KIN_angle_leg_left_pre_inter'],td_rt_tmp['KIN_angle_leg_left_inter']) )
        
        # EMG
        singal_EMG_lg_r.extend( join_lists(td_rt_tmp['EMG_RLG_pre_inter'],td_rt_tmp['EMG_RLG_inter']) )
        singal_EMG_vl_r.extend( join_lists(td_rt_tmp['EMG_RVL_pre_inter'],td_rt_tmp['EMG_RVL_inter']) )
        singal_EMG_ta_r.extend( join_lists(td_rt_tmp['EMG_RTA_pre_inter'],td_rt_tmp['EMG_RTA_inter']) )
        singal_EMG_rf_r.extend( join_lists(td_rt_tmp['EMG_RRF_pre_inter'],td_rt_tmp['EMG_RRF_inter']) )
        
        singal_EMG_lg_l.extend( join_lists(td_rt_tmp['EMG_LLG_pre_inter'],td_rt_tmp['EMG_LLG_inter']) )
        singal_EMG_vl_l.extend( join_lists(td_rt_tmp['EMG_LVL_pre_inter'],td_rt_tmp['EMG_LVL_inter']) )
        singal_EMG_ta_l.extend( join_lists(td_rt_tmp['EMG_LTA_pre_inter'],td_rt_tmp['EMG_LTA_inter']) )
        singal_EMG_rf_l.extend( join_lists(td_rt_tmp['EMG_LRF_pre_inter'],td_rt_tmp['EMG_LRF_inter']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_rt_tmp[lfp_l_pre_name],td_rt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_rt_tmp[lfp_h_pre_name],td_rt_tmp[lfp_h_name]) )
    
    col = col_scale(len(singal_KIN_foot))
    col_r = col_scale(len(singal_KIN_foot),'r')
    col_g = col_scale(len(singal_KIN_foot),'g')
    col_b = col_scale(len(singal_KIN_foot),'b')
    
    # LFP & EMGs
    fig, ax = plt.subplots(6,1)
    plt.suptitle('RIGHT INIT: LFP & EMGs')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_hb).T)
    ax[0].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[0].plot(m, color='r', linewidth=2)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].vlines(td_interval_rt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_lb).T)
    ax[1].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[1].plot(m, color='r', linewidth=2)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].vlines(td_interval_rt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_vl_l).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[2].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_vl_r).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[2].plot(m, color='b', linewidth=2)
    ax[2].set_title('EMG LVL : k; EMG RVL : b')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_rf_l).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[3].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_rf_r).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[3].plot(m, color='b', linewidth=2)
    ax[3].set_title('EMG LRF : k; EMG RRF : b')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[3].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_lg_l).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[4].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_lg_r).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[4].plot(m, color='b', linewidth=2)
    ax[4].set_title('EMG LLG : k; EMG RLG : b')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[4].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_EMG_ta_l).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[5].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_EMG_ta_r).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[5].plot(m, color='b', linewidth=2)
    ax[5].set_title('EMG LTA : k; EMG RTA : b')
    ax[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[5].vlines(td_interval_rt['EMG_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    plt.tight_layout()
    fig.savefig('{}_{}_KIN_RIGHT.pdf'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File']), lfp_h_name[0:19]), bbox_inches='tight')
    
    # LFP & KIN
    fig, ax = plt.subplots(6,1)
    plt.suptitle('RIGHT INIT: LFP & KIN')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_hb).T)
    ax[0].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[0].plot(m, color='r', linewidth=2)
    ax[0].set_title(lfp_h_name)
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].vlines(td_interval_rt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw, up = confidence_interval(np.array(singal_LFP_lb).T)
    ax[1].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
    ax[1].plot(m, color='r', linewidth=2)
    ax[1].set_title(lfp_l_name)
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].vlines(td_interval_rt['LFP_pre'],dw.min(),up.max(),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_arm_l).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[2].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_arm_r).T)
    ax[2].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[2].plot(m, color='b', linewidth=2)
    ax[2].set_title('LEFT ANG ARM: k; RIGHT ANG ARM: b')
    ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].vlines(td_interval_rt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_trunk_l).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[3].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_trunk_r).T)
    ax[3].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[3].plot(m, color='b', linewidth=2)
    ax[3].set_title('LEFT ANG TRUNK: k; RIGHT ANG TRUNK: b')
    ax[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[3].vlines(td_interval_rt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_leg_l).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[4].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_leg_r).T)
    ax[4].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[4].plot(m, color='b', linewidth=2)
    ax[4].set_title('LEFT ANG LEG: k; RIGHT ANG LEG: b')
    ax[4].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[4].vlines(td_interval_rt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')
    
    m, dw1, up1 = confidence_interval(np.array(singal_KIN_foot_l).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw1, up1, alpha=0.1,color="k")
    ax[5].plot(m, color='k', linewidth=2)
    m, dw2, up2 = confidence_interval(np.array(singal_KIN_foot_r).T)
    ax[5].fill_between(np.arange(m.shape[0]),dw2, up2, alpha=0.1,color="b")
    ax[5].plot(m, color='b', linewidth=2)
    ax[5].set_title('LEFT ANG FOOT: k; RIGHT ANG FOOT: b')
    ax[5].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[5].vlines(td_interval_rt['KIN_pre'],min([dw1.min(),dw1.min()]),max([up1.max(),up2.max()]),'k')

    plt.tight_layout()
    fig.savefig('{}_{}_EMG_RIGHT.pdf'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File']), lfp_h_name[0:19]), bbox_inches='tight')


# EOF