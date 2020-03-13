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
from td_utils import remove_fields, is_field, remove_all_fields_but, combine_dicts, td_plot
from utils import flatten_list
from os import path
import pickle
import matplotlib.pyplot as plt

folder = ['/Volumes/MK_EPIOS/PD/Initiation/Data/Patient6_May2019/PostSx_Day2']
# folder = ['/Volumes/MK_EPIOS/PD/Initiation/Data/Patient5_Oct2018/PostSx_Day1']
file_num = [[3,4,5]]
# file_num = [[6,7]]
# file_num = [[15,16,17,18]]

save_name = path.join(folder[0],'td_' + '_'.join(list(map(str, file_num[0]))))

file_format = '.mat'

signal_kin_time = 'KIN_time'
signal_kin = ['KIN_RightShoulder_P_y','KIN_RightShoulder_P_z','KIN_RightShoulder_P_x',
              'KIN_LeftShoulder_P_y','KIN_LeftShoulder_P_z','KIN_LeftShoulder_P_x',
              'KIN_RightUpLeg_P_y','KIN_RightUpLeg_P_z','KIN_RightUpLeg_P_x',
              'KIN_LeftUpLeg_P_y','KIN_LeftUpLeg_P_z','KIN_LeftUpLeg_P_x',
              'KIN_RightLeg_P_y','KIN_RightLeg_P_z','KIN_RightLeg_P_x',
              'KIN_LeftLeg_P_y','KIN_LeftLeg_P_z','KIN_LeftLeg_P_x',
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

# Loop over the files
for td_tmp in td:
    td_plot(td_tmp,  y = [['EMG_LRF','EMG_RRF'],['EMG_LVL','EMG_RVL'],['EMG_LTA','EMG_RTA'],['EMG_LMG','EMG_RMG'],['EMG_LLG','EMG_RLG']], x = 'EMG_time',
            colours = ['k','b'], maximise = False , save = '{}_raw_EMG'.format(path.join(td_tmp['Folder'],td_tmp['File'][:-4])), save_format = ['pickle', 'pdf'])
    
#%% Extract and process data
plt.close('all')

from filters import butter_lowpass_filter as lpf
from filters import butter_bandpass_filtfilt as bpf
from filters import envelope as env
from filters import downsample_signal
from power_estimation import hilbert_transform

events_to_use = []
if 'Right' in events_side:
    events_to_use += ['RFS','RFO']
if 'Left' in events_side:
    events_to_use += ['LFS','LFO']

signal_n = len(signal_to_use)
if not is_field(td, signal_to_use) or not is_field(td, events_to_use):
    raise Exception('Missing fields in td list!')

# Decoder list
td_initiation = []

# Loop over the files
for iTd, td_tmp in enumerate(td):
    print('Preparing data in file {}: {}/{}'.format(td_tmp['File'], iTd+1, len(td)))
    
    td_init_tmp = dict()
    
    td_init_tmp['File'] = td_tmp['File']
    td_init_tmp['Folder'] = td_tmp['Folder']
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
            signal_names.append('{}-{}_lBP'.format(signal[0],signal[1]))
        else:
            # signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 10, highcut = 20, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 13, highcut = 23, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}_lBP'.format(signal))
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
            signal_names.append('{}-{}_hBP'.format(signal[0],signal[1]))
        else:
            # signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 20, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_tmp = bpf(np.array(td_tmp[signal]), lowcut = 28, highcut = 35, fs = td_init_tmp['LFP_Fs'], order=3)
            signal_names.append('{}_hBP'.format(signal))
        signal_tmp, Fs = downsample_signal(signal_tmp,td_init_tmp['LFP_Fs'],2000)
        td_init_tmp[signal_names[iSig]] = hilbert_transform(signal_tmp)
    
    td_init_tmp['LFP_hbp_name'] = signal_names
    td_init_tmp['LFP_Fs'] = Fs
    td_init_tmp['LFP_time'] = np.arange(0,td_init_tmp[signal_names[iSig]].shape[0]/Fs,1/Fs)
    td_tmp['LFP_time'][-1]
    
    td_initiation.append(td_init_tmp)    

#%% Save data
pickle_out = open(save_name + '.pickle','wb')
pickle.dump([td, td_initiation], pickle_out)
pickle_out.close()

#%% Load data
pickle_in = open(save_name + '.pickle',"rb")
td, td_initiation = pickle.load(pickle_in)
#%% Add Kinematic variables
from kinematics import compute_angle_3d
from utils import euclidean_distance

td_init = td_initiation.copy()

signal_angles = [('KIN_R_angle_leg'  ,'KIN_RightFoot'    , 'KIN_RightUpLeg'   ),
                 ('KIN_L_angle_leg'  ,'KIN_LeftFoot'     , 'KIN_LeftUpLeg'    ),
                 ('KIN_R_angle_knee' ,'KIN_RightUpLeg'   , 'KIN_RightLeg'   ,'KIN_RightFoot' , 'KIN_RightLeg'),
                 ('KIN_L_angle_knee' ,'KIN_LeftUpLeg'    ,  'KIN_LeftLeg'   ,'KIN_LeftFoot'  , 'KIN_LeftLeg'),
                 ('KIN_R_angle_trunk','KIN_RightShoulder', 'KIN_RightUpLeg'   ),
                 ('KIN_L_angle_trunk','KIN_LeftShoulder' , 'KIN_LeftUpLeg'    ),
                 ('KIN_R_angle_arm'  ,'KIN_RightHand'    , 'KIN_RightShoulder'),
                 ('KIN_L_angle_arm'  ,'KIN_LeftHand'     , 'KIN_LeftShoulder' )]

signal_displace = [('KIN_R_UpLeg-Foot_dist','KIN_RightUpLeg','KIN_RightFoot'),
                   ('KIN_L_UpLeg-Foot_dist','KIN_LeftUpLeg','KIN_LeftFoot')]

def get_vector(td, name_point1, name_point2):
    return np.array([np.array(td[name_point1 + '_P_z'])-np.array(td[name_point2 + '_P_z']),
                     np.array(td[name_point1 + '_P_y'])-np.array(td[name_point2 + '_P_y']),
                     np.array(td[name_point1 + '_P_x'])-np.array(td[name_point2 + '_P_x'])]).T
    

for iTd, td_tmp in enumerate(td_init):
    for sig in signal_angles:
        if len(sig) == 3:
            vect = get_vector(td_tmp, sig[1], sig[2])
            td_tmp[sig[0]] = compute_angle_3d(vect, np.tile([1,0,0],(vect.shape[0],1)),'acos')-90
        elif len(sig) == 5:
            vect_1 = get_vector(td_tmp, sig[1], sig[2])
            vect_2 = get_vector(td_tmp, sig[3], sig[4])
            td_tmp[sig[0]] = compute_angle_3d(vect_1, vect_2,'acos')
        else:
            raise Exception('ERROR: wrong number of signals in signal_angles!')
    
    for sig in signal_displace:
        vect_1 = np.array([np.array(td_tmp[sig[1] + '_P_x']),np.array(td_tmp[sig[1] + '_P_z'])])
        vect_2 = np.array([np.array(td_tmp[sig[2] + '_P_x']),np.array(td_tmp[sig[2] + '_P_z'])])
    
        td_tmp[sig[0]] = euclidean_distance(vect_1, vect_2)
    
    td_tmp['KIN_name'] = [sig[0] for sig in signal_angles] + [sig[0] for sig in signal_displace] + ['KIN_RightFoot_P_y','KIN_LeftFoot_P_y']

str2rem = ['KIN_RightShoulder_P_y','KIN_RightShoulder_P_z','KIN_RightShoulder_P_x',
          'KIN_LeftShoulder_P_y','KIN_LeftShoulder_P_z','KIN_LeftShoulder_P_x',
          'KIN_RightUpLeg_P_y','KIN_RightUpLeg_P_z','KIN_RightUpLeg_P_x',
          'KIN_LeftUpLeg_P_y','KIN_LeftUpLeg_P_z','KIN_LeftUpLeg_P_x',
          'KIN_RightLeg_P_y','KIN_RightLeg_P_z','KIN_RightLeg_P_x',
          'KIN_LeftLeg_P_y','KIN_LeftLeg_P_z','KIN_LeftLeg_P_x',
          'KIN_RightHand_P_y','KIN_RightHand_P_z','KIN_RightHand_P_x',
          'KIN_LeftHand_P_y','KIN_LeftHand_P_z','KIN_LeftHand_P_x',
          'KIN_RightFoot_P_z','KIN_RightFoot_P_x',
          'KIN_LeftFoot_P_z','KIN_LeftFoot_P_x']
remove_fields(td_init, str2rem, exact_field = True, inplace = True)


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
    
#%% Plot data pre normalization
'''
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
lfp_lb_a_name = td_lt[0]['LFP_lbp_name_a']
lfp_hb_name = td_lt[0]['LFP_hbp_name']
lfp_hb_a_name = td_lt[0]['LFP_hbp_name_a']

# Left side
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_a_name,lfp_hb_a_name):
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
        singal_KIN_foot.extend( join_lists(td_lt_tmp['KIN_LeftFoot_P_y_a'],td_lt_tmp['KIN_LeftFoot_P_y']) )
        singal_KIN_arm.extend(  join_lists(td_lt_tmp['KIN_angle_arm_left_a'],td_lt_tmp['KIN_angle_arm_left']) )
        singal_KIN_trunk.extend(join_lists(td_lt_tmp['KIN_angle_trunk_left_a'],td_lt_tmp['KIN_angle_trunk_left']) )
        singal_KIN_leg.extend(  join_lists(td_lt_tmp['KIN_angle_leg_left_a'],td_lt_tmp['KIN_angle_leg_left']) )
        
        # EMG
        singal_EMG_lg.extend( join_lists(td_lt_tmp['EMG_LLG_a'],td_lt_tmp['EMG_LLG']) )
        singal_EMG_vl.extend( join_lists(td_lt_tmp['EMG_LVL_a'],td_lt_tmp['EMG_LVL']) )
        singal_EMG_ta.extend( join_lists(td_lt_tmp['EMG_LTA_a'],td_lt_tmp['EMG_LTA']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_lt_tmp[lfp_l_a_name],td_lt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_lt_tmp[lfp_h_a_name],td_lt_tmp[lfp_h_name]) )
    
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
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_lb_name,lfp_hb_name,lfp_lb_a_name,lfp_hb_a_name):
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
        singal_KIN_foot.extend( join_lists(td_rt_tmp['KIN_RightFoot_P_y_a'],td_rt_tmp['KIN_RightFoot_P_y']) )
        singal_KIN_arm.extend(  join_lists(td_rt_tmp['KIN_angle_arm_right_a'],td_rt_tmp['KIN_angle_arm_right']) )
        singal_KIN_trunk.extend(join_lists(td_rt_tmp['KIN_angle_trunk_right_a'],td_rt_tmp['KIN_angle_trunk_right']) )
        singal_KIN_leg.extend(  join_lists(td_rt_tmp['KIN_angle_leg_right_a'],td_rt_tmp['KIN_angle_leg_right']) )
        
        # EMG
        singal_EMG_lg.extend( join_lists(td_rt_tmp['EMG_RLG_a'],td_rt_tmp['EMG_RLG']) )
        singal_EMG_vl.extend( join_lists(td_rt_tmp['EMG_RVL_a'],td_rt_tmp['EMG_RVL']) )
        singal_EMG_ta.extend( join_lists(td_lt_tmp['EMG_RTA_a'],td_lt_tmp['EMG_RTA']) )
        
        # LFP lowbeta
        singal_LFP_lb.extend( join_lists(td_rt_tmp[lfp_l_a_name],td_rt_tmp[lfp_l_name]) )
        
        # LFP highbeta
        singal_LFP_hb.extend( join_lists(td_rt_tmp[lfp_h_a_name],td_rt_tmp[lfp_h_name]) )
    
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
'''
#%% Normalise data
from processing import interpolate1D

# Collect average init length

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

#%% Load data
pickle_in = open(save_name + '.pickle',"rb")
td, td_initiation, td_init, td_lt, td_rt = pickle.load(pickle_in)

#%% Plot data

from stats import confidence_interval
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

lfp_R_lb_name = td_lt[0]['LFP_lbp_name_nor'][:2]
lfp_R_lb_a_name = td_lt[0]['LFP_lbp_name_a_nor'][:2]
lfp_R_hb_name = td_lt[0]['LFP_hbp_name_nor'][:2]
lfp_R_hb_a_name = td_lt[0]['LFP_hbp_name_a_nor'][:2]

lfp_L_lb_name = td_lt[0]['LFP_lbp_name_nor'][2:]
lfp_L_lb_a_name = td_lt[0]['LFP_lbp_name_a_nor'][2:]
lfp_L_hb_name = td_lt[0]['LFP_hbp_name_nor'][2:]
lfp_L_hb_a_name = td_lt[0]['LFP_hbp_name_a_nor'][2:]

data_plot_KIN_l = [('KIN_ARM'  ,'KIN_L_angle_arm_a_nor'  ,'KIN_L_angle_arm_nor'  ,'KIN_a'),
                   ('KIN_TRUNK','KIN_L_angle_trunk_a_nor','KIN_L_angle_trunk_nor','KIN_a'),
                   ('KIN_LEG'  ,'KIN_L_angle_leg_a_nor'  ,'KIN_L_angle_leg_nor'  ,'KIN_a'),
                   ('KIN_FOOT' ,'KIN_LeftFoot_P_y_a_nor' ,'KIN_LeftFoot_P_y_nor' ,'KIN_a'),
                   # ('KIN_disp' ,'KIN_L_UpLeg-Foot_dist_a_nor' ,'KIN_L_UpLeg-Foot_dist_nor' ,'KIN_a'),
                   ]
        
data_plot_KIN_r = [('KIN_ARM'  ,'KIN_R_angle_arm_a_nor'  ,'KIN_R_angle_arm_nor'  ,'KIN_a'),
                   ('KIN_TRUNK','KIN_R_angle_trunk_a_nor','KIN_R_angle_trunk_nor','KIN_a'),
                   ('KIN_LEG'  ,'KIN_R_angle_leg_a_nor'  ,'KIN_R_angle_leg_nor'  ,'KIN_a'),
                   ('KIN_FOOT' ,'KIN_RightFoot_P_y_a_nor','KIN_RightFoot_P_y_nor','KIN_a'),
                   # ('KIN_disp' ,'KIN_R_UpLeg-Foot_dist_a_nor' ,'KIN_R_UpLeg-Foot_dist_nor' ,'KIN_a'),
                   ]

data_plot_EMG_l = [('EMG_VL','EMG_LVL_a_nor','EMG_LVL_nor','EMG_a'),
                   ('EMG_RF','EMG_LRF_a_nor','EMG_LRF_nor','EMG_a'),
                   ('EMG_LG','EMG_LLG_a_nor','EMG_LLG_nor','EMG_a'),
                   ('EMG_TA','EMG_LTA_a_nor','EMG_LTA_nor','EMG_a')]
    
data_plot_EMG_r = [('EMG_VL','EMG_RVL_a_nor','EMG_RVL_nor','EMG_a'),
                   ('EMG_RF','EMG_RRF_a_nor','EMG_RRF_nor','EMG_a'),
                   ('EMG_LG','EMG_RLG_a_nor','EMG_RLG_nor','EMG_a'),
                   ('EMG_TA','EMG_RTA_a_nor','EMG_RTA_nor','EMG_a')]

data_plot_MIX_l = [('EMG_VL','EMG_LVL_a_nor','EMG_LVL_nor','EMG_a'),
                   ('KIN_KNEE','KIN_L_angle_knee_a_nor','KIN_L_angle_knee_nor','KIN_a'),
                   ('KIN_FOOT','KIN_LeftFoot_P_y_a_nor','KIN_LeftFoot_P_y_nor','KIN_a')]

data_plot_MIX_r = [('EMG_VL','EMG_RVL_a_nor','EMG_RVL_nor','EMG_a'),
                   ('KIN_KNEE','KIN_R_angle_knee_a_nor','KIN_R_angle_knee_nor','KIN_a'),
                   ('KIN_FOOT','KIN_RightFoot_P_y_a_nor','KIN_RightFoot_P_y_nor','KIN_a')]

# Left side singular trials
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_R_lb_name,lfp_R_hb_name,lfp_R_lb_a_name,lfp_R_hb_a_name):
    for td_tmp in td_lt:
        data_LFP = dict()
        data_KIN_l = dict()
        data_EMG_l = dict()
        data_MIX_l = dict()
        data_KIN_r = dict()
        data_EMG_r = dict()
        data_MIX_r = dict()
        
        data_plot_LFP = [('LFP_hb',lfp_h_a_name,lfp_h_name,'LFP_a'),
                         ('LFP_lb',lfp_l_a_name,lfp_l_name,'LFP_a')]
            
        for info_LFP in data_plot_LFP:
                data_LFP[info_LFP[0]] = join_lists(td_tmp[info_LFP[1]],td_tmp[info_LFP[2]]) 
            
        for info_KIN_l, info_KIN_r in zip(data_plot_KIN_l,data_plot_KIN_r):
                data_KIN_l[info_KIN_l[0]] = join_lists(td_tmp[info_KIN_l[1]],td_tmp[info_KIN_l[2]])
                data_KIN_r[info_KIN_r[0]] = join_lists(td_tmp[info_KIN_r[1]],td_tmp[info_KIN_r[2]])
                
        for info_EMG_l, info_EMG_r in zip(data_plot_EMG_l,data_plot_EMG_r):
                data_EMG_l[info_EMG_l[0]] = join_lists(td_tmp[info_EMG_l[1]],td_tmp[info_EMG_l[2]])
                data_EMG_r[info_EMG_r[0]] = join_lists(td_tmp[info_EMG_r[1]],td_tmp[info_EMG_r[2]])
                
        for info_MIX_l, info_MIX_r in zip(data_plot_MIX_l,data_plot_MIX_r):
                data_MIX_l[info_MIX_l[0]] = join_lists(td_tmp[info_MIX_l[1]],td_tmp[info_MIX_l[2]])
                data_MIX_r[info_MIX_r[0]] = join_lists(td_tmp[info_MIX_r[1]],td_tmp[info_MIX_r[2]])
        
        # LFP & EMGs
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_EMG_l),1)
        plt.suptitle('LEFT INIT: LFP & EMGs')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_EMG_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_EMG_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_EMG_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_EMG_LEFT'.format(path.join(td_tmp['Folder'],td_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')
        
        
        # LFP & KINs
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_KIN_l),1)
        plt.suptitle('LEFT INIT: LFP & KINs')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_KIN_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_KIN_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_KIN_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_KIN_LEFT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')
        
        # MIX
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_MIX_l),1)
        plt.suptitle('LEFT INIT: EMG & KIN')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_MIX_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_MIX_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_MIX_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_EMG_KIN_LEFT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')

# Left side all trials
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_R_lb_name,lfp_R_hb_name,lfp_R_lb_a_name,lfp_R_hb_a_name):
    # break
    data_LFP = dict()
    data_KIN_l = dict()
    data_EMG_l = dict()
    data_MIX_l = dict()
    data_KIN_r = dict()
    data_EMG_r = dict()
    data_MIX_r = dict()
    files = []
    
    data_plot_LFP = [('LFP_hb',lfp_h_a_name,lfp_h_name,'LFP_a'),
                     ('LFP_lb',lfp_l_a_name,lfp_l_name,'LFP_a')]
    
    for info_LFP in data_plot_LFP:
        data_LFP[info_LFP[0]] = []
    for info_KIN in data_plot_KIN_l:
        data_KIN_l[info_KIN[0]] = []
        data_KIN_r[info_KIN[0]] = []
    for info_EMG in data_plot_EMG_l:
        data_EMG_l[info_EMG[0]] = []
        data_EMG_r[info_EMG[0]] = []
    for info_MIX in data_plot_MIX_l:
        data_MIX_l[info_MIX[0]] = []
        data_MIX_r[info_MIX[0]] = []
        
    for info_LFP in data_plot_LFP:
        for td_tmp in td_lt:
            data_LFP[info_LFP[0]].extend( join_lists(td_tmp[info_LFP[1]],td_tmp[info_LFP[2]]) )
        
    for info_KIN_l, info_KIN_r in zip(data_plot_KIN_l,data_plot_KIN_r):
        for td_tmp in td_lt:
            data_KIN_l[info_KIN_l[0]].extend( join_lists(td_tmp[info_KIN_l[1]],td_tmp[info_KIN_l[2]]) )
            data_KIN_r[info_KIN_r[0]].extend( join_lists(td_tmp[info_KIN_r[1]],td_tmp[info_KIN_r[2]]) )
            
    for info_EMG_l, info_EMG_r in zip(data_plot_EMG_l,data_plot_EMG_r):
        for td_tmp in td_lt:
            data_EMG_l[info_EMG_l[0]].extend( join_lists(td_tmp[info_EMG_l[1]],td_tmp[info_EMG_l[2]]) )
            data_EMG_r[info_EMG_r[0]].extend( join_lists(td_tmp[info_EMG_r[1]],td_tmp[info_EMG_r[2]]) )
            
    for info_MIX_l, info_MIX_r in zip(data_plot_MIX_l,data_plot_MIX_r):
        for td_tmp in td_lt:
            data_MIX_l[info_MIX_l[0]].extend( join_lists(td_tmp[info_MIX_l[1]],td_tmp[info_MIX_l[2]]) )
            data_MIX_r[info_MIX_r[0]].extend( join_lists(td_tmp[info_MIX_r[1]],td_tmp[info_MIX_r[2]]) )
    
    # LFP & EMGs
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_EMG_l),1)
    plt.suptitle('LEFT INIT: LFP & EMGs')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_EMG_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_EMG_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_EMG_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_EMG_LEFT'.format(path.join(td_tmp['Folder'],td_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')
    
    
    # LFP & KINs
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_KIN_l),1)
    plt.suptitle('LEFT INIT: LFP & KINs')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_KIN_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_KIN_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_KIN_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_KIN_LEFT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')
    
    # MIX
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_MIX_l),1)
    plt.suptitle('LEFT INIT: EMG & KIN')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_MIX_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_MIX_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_MIX_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_EMG_KIN_LEFT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')

#%% Right side singular trial
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_L_lb_name,lfp_L_hb_name,lfp_L_lb_a_name,lfp_L_hb_a_name):
    for td_tmp in td_rt:
        data_LFP = dict()
        data_KIN_l = dict()
        data_EMG_l = dict()
        data_MIX_l = dict()
        data_KIN_r = dict()
        data_EMG_r = dict()
        data_MIX_r = dict()
        
        data_plot_LFP = [('LFP_hb',lfp_h_a_name,lfp_h_name,'LFP_a'),
                         ('LFP_lb',lfp_l_a_name,lfp_l_name,'LFP_a')]
            
        for info_LFP in data_plot_LFP:
                data_LFP[info_LFP[0]] = join_lists(td_tmp[info_LFP[1]],td_tmp[info_LFP[2]]) 
            
        for info_KIN_l, info_KIN_r in zip(data_plot_KIN_l,data_plot_KIN_r):
                data_KIN_l[info_KIN_l[0]] = join_lists(td_tmp[info_KIN_l[1]],td_tmp[info_KIN_l[2]])
                data_KIN_r[info_KIN_r[0]] = join_lists(td_tmp[info_KIN_r[1]],td_tmp[info_KIN_r[2]])
                
        for info_EMG_l, info_EMG_r in zip(data_plot_EMG_l,data_plot_EMG_r):
                data_EMG_l[info_EMG_l[0]] = join_lists(td_tmp[info_EMG_l[1]],td_tmp[info_EMG_l[2]])
                data_EMG_r[info_EMG_r[0]] = join_lists(td_tmp[info_EMG_r[1]],td_tmp[info_EMG_r[2]])
                
        for info_MIX_l, info_MIX_r in zip(data_plot_MIX_l,data_plot_MIX_r):
                data_MIX_l[info_MIX_l[0]] = join_lists(td_tmp[info_MIX_l[1]],td_tmp[info_MIX_l[2]])
                data_MIX_r[info_MIX_r[0]] = join_lists(td_tmp[info_MIX_r[1]],td_tmp[info_MIX_r[2]])
        
        # LFP & EMGs
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_EMG_l),1)
        plt.suptitle('RIGHT INIT: LFP & EMGs')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_EMG_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_EMG_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_EMG_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_EMG_RIGHT'.format(path.join(td_tmp['Folder'],td_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')
        
        
        # LFP & KINs
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_KIN_l),1)
        plt.suptitle('RIGHT INIT: LFP & KINs')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_KIN_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_KIN_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_KIN_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_KIN_RIGHT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')
        
        # MIX
        fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_MIX_l),1)
        plt.suptitle('RIGHT INIT: EMG & KIN')
        
        for iCount, info in enumerate(data_plot_LFP):
            m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
            ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
            ax[iCount].plot(m, color='r', linewidth=2)
            ax[iCount].set_title(info[1])
            ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
        
        for iCount, info in enumerate(data_plot_MIX_l):
            m_l, dw_l, up_l = confidence_interval(np.array(data_MIX_l[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
            ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
            m_r, dw_r, up_r = confidence_interval(np.array(data_MIX_r[info[0]]).T)
            ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
            ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
            
            ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
            ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
        
        plt.tight_layout()
        # Save figure
        fig_name = '{}_{}_EMG_KIN_RIGHT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-4]), lfp_h_name[0:19])
        pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
        fig.savefig(fig_name + '.pdf', bbox_inches='tight')

# Right side all trials
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_L_lb_name,lfp_L_hb_name,lfp_L_lb_a_name,lfp_L_hb_a_name):
    # break
    data_LFP = dict()
    data_KIN_l = dict()
    data_EMG_l = dict()
    data_MIX_l = dict()
    data_KIN_r = dict()
    data_EMG_r = dict()
    data_MIX_r = dict()
    files = []
    
    data_plot_LFP = [('LFP_hb',lfp_h_a_name,lfp_h_name,'LFP_a'),
                     ('LFP_lb',lfp_l_a_name,lfp_l_name,'LFP_a')]
    
    for info_LFP in data_plot_LFP:
        data_LFP[info_LFP[0]] = []
    for info_KIN in data_plot_KIN_l:
        data_KIN_l[info_KIN[0]] = []
        data_KIN_r[info_KIN[0]] = []
    for info_EMG in data_plot_EMG_l:
        data_EMG_l[info_EMG[0]] = []
        data_EMG_r[info_EMG[0]] = []
    for info_MIX in data_plot_MIX_l:
        data_MIX_l[info_MIX[0]] = []
        data_MIX_r[info_MIX[0]] = []
        
    for info_LFP in data_plot_LFP:
        for td_tmp in td_rt:
            data_LFP[info_LFP[0]].extend( join_lists(td_tmp[info_LFP[1]],td_tmp[info_LFP[2]]) )
        
    for info_KIN_l, info_KIN_r in zip(data_plot_KIN_l,data_plot_KIN_r):
        for td_tmp in td_rt:
            data_KIN_l[info_KIN_l[0]].extend( join_lists(td_tmp[info_KIN_l[1]],td_tmp[info_KIN_l[2]]) )
            data_KIN_r[info_KIN_r[0]].extend( join_lists(td_tmp[info_KIN_r[1]],td_tmp[info_KIN_r[2]]) )
            
    for info_EMG_l, info_EMG_r in zip(data_plot_EMG_l,data_plot_EMG_r):
        for td_tmp in td_rt:
            data_EMG_l[info_EMG_l[0]].extend( join_lists(td_tmp[info_EMG_l[1]],td_tmp[info_EMG_l[2]]) )
            data_EMG_r[info_EMG_r[0]].extend( join_lists(td_tmp[info_EMG_r[1]],td_tmp[info_EMG_r[2]]) )
            
    for info_MIX_l, info_MIX_r in zip(data_plot_MIX_l,data_plot_MIX_r):
        for td_tmp in td_rt:
            data_MIX_l[info_MIX_l[0]].extend( join_lists(td_tmp[info_MIX_l[1]],td_tmp[info_MIX_l[2]]) )
            data_MIX_r[info_MIX_r[0]].extend( join_lists(td_tmp[info_MIX_r[1]],td_tmp[info_MIX_r[2]]) )
    
    # LFP & EMGs
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_EMG_l),1)
    plt.suptitle('RIGHT INIT: LFP & EMGs')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_EMG_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_EMG_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_EMG_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_EMG_RIGHT'.format(path.join(td_tmp['Folder'],td_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')
    
    
    # LFP & KINs
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_KIN_l),1)
    plt.suptitle('RIGHT INIT: LFP & KINs')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_KIN_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_KIN_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_KIN_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_KIN_RIGHT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')
    
    # MIX
    fig, ax = plt.subplots(len(data_plot_LFP) + len(data_plot_MIX_l),1)
    plt.suptitle('RIGHT INIT: EMG & KIN')
    
    for iCount, info in enumerate(data_plot_LFP):
        m, dw, up = confidence_interval(np.array(data_LFP[info[0]]).T)
        ax[iCount].fill_between(np.arange(m.shape[0]),dw, up, alpha=0.1,color="r")
        ax[iCount].plot(m, color='r', linewidth=2)
        ax[iCount].set_title(info[1])
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[iCount].vlines(td_interval_lt[info[3]],dw.min(),up.max(),'k')
    
    for iCount, info in enumerate(data_plot_MIX_l):
        m_l, dw_l, up_l = confidence_interval(np.array(data_MIX_l[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_l.shape[0]),dw_l, up_l, alpha=0.1,color="k")
        ax[len(data_plot_LFP)+iCount].plot(m_l, color='k', linewidth=2)
        m_r, dw_r, up_r = confidence_interval(np.array(data_MIX_r[info[0]]).T)
        ax[len(data_plot_LFP)+iCount].fill_between(np.arange(m_r.shape[0]),dw_r, up_r, alpha=0.1,color="b")
        ax[len(data_plot_LFP)+iCount].plot(m_r, color='b', linewidth=2)
        
        ax[len(data_plot_LFP)+iCount].set_title(info[0] + '. Left: k; Right : b')
        ax[len(data_plot_LFP)+iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax[len(data_plot_LFP)+iCount].vlines(td_interval_lt[info[3]],min([dw_l.min(),dw_r.min()]),max([up_l.max(),up_r.max()]),'k')
    
    plt.tight_layout()
    # Save figure
    fig_name = '{}_{}_EMG_KIN_RIGHT'.format(path.join(td_lt_tmp['Folder'],td_lt_tmp['File'][:-6] + '_' + '_'.join(files)), lfp_h_name[0:19])
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')

# EOF