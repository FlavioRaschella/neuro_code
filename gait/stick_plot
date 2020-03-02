#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:47:21 2020

@author: raschell
"""
#%% Input information

# File to load
folder = '/Volumes/MK_EPIOS/PD/from ED'
file_num = [6]
file_format = '.mat'

# Choose the name you want for the events to select
leg_r = [['KIN_RightUpLeg_P_x','KIN_RightUpLeg_P_z','KIN_RightUpLeg_P_y'],
         ['KIN_RightLeg_P_x','KIN_RightLeg_P_z','KIN_RightLeg_P_y'],
         ['KIN_RightFoot_P_x','KIN_RightFoot_P_z','KIN_RightFoot_P_y'],
         ['KIN_RightToe_P_x','KIN_RightToe_P_z','KIN_RightToe_P_y']]

leg_l = [['KIN_LeftUpLeg_P_x','KIN_LeftUpLeg_P_z','KIN_LeftUpLeg_P_y'],
         ['KIN_LeftLeg_P_x','KIN_LeftLeg_P_z','KIN_LeftLeg_P_y'],
         ['KIN_LeftFoot_P_x','KIN_LeftFoot_P_z','KIN_LeftFoot_P_y'],
         ['KIN_LeftToe_P_x','KIN_LeftToe_P_z','KIN_LeftToe_P_y']]

arm_r = [['KIN_RightShoulder_P_x','KIN_RightShoulder_P_z','KIN_RightShoulder_P_y'],
         ['KIN_RightArm_P_x','KIN_RightArm_P_z','KIN_RightArm_P_y'],
         ['KIN_RightForeArm_P_x','KIN_RightForeArm_P_z','KIN_RightForeArm_P_y'],
         ['KIN_RightHand_P_x','KIN_RightHand_P_z','KIN_RightHand_P_y']]

arm_l = [['KIN_LeftShoulder_P_x','KIN_LeftShoulder_P_z','KIN_LeftShoulder_P_y'],
         ['KIN_LeftArm_P_x','KIN_LeftArm_P_z','KIN_LeftArm_P_y'],
         ['KIN_LeftForeArm_P_x','KIN_LeftForeArm_P_z','KIN_LeftForeArm_P_y'],
         ['KIN_LeftHand_P_x','KIN_LeftHand_P_z','KIN_LeftHand_P_y']]

trunk = [['KIN_Spine_P_x','KIN_Spine_P_z','KIN_Spine_P_y'],
         ['KIN_Hips_P_x','KIN_Hips_P_z','KIN_Hips_P_y']]

head = [['KIN_Neck_P_x','KIN_Neck_P_z','KIN_Neck_P_y'],
        ['KIN_Head_P_x','KIN_Head_P_z','KIN_Head_P_y']]

kin_info = {'leg_r': leg_r, 'leg_l': leg_l, 'arm_r': arm_r, 'arm_l': arm_l, 'trunk': trunk, 'head': head}

#%% Import libraries
# Import data management libraries
import numpy as np
# Import loading functions
from loading_data import load_data_from_folder
# Import data processing
from td_utils import remove_fields, is_field
# Flatten list
from utils import flatten_list

#%% Load data  
# Load LFP
td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format)

# Remove fields from td
remove_fields(td,['EMG','EV','LFP'], exact_field = False, inplace = True)

#%% Collect points in the space

signals = []
for k,v in kin_info.items():
    signals += v
signals = flatten_list(signals)

if not is_field(td, signals, True):
    raise Exception('ERROR: Some fields are missing from the trial data!!')

kin_var = dict()
for body_part, fields in kin_info.items():
    observation = []
    for field in fields:
        for iax, ax in enumerate(field):
            

#%% Correct the data using stickplot


fig, axs = plt.subplots(nrows = 1, ncols = 1)
for iTd, (gait_events_tmp, td_tmp) in enumerate(zip(gait_events,td)):
    for event_name in events_to_select:
        if 'R' in event_name:
            leg_variables_forward_names = ['KIN_RightUpLeg_P_z','KIN_RightLeg_P_z','KIN_RightFoot_P_z','KIN_RightToe_P_z','KIN_RightToeEnd_P_z']
            leg_variables_vertica_names = ['KIN_RightUpLeg_P_y','KIN_RightLeg_P_y','KIN_RightFoot_P_y','KIN_RightToe_P_y','KIN_RightToeEnd_P_y']
        elif 'L' in event_name:
            leg_variables_forward_names = ['KIN_LeftUpLeg_P_z','KIN_LeftLeg_P_z','KIN_LeftFoot_P_z','KIN_LeftToe_P_z','KIN_LeftToeEnd_P_z']
            leg_variables_vertica_names = ['KIN_LeftUpLeg_P_y','KIN_LeftLeg_P_y','KIN_LeftFoot_P_y','KIN_LeftToe_P_y','KIN_LeftToeEnd_P_y']
            
        # Check existance of variables in the dataset
        if not is_field(td,leg_variables_forward_names+leg_variables_vertica_names):
            raise Exception('Missing fields in the dictionaries...')
        
        X = np.array([]).reshape(len(td_tmp[leg_variables_forward_names[0]]),0)
        Y = np.array([]).reshape(len(td_tmp[leg_variables_vertica_names[0]]),0)
        for var_x, var_y in zip(leg_variables_forward_names,leg_variables_vertica_names):
            X = np.hstack([ X, np.array(td_tmp[var_x]).reshape(len(td_tmp[var_x]),1) ])
            Y = np.hstack([ Y, np.array(td_tmp[var_y]).reshape(len(td_tmp[var_y]),1) ])
        
        event_to_remove = []
        for iEv, event in enumerate(gait_events_tmp[event_name]):
            fig.suptitle('File {}/{}. Event {}. #{}/{}'.format(iTd+1,len(td),event_name,iEv+1,len(gait_events_tmp[event_name])))
            not_stop_loop = True
            while not_stop_loop:
                event_interval = np.arange(event-50,event+50)
                print(event)
                plt.plot(X[event_interval,:].T,Y[event_interval,:].T,'k')
                plt.plot(X[event,:],Y[event,:],'r')
                plt.axis('equal')
                plt.draw()
                plt.pause(0.1)
                
                fig.canvas.mpl_connect('key_press_event', press)
                _ = plt.waitforbuttonpress(0) #non specific key press!!!
                
                key = mutable_object['key']
                if key == 1:
                    event += 1
                    print('-->')
                elif key == -1:
                    event -= 1
                    print('<--')
                elif key == 2:
                    not_stop_loop = False
                    print('ESC')
                elif key == 3:
                    event_to_remove.append(iEv)
                    not_stop_loop = False
                    print('DEL')
                else:
                    print('Possible keys are: left, right, delete, escape.\nTry again...')
                
                plt.cla()
            
            gait_events_tmp[event_name][iEv] = event
        gait_events_tmp[event_name] = np.delete(gait_events_tmp[event_name], event_to_remove)
plt.close()
