#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:01:44 2020

@author: raschell

This function is used for the manual selection of the gait events
"""


# Files to load
folder = ['/Volumes/MK_EPIOS/PD/from ED']
file_num = [[6,7]]
file_format = '.mat'

# Choose the name you want for the events to select
events_to_select = ['RFS','RFO','LFS','LFO']
save_filename = 'initation'
signal_to_use = [['KIN_RightFoot_P_y','KIN_RightToe_P_y'],['KIN_LeftFoot_P_y','KIN_LeftToe_P_y']]

correct_data = True 

#%% Import libraries
# Import data management libraries
import numpy as np

# Import loading functions
from loading_data import load_data_from_folder

# Import data processing
from td_utils import remove_fields, is_field


#%% Load data  
# Load LFP
td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format)

# Remove fields from td
remove_fields(td,['EMG','EV','LFP'], exact_field = False, inplace = True)

#%% Find gait initiation

# Plot library
from filters import butter_lowpass_filtfilt
import matplotlib.pyplot as plt

if not is_field(td,signal_to_use):
    raise Exception('Missing fields in the dictionaries...')

dataset_len = len(td)
gait_events = []

signal_separation_sec = 30
if is_field(td[0],'KIN_Fs'):
    signal_separation_smp = np.round(signal_separation_sec*td[0]['KIN_Fs']).astype('int')
elif is_field(td[0],'KIN_time'):
    signal_separation_smp = np.round(signal_separation_sec * 1/(td[0]['KIN_time'][1] - td[0]['KIN_time'][0])).astype('int')
else:
    raise Exception('ERROR: Missing frequency information from the input dictionary!')

# Figure
fig, axs = plt.subplots(nrows = len(signal_to_use), ncols = 1, sharex=True)
# Loop over the dictionaries
for iTd, td_tmp in enumerate(td):
    # Gait event array for the current file
    gait_events_tmp = dict()
    for ev in events_to_select:
        gait_events_tmp[ev] = []
        
    # Set title
    fig.suptitle('File {}\nPress ESC to switch event type.'.format(td_tmp['File'],events_to_select[0]), fontsize=10)
    
    # Maximise figure
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    # Get data
    signal_name = []
    signal_data = []
    signal_ylim = []
    for iSig, signal in enumerate(signal_to_use):
        if type(signal) is list:
            signal_tmp = np.array([]).reshape(0,len(td_tmp[signal[0]]))
            signal_name_tmp = []
            for iSgl, sgl in enumerate(signal):
                signal_tmp = np.vstack([ signal_tmp, np.array(td_tmp[sgl]) ])
                signal_name_tmp.append(sgl)
        else:
            signal_tmp = np.array(td_tmp[signal])
            signal_name_tmp = signal
        
        signal_filt = butter_lowpass_filtfilt(data = signal_tmp, lowcut = 10, fs = 100, order = 4).T
        ylim_tmp = tuple([np.min(signal_filt), np.max(signal_filt)])
        
        signal_data.append(signal_filt)
        signal_name.append(signal_name_tmp)
        signal_ylim.append(ylim_tmp)
        
    intervals = np.append(np.arange(0,signal_data[0].shape[0],signal_separation_smp),signal_data[0].shape[0])
    intervals_range = np.arange(intervals.shape[0]-1)
    for iInt in intervals_range:
        for iSig, (signal,name,ylim) in enumerate(zip(signal_data,signal_name,signal_ylim)):
            axs[iSig].plot(signal[np.arange(intervals[iInt],intervals[iInt+1]), :])
            axs[iSig].set_ylim(ylim)
            axs[iSig].legend(loc='upper right', labels = name)
            if iSig != 0:
                axs[iSig].title.set_text(' + '.join(name))
            
        # Set the events
        for iEv, ev in enumerate(events_to_select):
            
            axs[0].title.set_text('Select {} event. '.format(events_to_select[iEv]) + ' + '.join(signal_name[0]))   
 
            fig.canvas.draw()           
            pts = plt.ginput(-1, timeout=0, show_clicks = True, mouse_add=1, mouse_pop=3)
            if len(pts) != 0:
                gait_events_tmp[ev].append(intervals[iInt] + np.round(np.asarray(pts)[:,0]).astype('int'))

            # if iEv+1 != len(events_to_select):
            #     axs[0].title.set_text('Select {} event. '.format(events_to_select[iEv+1]) + ' + '.join(signal_name))
                # fig.suptitle('File {}\n Select {} event. Press ESC to finish.'.format(td_tmp['File'],events_to_select[iEv+1]), fontsize=10)
    
        for iSig, signal in enumerate(signal_to_use):
            axs[iSig].cla()
    
    for ev in events_to_select:
        if len(gait_events_tmp[ev]) != 0:
            gait_events_tmp[ev] = np.concatenate(gait_events_tmp[ev])
        else:
            gait_events_tmp[ev] = np.array([])
    gait_events.append(gait_events_tmp)

plt.close()
#%% Correct the data using stickplot

if correct_data:
    import sys
    mutable_object = {}
    def press(input_key):
        sys.stdout.flush()
        # print('press:{}'.format(input_key.key))
        if input_key.key == 'right':
            output = +1
        elif input_key.key == 'left':
            output = -1
        elif input_key.key == 'escape':
            output = 2
        elif input_key.key == 'backspace':
            output = 3
        else:
            output = 0
        mutable_object['key'] = output

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
#%% Save the data
from scipy.io import savemat
from os import path

for td_tmp, gait_event in zip(td, gait_events):
    savemat(path.join(td_tmp['Folder'],td_tmp['File'][0:-4] + '_gait_events_' + save_filename + '.mat'), gait_event, appendmat=True)

#EOF