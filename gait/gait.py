#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:01:44 2020

@author: raschell

This function is used for the manual selection of the gait events
"""

# Import numpy
import numpy as np
import sys
# Loading lib
from loading_data import load_data_from_folder
# Plot library
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Filt library
from filters import butter_lowpass_filtfilt
# Import data processing libs
from td_utils import is_field, td_subfield, combine_dicts
from processing import interpolate1D
# Import utils
from utils import flatten_list, find_substring_indexes, copy_dict, find_values
# Save path
from scipy.io import savemat
import pickle

# =============================================================================
# Mark events
# =============================================================================
def gait_event_manual(td, signals, events, fs, **kwargs):
    '''
    This function helps selecting the gait events manually by displaying some 
    preselected signals. At the end, it saves a file with the marked gait events.
    
    Parameters
    ----------
    td : dict / list of dict, len (n_td)
        trialData structure containing the relevant signals to display.
        
    signals : str / list of str, len (n_signals)
        Signals to display for marking the events.
        
    events : str / list of str, len (n_events)
        Name of the events to mark.
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
        
    signal_interval : int / float, optional
        Interval (in seconds) of signal to display in the plot. The default is 
        30.
        
    offset : int / float, optional
        Offset for shitfting the gait events. The offset must be in samples.
        The default value is 0.
        
    events_file : str / list of str, optional
        Path of an existing file containing marked events. events_file must 
        contain PATH/FILENAME.FORMAT because this is the way the code regognises
        the file.
        
    output_type : str, optional
        Set whether the output values should be in 'time' or 'samples'.
        The default value is 'time'.
        
    save_name : str, optional
        Set the name of the file where to save the gait events. The default is
        a file index.
        
    kin_plot : dict, optional
        Dictionary containing the kinematic variables to plot. Example:
        kin_plot = {'Right':{'forward':['RightUpperLeg_x','RightLowerLeg_x','RightFoot_x','RightToe_x'],
                             'vertical':['RightUpperLeg_z','RightLowerLeg_z','RightFoot_z','RightToe_z']},
                    'Left' :{'forward':['LeftUpperLeg_x','LeftLowerLeg_x','LeftFoot_x','LeftToe_x'],
                             'vertical':['LeftUpperLeg_z','LeftLowerLeg_z','LeftFoot_z','LeftToe_z']}}
        
    verbose : str, optional
        Narrate the several operations in this method. The default is False.

    '''
    # Input variables
    signal_interval_sec = 30
    offset = 0
    events_file = []
    output_type = 'time'
    save_name = ''
    kin_plot = None
    correct_data_flag = False
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'signal_interval':
            signal_interval_sec = value
        elif key == 'offset':
            offset = value
        elif key == 'events_file':
            events_file = value
        elif key == 'output_type':
            output_type = value
        elif key == 'save_name':
            save_name = value
        elif key == 'kin_plot':
            kin_plot = value
            correct_data_flag = True
        elif key == 'verbose':
            verbose = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if type(save_name) is not str:
        raise Exception('ERROR: save_name must be a string. You inputed a {}'.format(type(save_name)))
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # Check that signals are in trialData structure
    if not is_field(td,signals):
        raise Exception('ERROR: Missing fields in the dictionaries...')
        
    # Get lenght of the signal to plot
    signals_len = np.inf
    for idx, td_tmp in enumerate(td):
        signals_len_tmp = [len(td_tmp[signal]) for signal in flatten_list(signals)]
        if (np.diff(signals_len_tmp) > 0.1).any():
            raise Exception('ERROR: in td[{}] signals have different length! Not possible...'.format(idx))
        else:
            signals_len_tmp = signals_len_tmp[0]
        if signals_len_tmp<signals_len:
            signals_len = signals_len_tmp
    
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            if is_field(td[0],fs):
                fs = td[0][fs]
            else:
                raise Exception('ERROR: input field "{}" missing from td.'.format(fs))

    signal_interval_smp = np.round(signal_interval_sec*fs).astype('int')
    
    if signal_interval_smp>signals_len:
        print('WARNING: selected interval to plot is > than signal length. {}>{}'.format(signal_interval_smp,signals_len))
        print('signal_interval set = signals_len')
        signal_interval_smp = signals_len
    
    # Check if there is a file to get the events from
    if type(events_file) is str:
        events_file = [events_file]
    
    if len(events_file) != 0:
        if len(events_file) != len(td):
            raise Exception('ERROR: the number of events_file "{}" must be = to the number of td "{}"!'.format(len(events_file),len(td)))
        
        gaits_events_file = []
        for event_file in events_file:
            indexes_slash = find_substring_indexes(event_file,'/')
            indexes_point = find_substring_indexes(event_file,'.')
            if len(indexes_slash) != 0 and len(indexes_point) != 0:
                folder = event_file[:indexes_slash[-1]]
                file_name = event_file[indexes_slash[-1]+1:indexes_point[-1]]
                file_format = event_file[indexes_point[-1]:]
                gait_events_file_tmp = load_data_from_folder(folders = folder, files_name = file_name, files_format = file_format)[0]
                if 'offset' not in gait_events_file_tmp.keys() or\
                    'output_type' not in gait_events_file_tmp.keys() or\
                    'fs' not in gait_events_file_tmp.keys():
                    raise Exception('ERROR: "offset" or "output_type" or "fs" are missing from the loaded event_file!')
                if (gait_events_file_tmp['offset'] - offset)>0.1:
                    raise Exception('ERROR: "offset" from the loaded event_file is different from the set offset!')
                if (gait_events_file_tmp['fs'] - fs)>0.1:
                    raise Exception('ERROR: "fs" from the loaded event_file is different from the set fs!')
                
                gaits_events_file.append(gait_events_file_tmp)
            else:
                raise Exception('ERROR: event_file must follow the following structure: PATH/FILENAME.FORMAT .')
    else:
        gaits_events_file = [dict()] * len(td)
    # Set gait event variable
    gaits_events = []
    for iTd, td_tmp in enumerate(td):
        # Gait event array for the current file
        gait_events_tmp = dict()
        for ev in events:
            gait_events_tmp[ev] = np.array([])
        gaits_events.append(gait_events_tmp)
    
    # If there is a file to get the events from, copy them in gaits_events
    if len(events_file) != 0:
        for gait_events_file in gaits_events_file:
            for event in gait_events_file.keys():
                if event in events:
                    if gait_events_file['output_type'] == 'time':
                        gait_events_file[event] = (gait_events_file[event]*fs + offset).astype('int')
                    else:
                        gait_events_file[event] = (gait_events_file[event] + offset).astype('int')
    
    # ========================================================================
    # Plot data
    fig, axs = plt.subplots(nrows = len(signals), ncols = 1, sharex=True)
    # Loop over the dictionaries
    for iTd, (td_tmp, gait_events, gait_events_file) in enumerate(zip(td, gaits_events, gaits_events_file)):
        # Set title
        fig.suptitle('File {}/{}\nPress ESC to switch event type.'.format(iTd+1,len(td)), fontsize=10)
        
        # Maximise figure
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        
        # Get data
        signal_name = []
        signal_data = []
        signal_ylim = []
        for iSig, signal in enumerate(signals):
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
            
        intervals = np.append(np.arange(0,signal_data[0].shape[0],signal_interval_smp),signal_data[0].shape[0])
        intervals_range = np.arange(intervals.shape[0]-1)
        for iInt in intervals_range:
            # For each event
            for event in events:
                # Plot the signals
                for iSig, (signal,name,ylim) in enumerate(zip(signal_data,signal_name,signal_ylim)):
                    axs[iSig].plot(signal[np.arange(intervals[iInt],intervals[iInt+1]), :])
                    axs[iSig].set_ylim(ylim)
                    axs[iSig].legend(loc='upper right', labels = name)
                    if iSig != 0:
                        axs[iSig].title.set_text(' + '.join(name))
                
                # Set the title related to the event                
                axs[0].title.set_text('Select {} event. '.format(event) + ' + '.join(signal_name[0]))   
                
                # Plot existing events
                pts_add = np.array([])
                if event in gait_events_file:
                    if gait_events_file[event].size != 0:
                        events_in_interval = np.logical_and(np.array(gait_events_file[event]) > intervals[iInt], np.array(gait_events_file[event]) < intervals[iInt+1])
                        if events_in_interval.any():
                            pts_add = gait_events_file[event][events_in_interval] - intervals[iInt]
                            for ev in pts_add:
                                for iSig in np.arange(len(signals)):
                                    axs[iSig].axvline(ev,0,1)
                    
                # Draw everything
                fig.canvas.draw()   
                
                # Mark points on plot
                pts = plt.ginput(-1, timeout=0, show_clicks = True, mouse_add=1, mouse_pop=3)
                
                # Save points
                if len(pts) != 0:
                    # Add pre-existing events if there are
                    pts = np.round(np.sort(np.concatenate((np.asarray(pts)[:,0],pts_add)))).astype('int')
                    gait_events[event] = np.concatenate( (gait_events[event], intervals[iInt]+pts) ).astype('int')
                elif pts_add.size != 0:
                    gait_events[event] = np.concatenate( (gait_events[event], intervals[iInt]+pts_add) ).astype('int')
                    
                # Clear the plots
                for iSig in np.arange(len(signals)):
                    axs[iSig].cla()
    # Close the figure    
    plt.close()
    
    # ========================================================================
    # Correct the data using stickplot
    if correct_data_flag:
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
        for iTd, (gait_events, td_tmp) in enumerate(zip(gaits_events,td)):
            for event in events:
                if 'R' in event:
                    leg_variables_forward_names = kin_plot['Right']['forward']
                    leg_variables_vertica_names = kin_plot['Right']['vertical']
                elif 'L' in event:
                    leg_variables_forward_names = kin_plot['Left']['forward']
                    leg_variables_vertica_names = kin_plot['Left']['vertical']
                else:
                    continue
                    
                # Check existance of variables in the dataset
                if not is_field(td,leg_variables_forward_names+leg_variables_vertica_names):
                    raise Exception('Missing fields in the dictionaries...')
                
                X = np.array([]).reshape(len(td_tmp[leg_variables_forward_names[0]]),0)
                Y = np.array([]).reshape(len(td_tmp[leg_variables_vertica_names[0]]),0)
                for var_x, var_y in zip(leg_variables_forward_names,leg_variables_vertica_names):
                    X = np.hstack([ X, np.array(td_tmp[var_x]).reshape(len(td_tmp[var_x]),1) ])
                    Y = np.hstack([ Y, np.array(td_tmp[var_y]).reshape(len(td_tmp[var_y]),1) ])
                
                event_to_remove = []
                for iEv, ev in enumerate(gait_events[event]):
                    fig.suptitle('File {}/{}. Event {}. #{}/{}'.format(iTd+1,len(td),event,iEv+1,len(gait_events[event])))
                    not_stop_loop = True
                    plus_ev = 50
                    while not_stop_loop:
                        if X.shape[0]-ev >=plus_ev:
                            event_stop = ev+plus_ev
                        else:
                            event_stop = X.shape[0]
                        if ev-plus_ev >=0:
                            event_start = ev-plus_ev
                        else:
                            event_start = 0
                        event_interval = np.arange(event_start,event_stop)
                        if verbose: print(ev)
                        plt.plot(X[event_interval,:].T,Y[event_interval,:].T,'k')
                        plt.plot(X[ev,:],Y[ev,:],'r')
                        plt.axis('equal')
                        plt.draw()
                        plt.pause(0.1)
                        
                        fig.canvas.mpl_connect('key_press_event', press)
                        _ = plt.waitforbuttonpress(0) #non specific key press!!!
                        
                        key = mutable_object['key']
                        if key == 1:
                            if ev < X.shape[0]:
                                ev += 1
                            if verbose: print('-->')
                        elif key == -1:
                            if ev > 0:
                                ev -= 1
                            if verbose: print('<--')
                        elif key == 2:
                            not_stop_loop = False
                            if verbose: print('ESC')
                        elif key == 3:
                            event_to_remove.append(iEv)
                            not_stop_loop = False
                            if verbose: print('DEL')
                        else:
                            print('Possible keys are: left, right, delete, escape.\nTry again...')
                        
                        plt.cla()
                    
                    gait_events[event][iEv] = ev
                gait_events[event] = np.delete(gait_events[event], event_to_remove)
        plt.close()
        
    # ========================================================================
    # Correct data with offset if present
    for gait_events in gaits_events:
        gait_events['offset'] = offset
        if offset != 0:
            for event in events:
                gait_events[event] = gait_events[event] - offset
        
    # Adjust outcome type
    if output_type == 'time':
        for gait_events in gaits_events:
            gait_events['output_type'] = 'time'
            for event in events:
                gait_events[event] = gait_events[event]/fs
    else: 
        for gait_events in gaits_events:
            gait_events['output_type'] = 'samples'
    
    # Save frequency
    for gait_events in gaits_events:
        gait_events['fs'] = fs
    
    # Save the data
    for idx, (td_tmp, gait_events) in enumerate(zip(td, gaits_events)):
        savemat('gait_events_' + str(idx) + save_name + '.mat', gait_events, appendmat=True)

# =============================================================================
# Mark events 2/3D from stick plot
# =============================================================================
def gait_event_manual_stick_plot(td, kinematics, fs, **kwargs):
    '''
    This function helps marking the foot events from the 3d stick plot of the 
    subject kinematics.
    
    Events to mark are hard-coded:
        RHS: 1
        RTO: 2
        LHS: 3
        LTO: 4
        Turn_on: 5
        Turn_off: 6
    
    Parameters
    ----------
    td : dict
        Dictionary(ies) containig the kinematic data
        
    kinematics : dict
        Dictionary containing the marker 2D/3D information for each time instant.
        Separate the dictionary in different body parts: 
            'leg_r','leg_l','arm_r','arm_l','head','trunk','other'.
        
    fs : str / int
        Sampling frequency.
        If str, it can either be one key in td or the path in td where to find 
        the fs in td (e.g. params/data/data).
            
    coordinates : list of str, len (n_coordinates), optional
        Coordinates as ordered in the kinematics dictionary. 
        The default is ['x','y','z'].
        
    offset : int / float, optional
        Offset for shitfting the gait events. The offset must be in samples.
        The default value is 0.
        
    output_type : str, optional
        Set whether the output values should be in 'time' or 'samples'.
        The default value is 'time'.
        
    events_file : str, optional
        Path of an existing file containing marked events. events_file must 
        contain PATH/FILENAME.FORMAT because this is the way the code regognises
        the file.
        
    save_name : str, optional
        Set the name of the file where to save the gait events. The default is
        a file index.
        
    verbose : str, optional
        Narrate the several operations in this method. The default is False.
        
    Example
    ----------
    leg_r = [['RightLeg_x','RightLeg_y','RightLeg_z'],
             ['RightFoot_x','RightFoot_y','RightFoot_z']]
    leg_l = [['LeftLeg_x','LeftLeg_y','LeftLeg_z'],
             ['LeftFoot_x','LeftFoot_y','LeftFoot_z']]
    
    kinematics = {'leg_r': leg_r, 'leg_l': leg_l}
    events = ['RHS','RTO','LHS','LTO','Turn_on','Turn_off']
    gait_event_manual_stick_plot(td, kin_info, events, 100, coordinates = ['x','y','z'])

    '''
    
    coordinates = ['x','y','z']
    offset = 0
    output_type = 'time'
    events_file = ''
    save_name = ''
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'coordinates':
            coordinates = value
        elif key == 'offset':
            offset = value
        elif key == 'output_type':
            output_type = value
        elif key == 'events_file':
            events_file = value
        elif key == 'save_name':
            save_name = value
        elif key == 'verbose':
            verbose = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))
    
    # ========================================================================
    # Check input variables
    if type(td) is not dict:
        raise Exception('ERROR: td must be a dict!')
    
    # Check kinematics fields
    body_parts = ['leg_r','leg_l','arm_r','arm_l','head','trunk','other']
    for body_part in kinematics.keys():
        if body_part not in body_parts:
            raise Exception('ERROR: Possible body parts are "leg_r","leg_l","arm_r","arm_l","head","trunk","other". You inserted "{}" !'.format(body_part))
            
    signals = []
    for k,v in kinematics.items():
        signals += v
    signals = flatten_list(signals)
    
    if not is_field(td, signals, True):
        raise Exception('ERROR: signals fields are missing from the trial data!')
        
    # check save_name
    if type(save_name) is not str:
        raise Exception('ERROR: save_name must be s string. You inputed a "{}".'.format(type(save_name)))
    if save_name == '' and 'File' in td.keys():
        save_name = td['File']
    
    # Check fs
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td,fs)['fs']
        else:
            if is_field(td,fs):
                fs = td[fs]
            else:
                raise Exception('ERROR: input field "{}" missing from td.'.format(fs))
    elif type(fs) is int or type(fs) is float:
        pass
    else:
        raise Exception('ERROR: fs is not a string/int/float. You inputed a "{}".'.format(type(fs)))

    # Check if there is a file to get the events from
    if type(events_file) is not str:
        raise Exception('ERROR: events_file must be s string. You inputed a "{}".'.format(type(events_file)))
    
    if events_file != '':
        indexes_slash = find_substring_indexes(events_file,'/')
        indexes_point = find_substring_indexes(events_file,'.')
        if len(indexes_slash) != 0 and len(indexes_point) != 0:
            folder = events_file[:indexes_slash[-1]]
            file_name = events_file[indexes_slash[-1]+1:indexes_point[-1]]
            file_format = events_file[indexes_point[-1]:]
            gait_events_file = load_data_from_folder(folders = folder, files_name = file_name, files_format = file_format)[0]
            if 'offset' not in gait_events_file.keys() or\
                'output_type' not in gait_events_file.keys() or\
                'fs' not in gait_events_file.keys():
                raise Exception('ERROR: "offset" or "output_type" or "fs" are missing from the loaded events_file!')
            if (gait_events_file['offset'] - offset)>0.1:
                raise Exception('ERROR: "offset" from the loaded events_file "{}" is different from the set offset "{}"!'.format(gait_events_file['offset'],offset))
            if (gait_events_file['fs'] - fs)>0.1:
                raise Exception('ERROR: "fs" from the loaded events_file "{}" is different from the set fs "{}"!'.format(gait_events_file['fs'],fs))
        else:
            raise Exception('ERROR: events_file must follow the following structure: PATH/FILENAME.FORMAT .')
    
    # Set gait event variable
    events_right = ['RHS', 'RTO']
    events_left  = ['LHS', 'LTO']
    events_other = ['Turn_on', 'Turn_off']
    events = events_right + events_left + events_other
    
    # Gait event array for the current file
    gait_events = dict()
    for ev in events:
        gait_events[ev] = np.array([])
    
    # If there is a file to get the events from, copy them in gaits_events
    if events_file != '':
        for event in gait_events_file.keys():
            if event in events:
                if gait_events_file['output_type'] == 'time':
                    gait_events[event] = (gait_events_file[event]*fs + offset).astype('int')
                else:
                    gait_events[event] = (gait_events_file[event] + offset).astype('int')
    
    # ========================================================================
    # Collect points in the space
    
    # Get data len
    signals_len = [len(td[signal]) for signal in signals]
    if (np.diff(signals_len) > 0.1).any():
        raise Exception('ERROR: signals have different length! Not possible...')
    else:
        signals_len = signals_len[0]
    
    kin_var = dict()
    for body_part in kinematics.keys():
        kin_var[body_part] = dict()
        for coordinate in coordinates:
            kin_var[body_part][coordinate] = np.array([]).reshape(signals_len,0)
    
    for body_part, fields in kinematics.items():
        for field in fields:
            for coordinate, field_coord in zip(coordinates, field):
                kin_var[body_part][coordinate] = np.hstack([kin_var[body_part][coordinate], np.array(td[field_coord]).reshape(signals_len,1) ])
    
    # ========================================================================
    # Mark the events using the stickplot
    
    mutable_object = {}
    def press(input_key):
        sys.stdout.flush()
        # print('press:{}'.format(input_key.key))
        if input_key.key == 'right':
            output = +1
        elif input_key.key == 'left':
            output = -1
        elif input_key.key == 'up':
            output = +30
        elif input_key.key == 'down':
            output = -30
        elif input_key.key == 'escape':
            output = 2
        elif input_key.key == 'backspace':
            output = 3
        elif input_key.key == '1':
            output = events_right[0] #'RHS'
        elif input_key.key == '2':
            output = events_right[1] #'RTO'
        elif input_key.key == '3':
            output = events_left[0] #'LHS'
        elif input_key.key == '4':
            output = events_left[1] #'LTO'
        elif input_key.key == '5':
            output = events_other[0] #'Turn_on'
        elif input_key.key == '6':
            output = events_other[1] #'Turn_off'
        else:
            output = 0
        mutable_object['key'] = output
    
    
    # Plotting events characteristics
    body_part_color = {'leg_r' : np.array([240,128,128])/255,
                       'leg_l' : np.array([60,179,113])/255,
                       'arm_r' : np.array([178,34,34])/255,
                       'arm_l' : np.array([34,139,34])/255,
                       'head'  : np.array([0,191,255])/255,
                       'trunk' : np.array([138,43,226])/255,
                       'other' : np.array([125,125,125])/255}
    
    # Get axis lim
    xyz_lim = dict()
    for coordinate in coordinates:
        xyz_lim[coordinate] = [+np.inf, -np.inf]
    
    for body_part in kin_var.keys():
        for coordinate in coordinates:
            if xyz_lim[coordinate][0] > np.min(kin_var[body_part][coordinate]):
                xyz_lim[coordinate][0] = np.min(kin_var[body_part][coordinate])
            if xyz_lim[coordinate][1] < np.max(kin_var[body_part][coordinate]):
                xyz_lim[coordinate][1] = np.max(kin_var[body_part][coordinate])

    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if len(coordinates) == 3:
        ax.set_xlabel('{} axis'.format(coordinates[0]))
        ax.set_ylabel('{} axis'.format(coordinates[1]))
        ax.set_zlabel('{} axis'.format(coordinates[2]))
        ax.set_xlim(xyz_lim[coordinates[0]])
        ax.set_ylim(xyz_lim[coordinates[1]])
        ax.set_zlim(xyz_lim[coordinates[2]])
    else:
        ax.set_xlabel('{} axis'.format(coordinates[0]))
        ax.set_ylabel('{} axis'.format(coordinates[1]))
        ax.set_xlim(xyz_lim[coordinates[0]])
        ax.set_ylim(xyz_lim[coordinates[1]])
    
    # Flag for stopping the marking
    not_stop_loop = True
    # Sample idx
    sample_idx = 0
    # Start marking
    while not_stop_loop:
        # Plot sticks
        for body_part, values in kin_var.items():
            if len(coordinates) == 3:
                ax.plot(kin_var[body_part][coordinates[0]][sample_idx,:],kin_var[body_part][coordinates[1]][sample_idx,:],kin_var[body_part][coordinates[2]][sample_idx,:], Color = body_part_color[body_part])
                ax.set_xlabel('{} axis'.format(coordinates[0]))
                ax.set_ylabel('{} axis'.format(coordinates[1]))
                ax.set_zlabel('{} axis'.format(coordinates[2]))
                ax.set_xlim(xyz_lim[coordinates[0]])
                ax.set_ylim(xyz_lim[coordinates[1]])
                ax.set_zlim(xyz_lim[coordinates[2]])
            else:
                ax.plot(kin_var[body_part][coordinates[0]][sample_idx,:],kin_var[body_part][coordinates[1]][sample_idx,:], Color = body_part_color[body_part])
                ax.set_xlabel('{} axis'.format(coordinates[0]))
                ax.set_ylabel('{} axis'.format(coordinates[1]))
                ax.set_xlim(xyz_lim[coordinates[0]])
                ax.set_ylim(xyz_lim[coordinates[1]])
        
        # Plot title
        if 'File' in td.keys():
            plt.suptitle('Stick plot. File: {}\nRHS:1, RTO:2, LHS:3, LTO:4, T_on:5, T_off:6\nSample {}/{}'.format(td['File'],sample_idx,signals_len-1))
        else:
            plt.suptitle('Stick plot./nRHS:1, RTO:2, LHS:3, LTO:4, T_on:5, T_off:6\nSample {}/{}'.format(sample_idx,signals_len-1))
        
        # Check whether there are events
        is_there_an_event = False
        for event in gait_events.keys():
            if gait_events[event].size != 0:
                if (np.abs(gait_events[event] - sample_idx) < 0.1).any():
                    is_there_an_event = True
                    if event in events_right:
                        part = 'leg_r'
                        col = body_part_color[part]
                    elif event in events_left:
                        part = 'leg_l'
                        col = body_part_color[part]
                    elif event in events_other:
                        part = 'leg_r'
                        col = 'y'
                    
                    if len(coordinates) == 3:
                        ax.plot([kin_var[part][coordinates[0]][sample_idx,-1]]*2,[kin_var[part][coordinates[1]][sample_idx,-1]]*2, xyz_lim[coordinates[2]], Color = col)
                    else:
                        ax.plot([kin_var[part][coordinates[0]][sample_idx,-1]]*2, xyz_lim[coordinates[1]], Color = col)
                    
                    ax.text2D(0.05, 0.95, event, transform=ax.transAxes)
                    
        # Add text
        if is_there_an_event == False:
            ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
                    
        # Take a decision by clicking a button
        fig.canvas.mpl_connect('key_press_event', press)
        _ = plt.waitforbuttonpress(0) #non specific key press!!!
        
        key = mutable_object['key']
        if key == 1:
            if sample_idx < signals_len-1:
                sample_idx += 1
            if verbose: print('-->')
        elif key == -1:
            if sample_idx > 0:
                sample_idx -= 1
            if verbose: print('<--')
        elif key == +30:
            if signals_len - sample_idx > 30:
                sample_idx += 30
            else:
                sample_idx = signals_len-1
            if verbose: print('-->')
        elif key == -30:
            if sample_idx > 30:
                sample_idx -= 30
            else:
                sample_idx = 0
            if verbose: print('<--')
        elif key == 2:
            not_stop_loop = False
            if verbose: print('ESC')
        elif key == 3: # Remove event
            del_sample = False
            for event in gait_events.keys():
                if (np.abs(gait_events[event] - sample_idx) < 0.1).any():
                    del_sample = True
                    del_idx = np.where((np.abs(gait_events[event] - sample_idx) < 0.1))[0]
                    gait_events[event] = np.delete(gait_events[event], del_idx).astype('int')
                    print('Sample {} removed from key {}!'.format(sample_idx, event))
            if del_sample == False: print('No sample removed!')
            if verbose: print('DEL')
        elif key == events_right[0]: # RHS
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('RHS')
        elif key == events_right[1]: # RTO
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('RTO')
        elif key == events_left[0]: # LHS
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('LHS')
        elif key == events_left[1]: # LTO
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('LTO')
        elif key == events_other[0]: # Turn_on
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('Turn_on')
        elif key == events_other[1]: # Turn_off
            if not (np.abs(gait_events[key] - sample_idx) < 0.1).any():
                gait_events[key] = np.sort(np.insert(gait_events[key],0,sample_idx)).astype('int')
            if verbose: print('Turn_off')
        else:
            print('Possible keys are: left, right, delete, escape.\nTry again...')
        
        plt.draw()
        plt.cla()
    # End while
    plt.close()
    
    # ========================================================================
    # Correct data
    # Check offset if present
    if offset != 0:
        gait_events['offset'] = offset
        for event in events:
            gait_events[event] = gait_events[event] - offset
            
    # Adjust outcome type
    if output_type == 'time':
        gait_events['output_type'] = 'time'
        for event in events:
            gait_events[event] = gait_events[event]/fs
    else:
        gait_events['output_type'] = 'samples'
    
    # Save frequency
    gait_events['fs'] = fs
    
    # Save the data
    savemat(save_name + '_gait_events.mat', gait_events, appendmat=True)

# =============================================================================
# Collect data around initiation events
# =============================================================================
def get_initiation(td, fields, events, fs, **kwargs):
    '''
    This function extracts the signals around certain events.
    
    Parameters
    ----------
    td : dict / list of dict, len (n_td)
        Trial data dictionary containing the data.
        
    fields : str / list of str, len (n_fields)
        Fields in td with the signals to extract around the events.
        If str, it can either be one key in td or the path in td where to find
        the fs in td (e.g. params/data/data). In case of multiple td, the fields
        input must be shared among the td.
        
    events : str / np.ndarray, shape (n_events,) 
        Initiation events. If str, it takes the events from a field in td.
        events are considered to be in samples, otherwise change the events_kind
        parameter. In case of multiple td, the events input must be shared among
        the td.
        
    fs : str / int
        Sampling frequency. If str, it can either be one key in td or the path 
        in td where to find the fs in td (e.g. params/data/data).
         In case of multiple td, the fs input must be shared among the td.
        
    pre_events : str / float / np.ndarray, shape (n_events,), optional
        Before the event. If str, it takes the events from a field in td.
        If float, it is a constant value. The default is 1 second. 
        
    post_events : str / float / np.ndarray, shape (n_events,), optional
        After the event. If str, it takes the events from a field in td.
        If float, it is a constant value. The default is 1 second. 
        
    events_kind : str, optional
        Specify whether events are in "time" or "samples". Default is "samples".
        The default is "samples".
        
    Return
    ----------
    td_init_norm : dict / list of dict, len (n_td)
        Trial data dictionary containing the data in fields around the events
        normalised to their average length.
        
    td_init : dict / list of dict, len (n_td)
        Trial data dictionary containing the data in fields around the events
        NOT normalised.

    '''
    # Input variables
    pre_events = 1
    post_events = 1
    events_kind = 'samples'
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'pre_events':
            pre_events = value
        elif key == 'post_events':
            post_events = value
        elif key == 'events_kind':
            events_kind = value

    # ========================================================================
    # Check input variables
    
    # Check td
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
    
    # Check fields
    if type(fields) is str:
        if '/' in fields:
            fields = td_subfield(td[0],fields)['signals']
        else:
            fields = [fields]
    
    if type(fields) is not list:
        raise Exception('ERROR: fields must be a list of strings!')
        
    if not is_field(td, fields, True):
        raise Exception('ERROR: missing fields in td list!')  
    
    # Check fs
    if type(fs) is str:
        if '/' in fs:
            fs = td_subfield(td[0],fs)['fs']
        else:
            if is_field(td[0],fs):
                fs = td[0][fs]
            else:
                raise Exception('ERROR: input field "{}" missing from td.'.format(fs))
    elif type(fs) is int or type(fs) is float:
        pass
    else:
        raise Exception('ERROR: fs is not a string/int/float. You inputed a "{}".'.format(type(fs)))
    
    if events_kind not in ['time', 'samples']:
        raise Exception('ERROR: events_kind can only be "time" or "samples". You inputed a "{}".'.format(events_kind))
    
    # Check events
    events_list = []
    pre_events_list = []
    post_events_list = []
    for td_tmp in td:
        if type(events) is str:
            events_tmp = np.sort(np.array(td_tmp[events]))
        elif type(events) is np.ndarray:
            pass
        else:
            raise Exception('ERROR: events is not a string/np.ndarray. You inputed a "{}".'.format(type(events)))
            
        if events_kind == 'time':
            events_tmp = np.sort(np.round(events_tmp*fs)).astype('int')
        # Store events_tmp in events_list
        events_list.append(events_tmp)
        
        # Check pre_events
        if type(pre_events) is str:
            pre_events_tmp = np.sort(np.array(td_tmp[pre_events]))
            if events_kind == 'time':
                pre_events_tmp = np.round(pre_events_tmp*fs).astype('int')
        elif type(pre_events) is int or type(pre_events) is float:
            pre_events_tmp = (events_tmp - np.round(pre_events*fs)).astype('int')
        elif type(pre_events) is np.ndarray:
            if events_kind == 'time':
                pre_events_tmp = np.sort(np.round(pre_events*fs)).astype('int')
        else:
            raise Exception('ERROR: pre_events is not a string/np.ndarray. You inputed a "{}".'.format(type(pre_events)))
        
        if len(pre_events_tmp) != len(events_tmp):
            raise Exception('ERROR: pre_events length "{}" != events length "{}".'.format(len(pre_events_tmp), len(events_tmp)))
        # Store events_tmp in events_list
        pre_events_list.append(pre_events_tmp)
        
        # Check post_events
        if type(post_events) is str:
            post_events_tmp = np.sort(np.array(td_tmp[post_events]))
            if events_kind == 'time':
                post_events_tmp = np.round(post_events_tmp*fs).astype('int')
        elif type(post_events) is int or type(post_events) is float:
            post_events_tmp = (events_tmp + np.round(post_events*fs)).astype('int')
        elif type(post_events) is np.ndarray:
            if events_kind == 'time':
                post_events_tmp = np.sort(np.round(post_events*fs).astype('int'))
        else:
            raise Exception('ERROR: post_events is not a string/np.ndarray. You inputed a "{}".'.format(type(post_events)))
        
        if len(post_events_tmp) != len(events_tmp):
            raise Exception('ERROR: post_events length "{}" != events length "{}".'.format(len(post_events_tmp), len(events_tmp)))
        # Store events_tmp in events_list
        post_events_list.append(post_events_tmp)
    
    # ========================================================================
    # Get intervals of interest
    
    intervals_pre_list = []
    intervals_list = []
    for pre_events, events, post_events in zip(pre_events_list, events_list, post_events_list):
        intervals_pre_list.append([np.arange(pre_event,event)  for pre_event, event  in zip(pre_events, events)])
        intervals_list.append([np.arange(event+1,post_event) for event, post_event in zip(events, post_events)])
    
    intervals_pre_mean = np.array([len(interval) for intervals_pre in intervals_pre_list for interval in intervals_pre]).mean().round().astype('int')
    intervals_mean     = np.array([len(interval) for intervals in intervals_list for interval in intervals]).mean().round().astype('int')
    
    # ========================================================================
    # Extract the data for each interval
    td_init_pre = []
    td_init     = []
    
    # Collect data in intervals
    for td_tmp, intervals_pre, intervals  in zip(td, intervals_pre_list, intervals_list):
        td_init_pre_tmp = dict()
        td_init_tmp     = dict()
        for field in fields:
            # Before event
            td_init_pre_tmp[field + '_pre'] = []
            for interval in intervals_pre:
                td_init_pre_tmp[field + '_pre'].append(td_tmp[field][interval])
            # After event
            td_init_tmp[field] = []
            for interval in intervals:
                td_init_tmp[field].append(td_tmp[field][interval])
        
        td_init_pre.append(td_init_pre_tmp)
        td_init.append(td_init_tmp)
    
    # ========================================================================
    # Normalise data
    td_init_pre_norm = []
    td_init_norm     = []
    
    # Normalise data in intervals
    for td_init_pre_tmp, td_init_tmp  in zip(td_init_pre, td_init):
        td_init_pre_norm_tmp = dict()
        td_init_norm_tmp     = dict()
        # Before event
        for field in td_init_pre_tmp.keys():
            td_init_pre_norm_tmp[field] = []
            for signal in td_init_pre_tmp[field]:
                td_init_pre_norm_tmp[field].append(interpolate1D(signal, intervals_pre_mean, kind = 'cubic'))
        # After event
        for field in td_init_tmp.keys():
            td_init_norm_tmp[field] = []
            for signal in td_init_tmp[field]:
                td_init_norm_tmp[field].append(interpolate1D(signal, intervals_mean, kind = 'cubic'))
    
        td_init_pre_norm.append(td_init_pre_norm_tmp)
        td_init_norm.append(td_init_norm_tmp)
        
    # ========================================================================
    # Combine data and return
    
    combine_dicts((td_init, td_init_pre), inplace = True)
    combine_dicts((td_init_norm, td_init_pre_norm), inplace = True)
    
    if input_dict:
        td_init_norm = td_init_norm[0]
        td_init = td_init[0]
    
    return td_init_norm, td_init

# =============================================================================
# Stick plot video
# =============================================================================
def stick_plot_video(td, kinematics, **kwargs):
    '''
    This function plots the sticks for the kinematics
    
    Parameters
    ----------
    td : dict
        Dictionary containig the kinematic data
        
    kinematics : dict
        Dictionary containing the marker 2D/3D information for each time instant.
        Separate the dictionary in different body parts: 
            'leg_r','leg_l','arm_r','arm_l','head','trunk','other'.
            
    coordinates : list of str, len (n_coordinates)
        Coordinates as ordered in the kinematics dictionary. 
        The default is ['x','y','z'].
        
    step_plot : int/float, optional
        Step (in samples) between one representation and the next.
        The default is 1.
        
    pause : int/float, optional
        Pause (in seconds) between one representation and the next.
        The default is 1.
        
    idx_start : int/float, optional
        Starting point of the stick plot. It is in samples or percentage (0-1)
        of the whole signal. The default is 0.
        
    idx_stop : int/float, optional
        Stopping point of the stick plot. It is in samples or percentage (0-1)
        of the whole signal. The default is 0.
        
    events : str / list of str, len (n_events)
        Name of the events to plot.
    
    verbose : str, optional
        Narrate the several operations in this method. The default is False.
    
    Example:
    ----------
    leg_r = [['RightLeg_x','RightLeg_y','RightLeg_z'],
             ['RightFoot_x','RightFoot_y','RightFoot_z']]
    leg_l = [['LeftLeg_x','LeftLeg_y','LeftLeg_z'],
             ['LeftFoot_x','LeftFoot_y','LeftFoot_z']]
    
    kinematics = {'leg_r': leg_r, 'leg_l': leg_l}
    stick_plot(td, kinematics, coordinates = ['x','y','z'], step_plot = 10)

    '''
    
    coordinates = ['x','y','z']
    step_plot = 10
    pause = .1
    idx_start = 0
    idx_stop = 0
    events = None
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'coordinates':
            coordinates = value
        elif key == 'step_plot':
            step_plot = value
        elif key == 'pause':
            pause = value
        elif key == 'idx_start':
            idx_start = value
        elif key == 'idx_stop':
            idx_stop = value
        elif key == 'events':
            events = value
        elif key == 'verbose':
            verbose = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))
    
    # ========================================================================
    # Check input variables
    body_parts = ['leg_r','leg_l','arm_r','arm_l','head','trunk','other']
    for body_part in kinematics.keys():
        if body_part not in body_parts:
            raise Exception('ERROR: Possible body parts are "leg_r","leg_l","arm_r","arm_l","head","trunk","other". You inserted "{}" !'.format(body_part))

    signals = []
    for k,v in kinematics.items():
        signals += v
    signals = flatten_list(signals)
    
    if not is_field(td, signals, True):
        raise Exception('ERROR: signals fields are missing from the trial data!')
    
    # Check events field in td
    # if not is_field(td, events, True):
    #     raise Exception('ERROR: events fields are missing from the trial data!')
    
    # ========================================================================
    # Collect points in the space
    
    # Get data len
    signals_len = [len(td[signal]) for signal in signals]
    if (np.diff(signals_len) > 0.1).any():
        raise Exception('ERROR: signals have different length! Not possible...')
    else:
        signals_len = signals_len[0]
    
    # Check idx_start and idx_stop
    if idx_start != 0:
        if idx_start<1: # It is a percentage
            idx_start = signals_len*idx_start
        else: # It is a value
            if idx_start >= signals_len:
                raise Exception('ERROR: idx_start > length of the signal! idx_start = {}, signals len = {}'.format(idx_start,signals_len))
        
    if idx_stop != 0:
        if idx_stop<1: # It is a percentage
            idx_stop = idx_stop*idx_start
        else: # It is a value
            if idx_stop >= signals_len:
                idx_stop = signals_len
    else:
        idx_stop = signals_len
    
    kin_var = dict()
    for body_part in kinematics.keys():
        kin_var[body_part] = dict()
        for coordinate in coordinates:
            kin_var[body_part][coordinate] = np.array([]).reshape(signals_len,0)
    
    for body_part, fields in kinematics.items():
        for field in fields:
            for coordinate, field_coord in zip(coordinates, field):
                kin_var[body_part][coordinate] = np.hstack([kin_var[body_part][coordinate], np.array(td[field_coord]).reshape(signals_len,1) ])
    
    # ========================================================================
    # Correct the data using stickplot
    
    # Plotting events characteristics
    body_part_color = {'leg_r' : np.array([240,128,128])/255,
                       'leg_l' : np.array([60,179,113])/255,
                       'arm_r' : np.array([178,34,34])/255,
                       'arm_l' : np.array([34,139,34])/255,
                       'head'  : np.array([0,191,255])/255,
                       'trunk' : np.array([138,43,226])/255,
                       'other' : np.array([125,125,125])/255}
    
    # Get axis lim
    xyz_lim = dict()
    for coordinate in coordinates:
        xyz_lim[coordinate] = [+np.inf, -np.inf]
    
    for body_part in kin_var.keys():
        for coordinate in coordinates:
            if xyz_lim[coordinate][0] > np.min(kin_var[body_part][coordinate]):
                xyz_lim[coordinate][0] = np.min(kin_var[body_part][coordinate])
            if xyz_lim[coordinate][1] < np.max(kin_var[body_part][coordinate]):
                xyz_lim[coordinate][1] = np.max(kin_var[body_part][coordinate])

    # Plot
    fig = plt.figure()
    if len(coordinates) == 3:
        ax = fig.gca(projection='3d')
    else:
        ax = fig.add_subplot(111)
    if 'File' in td.keys():
        plt.suptitle('Stick plot. File: {}'.format(td['File']))
    if len(coordinates) == 3:
        ax.set_xlabel('{} axis'.format(coordinates[0]))
        ax.set_ylabel('{} axis'.format(coordinates[1]))
        ax.set_zlabel('{} axis'.format(coordinates[2]))
        ax.set_xlim(xyz_lim[coordinates[0]])
        ax.set_ylim(xyz_lim[coordinates[1]])
        ax.set_zlim(xyz_lim[coordinates[2]])
    else:
        ax.set_xlabel('{} axis'.format(coordinates[0]))
        ax.set_ylabel('{} axis'.format(coordinates[1]))
        ax.set_xlim(xyz_lim[coordinates[0]])
        ax.set_ylim(xyz_lim[coordinates[1]])
    
    for idx in range(idx_start,idx_stop,step_plot):
        if verbose:
            print('Index: {}/{}'.format(idx,signals_len))
        for body_part, values in kin_var.items():
            if len(coordinates) == 3:
                ax.plot(kin_var[body_part][coordinates[0]][idx,:],kin_var[body_part][coordinates[1]][idx,:],kin_var[body_part][coordinates[2]][idx,:], Color = body_part_color[body_part])
                ax.set_xlim(xyz_lim[coordinates[0]])
                ax.set_ylim(xyz_lim[coordinates[1]])
                ax.set_zlim(xyz_lim[coordinates[2]])
            else:
                ax.plot(kin_var[body_part][coordinates[0]][idx,:],kin_var[body_part][coordinates[1]][idx,:], Color = body_part_color[body_part])
                ax.set_xlim(xyz_lim[coordinates[0]])
                ax.set_ylim(xyz_lim[coordinates[1]])

        # ax.axis('equal')
        plt.draw()
        plt.pause(pause)
        plt.cla()

# =============================================================================
# Stick plot at events
# =============================================================================
def stick_plot_at_events(td, kinematics, events, **kwargs):
    '''
    This function plots the sticks for the kinematics
    
    Parameters
    ----------
    td : dict
        Dictionary containig the kinematic data
    
    kinematics : dict
        Dictionary containing the marker 2D/3D information for each time instant.
        Separate the dictionary in different body parts: 
            'leg_r','leg_l','arm_r','arm_l','head','trunk','other'.
    
    events : str / np.ndarray / list, len (n_events)
        Name of the events to plot or array with the events to plot. If the type
        is string, it is considered as an array of zeros with ones where the events
        occurr.
    
    coordinates : list of str, len (n_coordinates)
        Coordinates as ordered in the kinematics dictionary. 
        The default is ['x','y','z'].
    
    events_plus : int/float, optional
        Number of samples to plot around the event. The default is 50.
    
    events_plus_step : int/float, optional
        Step in plotting the events_plus. The default is 5.
        
    save_name : str, optional
        Set the name of the file where to save the gait events. The default is
        a file index.
        
    verbose : str, optional
        Narrate the several operations in this method. The default is False.
    
    Example:
    ----------
    leg_r = [['RightLeg_x','RightLeg_y','RightLeg_z'],
             ['RightFoot_x','RightFoot_y','RightFoot_z']]
    leg_l = [['LeftLeg_x','LeftLeg_y','LeftLeg_z'],
             ['LeftFoot_x','LeftFoot_y','LeftFoot_z']]
    
    kinematics = {'leg_r': leg_r, 'leg_l': leg_l}
    stick_plot(td, kinematics, coordinates = ['x','y','z'], events_plus = 10)
    
    '''
    
    coordinates = ['x','y','z']
    events_plus = 50
    events_plus_step = 5
    save_figure = False
    save_name = ''
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'coordinates':
            coordinates = value
        elif key == 'events_plus':
            events_plus = value
        elif key == 'events_plus_step':
            events_plus_step = value
        elif key == 'save_name':
            save_figure = True
            save_name = value
        elif key == 'verbose':
            verbose = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))
    
    # ========================================================================
    # Check input variables
    body_parts = ['leg_r','leg_l','arm_r','arm_l','head','trunk','other']
    for body_part in kinematics.keys():
        if body_part not in body_parts:
            raise Exception('ERROR: Possible body parts are "leg_r","leg_l","arm_r","arm_l","head","trunk","other". You inserted "{}" !'.format(body_part))

    signals = []
    for k,v in kinematics.items():
        signals += v
    signals = flatten_list(signals)
    
    if not is_field(td, signals, True):
        raise Exception('ERROR: signals fields are missing from the trial data!')
    
    # Check events input
    if type(events) is str:
        if not is_field(td, events, True):
            raise Exception('ERROR: events fields are missing from the trial data!')
        events = find_values(td[events],1).tolist()
    elif type(events) is np.ndarray:
        events = events.tolist()
    
    if type(save_name) is not str:
        raise Exception('ERROR: save_name must be a string. You inputed a {}'.format(type(save_name)))
        
    # ========================================================================
    # Collect points in the space
    
    # Get data len
    signals_len = [len(td[signal]) for signal in signals]
    if (np.diff(signals_len) > 0.1).any():
        raise Exception('ERROR: signals have different length! Not possible...')
    else:
        signals_len = signals_len[0]
    
    for event in events:
        if event < 0 or event > signals_len-1:
            raise Exception('ERROR: an event "{}" is < 0 or > signals_len "{}"!'.format(event,signals_len))
    
    kin_var = dict()
    for body_part in kinematics.keys():
        kin_var[body_part] = dict()
        for coordinate in coordinates:
            kin_var[body_part][coordinate] = np.array([]).reshape(signals_len,0)
    
    for body_part, fields in kinematics.items():
        for field in fields:
            for coordinate, field_coord in zip(coordinates, field):
                kin_var[body_part][coordinate] = np.hstack([kin_var[body_part][coordinate], np.array(td[field_coord]).reshape(signals_len,1) ])
    
    # ========================================================================
    # Correct the data using stickplot
    
    # Plotting events characteristics
    body_part_color = {'leg_r' : np.array([240,128,128])/255,
                       'leg_l' : np.array([60,179,113])/255,
                       'arm_r' : np.array([178,34,34])/255,
                       'arm_l' : np.array([34,139,34])/255,
                       'head'  : np.array([0,191,255])/255,
                       'trunk' : np.array([138,43,226])/255,
                       'other' : np.array([125,125,125])/255}
    
    # Plot
    for iEv, event in enumerate(events):
        if verbose: print('Plotting event {}/{}'.format(iEv+1, len(events)))
        event_range = np.arange(event - events_plus,event + events_plus + 1, events_plus_step)
        
        # Get axis lim
        xyz_lim = dict()
        for coordinate in coordinates:
            xyz_lim[coordinate] = [+np.inf, -np.inf]
        for body_part in kin_var.keys():
            for coordinate in coordinates:
                if xyz_lim[coordinate][0] > np.min(kin_var[body_part][coordinate][event_range,:]):
                    xyz_lim[coordinate][0] = np.min(kin_var[body_part][coordinate][event_range,:])
                if xyz_lim[coordinate][1] < np.max(kin_var[body_part][coordinate][event_range,:]):
                    xyz_lim[coordinate][1] = np.max(kin_var[body_part][coordinate][event_range,:])

        # Plot
        fig = plt.figure()
        if len(coordinates) == 3:
            ax = fig.gca(projection='3d')
        else:
            ax = fig.add_subplot(111)
        if 'File' in td.keys():
            plt.suptitle('Stick plot. File: {}. Event {}'.format(td['File'], event))
        else:
            plt.suptitle('Stick plot. Event {}'.format(event))
        if len(coordinates) == 3:
            ax.set_xlabel('{} axis'.format(coordinates[0]))
            ax.set_ylabel('{} axis'.format(coordinates[1]))
            ax.set_zlabel('{} axis'.format(coordinates[2]))
            ax.set_xlim(xyz_lim[coordinates[0]])
            ax.set_ylim(xyz_lim[coordinates[1]])
            ax.set_zlim(xyz_lim[coordinates[2]])
        else:
            ax.set_xlabel('{} axis'.format(coordinates[0]))
            ax.set_ylabel('{} axis'.format(coordinates[1]))
            ax.set_xlim(xyz_lim[coordinates[0]])
            ax.set_ylim(xyz_lim[coordinates[1]])
        
        for idx in event_range:
            for body_part, values in kin_var.items():
                if len(coordinates) == 3:
                    ax.plot(kin_var[body_part][coordinates[0]][idx,:],kin_var[body_part][coordinates[1]][idx,:],kin_var[body_part][coordinates[2]][idx,:], Color = body_part_color[body_part])
                    ax.set_xlim(xyz_lim[coordinates[0]])
                    ax.set_ylim(xyz_lim[coordinates[1]])
                    ax.set_zlim(xyz_lim[coordinates[2]])
                else:
                    ax.plot(kin_var[body_part][coordinates[0]][idx,:],kin_var[body_part][coordinates[1]][idx,:], Color = body_part_color[body_part])
                    ax.set_xlim(xyz_lim[coordinates[0]])
                    ax.set_ylim(xyz_lim[coordinates[1]])
        
        ax.set_aspect('equal', adjustable='datalim') 
        
        if save_figure:
            pickle.dump(fig, open(save_name + '_kin_plot_event_' + str(event) + '.pickle', 'wb'))
            fig.savefig(save_name + '_kin_plot_event_' + str(event) + '.svg', bbox_inches='tight')
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            fig.savefig(save_name + '_kin_plot_event_' + str(event) + '.pdf', bbox_inches='tight')

# =============================================================================
# Plot events
# =============================================================================
def gait_events_plot(td, events):
    '''
    This function plots the events in td.
    
    Parameters
    ----------
    td : dict / list of dict, len (n_td)
        trialData structure containing the relevant signals to display.
        
    events : str / list of str, len (n_events)
        Name of the events to mark.
        
    '''
    
    td_c = copy_dict(td)
    
    if type(td_c) == dict:
        td_c = [td_c]
    
    if not is_field(td_c, events):
        raise Exception('ERROR: events fields are not in td!')
    
    for td_tmp in td_c:
        # Open figure
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        # Put title
        if 'File' in td_tmp.keys():
            ax.set_title(td_tmp['File'])
        
        events_line = []
        for event in events:
            if 'R' in event:
                col = [1,0,0]
            elif 'L' in event:
                col = [0,1,1]
            else:
                col = [1,1,0]
            
            if 'FS' in event or 'HS' in event:
                line_style = '-'
            elif 'FO' in event or 'TO' in event:
                line_style = '--'
            else:
                line_style = '-'
            
            if len(td_tmp[event]) > 1000: # Treat it as an array of zeros where the events are ones
                events_2_plot = find_values(td_tmp[event], value = 0.9, method = 'bigger')
            else: # Treat it as an array of events
                events_2_plot = td_tmp[event]
            
            for iEv, ev in enumerate(events_2_plot):
                if iEv == 0:
                    events_line.append(ax.axvline(x = ev, ymin = 0, ymax = 1, color = col + [0.5], linestyle = line_style))
                else:
                    ax.axvline(x = ev, ymin = 0, ymax = 1, color = col + [0.5], linestyle = line_style)
        # Add legend
        ax.legend(events_line,events)



# Main
if __name__ == '__main__':
# =============================================================================
# Stic plot video example
# =============================================================================
    # File to load
    folder = '../data_test/gait'
    file_num = [1]
    file_format = '.mat'
    
    # Load data
    td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format)
    if type(td) is not dict:
        if type(td) is list and len(td) == 1:
            td = td[0]
        else:
            raise Exception('ERROR: td format is neither a dict or a list with len == 1!. Check it!')
    
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
    
    # Plot sticks
    stick_plot_video(td, kin_info, coordinates = ['x','z','y'], step_plot = 10, pause = .1, verbose = True)
    
# EOF
    
    