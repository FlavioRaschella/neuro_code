#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:01:44 2020

@author: raschell

This function is used for the manual selection of the gait events
"""

# Plot library
import matplotlib.pyplot as plt
# Filt library
from filters import butter_lowpass_filtfilt
# Import data processing
from td_utils import is_field
# Import numpy
import numpy as np
# Import utils
from utils import flatten_list
# Save path
from scipy.io import savemat

def gait_event_manual(td, signals, events, **kwargs):
    '''
    This function helps selecting the gait events manually by displaying some preselected signals.
    This function saves a file with marked gait events.
    
    Parameters
    ----------
    td : dict / list of dict
        trialData structure containing the relevant signals to display.
    signals : str / list of str
        Signals to display for marking the events.
    signal_interval : int / float, optional
        Interval (in seconds) of signal to display in the plot. The default is 30 seconds.
    field_Fs : str, optional
        Field in td containing frequency information. The default value is 'Fs'.
    field_time : int / float, optional
        Field in td containing time information.The default value is 'time'.
    offset : int / float, optional
        Offset for shitfting the gait events. The offset must be in samples.The default value is None.
    output_type : str, optional
        Set whether the output values should be in 'time' or 'samples'.The default value is 'time'.

    '''
    # Input variables
    save_name = ''
    verbose = False
    signal_interval_sec = 30
    sampling_freq_field = 'Fs'
    time_field = 'time'
    offset = None
    output_type = 'time'
    correct_data_flag = False
    kin_plot = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'save_name':
            save_name = value
        elif key == 'signal_interval':
            signal_interval_sec = value
        elif key == 'field_Fs':
            sampling_freq_field = value
        elif key == 'field_time':
            time_field = value
        elif key == 'offset':
            offset = value
        elif key == 'output_type':
            output_type = value
        elif key == 'kin_plot':
            kin_plot = value
            correct_data_flag = True
        elif key == 'verbose':
            verbose = value
    
    if type(save_name) is not str:
        raise Exception('ERROR: save_name must be a string. You inputed a {}'.format(type(save_name)))
    if type(verbose) is not bool:
        raise Exception('ERROR: verbose must be a bool. You inputed a {}'.format(type(verbose)))
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # Check that signals are in trialData structure
    if not is_field(td,signals):
        raise Exception('ERROR: Missing fields in the dictionaries...')
    
    # Set gait event variable
    gait_events = []
    
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
        
    if is_field(td[0],sampling_freq_field):
        Fs = td[0][sampling_freq_field]
    elif is_field(td[0],time_field):
        Fs = 1/(td[0][time_field][1] - td[0][time_field][0])
    else:
        raise Exception('ERROR: Missing frequency information from the input dictionary!')
    signal_interval_smp = np.round(signal_interval_sec*Fs).astype('int')
    
    if signal_interval_smp>signals_len:
        raise Exception('ERROR: selected interval to plot is > than signal length. {}>{}'.format(signal_interval_smp,signals_len))
    
    #%% Plot data
    fig, axs = plt.subplots(nrows = len(signals), ncols = 1, sharex=True)
    # Loop over the dictionaries
    for iTd, td_tmp in enumerate(td):
        # Gait event array for the current file
        gait_events_tmp = dict()
        for ev in events:
            gait_events_tmp[ev] = []
            
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
            for iSig, (signal,name,ylim) in enumerate(zip(signal_data,signal_name,signal_ylim)):
                axs[iSig].plot(signal[np.arange(intervals[iInt],intervals[iInt+1]), :])
                axs[iSig].set_ylim(ylim)
                axs[iSig].legend(loc='upper right', labels = name)
                if iSig != 0:
                    axs[iSig].title.set_text(' + '.join(name))
                
            # Set the events
            for iEv, ev in enumerate(events):
                
                axs[0].title.set_text('Select {} event. '.format(events[iEv]) + ' + '.join(signal_name[0]))   
     
                fig.canvas.draw()           
                pts = plt.ginput(-1, timeout=0, show_clicks = True, mouse_add=1, mouse_pop=3)
                if len(pts) != 0:
                    gait_events_tmp[ev].append(intervals[iInt] + np.round(np.asarray(pts)[:,0]).astype('int'))
    
                # if iEv+1 != len(events_to_select):
                #     axs[0].title.set_text('Select {} event. '.format(events_to_select[iEv+1]) + ' + '.join(signal_name))
                    # fig.suptitle('File {}\n Select {} event. Press ESC to finish.'.format(td_tmp['File'],events_to_select[iEv+1]), fontsize=10)
        
            for iSig, signal in enumerate(signals):
                axs[iSig].cla()
        
        for ev in events:
            if len(gait_events_tmp[ev]) != 0:
                gait_events_tmp[ev] = np.concatenate(gait_events_tmp[ev])
            else:
                gait_events_tmp[ev] = np.array([])
        gait_events.append(gait_events_tmp)
    
    plt.close()
    #%% Correct the data using stickplot
    
    if correct_data_flag:
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
            for event_name in events:
                if 'R' in event_name:
                    leg_variables_forward_names = kin_plot['Right']['forward']
                    leg_variables_vertica_names = kin_plot['Right']['vertical']
                elif 'L' in event_name:
                    leg_variables_forward_names = kin_plot['Left']['forward']
                    leg_variables_vertica_names = kin_plot['Left']['vertical']
                    
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
                        if X.shape[0]-event >=50:
                            event_stop = event+50
                        else:
                            event_stop = X.shape[0]
                        if event-50 >=0:
                            event_start = event-50
                        else:
                            event_start = 0
                        event_interval = np.arange(event_start,event_stop)
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
                            if event < X.shape[0]:
                                event += 1
                            if verbose:
                                print('-->')
                        elif key == -1:
                            if event > 0:
                                event -= 1
                            if verbose:
                                print('<--')
                        elif key == 2:
                            not_stop_loop = False
                            if verbose:
                                print('ESC')
                        elif key == 3:
                            event_to_remove.append(iEv)
                            not_stop_loop = False
                            if verbose:
                                print('DEL')
                        else:
                            print('Possible keys are: left, right, delete, escape.\nTry again...')
                        
                        plt.cla()
                    
                    gait_events_tmp[event_name][iEv] = event
                gait_events_tmp[event_name] = np.delete(gait_events_tmp[event_name], event_to_remove)
        plt.close()
        
    #%% Correct data with offset if present
    if offset != None:
        for gait_event in gait_events:
            for event in events:
                gait_event[event] = gait_event[event] - offset
            
    #%% Adjust outcome type
    if output_type == 'time':
        for gait_event in gait_events:
            for event in events:
                gait_event[event] = gait_event[event]/Fs
    
    #%% Save the data
    for idx, (td_tmp, gait_event) in enumerate(zip(td, gait_events)):
        savemat('gait_events_' + str(idx) + save_name + '.mat', gait_event, appendmat=True)

#%% Main
if __name__ == '__main__':
    print('WARNING: example not implemented!')
    
# EOF
    
    