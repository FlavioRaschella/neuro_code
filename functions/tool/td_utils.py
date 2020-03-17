#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:48 2020

@author: raschell
"""

'''
td is the trial data structure.
The format of td is as following:
    NAME keys : contain signals
    PARAMS key : contain trial data information
        signals : name of the signals to be used for analysis
        time : name of the signal containing time information
        Fs : sampling frequency

'''

# Plot library
import matplotlib.pyplot as plt
# Enumerator library
from enum import Enum
# Numpy library
import numpy as np
# Processing
from processing import artefacts_removal, convert_points_to_target_vector, get_epochs
# Import loading functions
from loading_data import load_data_from_folder

from utils import bipolar, add, flatten_list, group_fields

from power_estimation import moving_pmtm

from filters import butter_bandpass_filtfilt as bpff
from filters import butter_lowpass_filtfilt as lpff
from filters import butter_highpass_filtfilt as hpff

# Plotting events characteristics
class event_color(Enum):
    FS = 'r'
    FO = 'c'

class event_linestyle(Enum):
    R = '-'
    L = '--'

def segment_data(_td, td_epoch):
    '''
    This function segments the trial data in several blocks

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    td_epoch : dict / list of dict

    Returns
    -------
    td_out : dict / list of dict
        Trial data.

    '''
    td_out = []
    
    td = _td.copy()
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
    
    if len(td) != len(td_epoch):
        raise Exception('ERROR: td and td_segment must have the same length!')
    
    for td_tmp, td_epo in zip(td, td_epoch):
        epochs = get_epochs(td_epo['epochs'], td_tmp['params']['Fs'])
        for epoch in epochs:
            td_out_tmp = td_tmp.copy()
            for k,v in td_out_tmp.items():
                if k != 'params':
                    td_out_tmp[k] = np.array(v)[epoch]
                elif 'time' in k:
                    td_out_tmp[k] = np.array(v)[epoch] - np.array(v)[epoch][0]
            # Append new dict
            td_out.append(td_out_tmp)
    
    return td_out

def identify_artefacts(_td, **kwargs):
    '''
    This function identifies the artefacts in the trial data dict(s)

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    fields : list of str
        Fields over which finding the artefacts
    Fs : int
        Sample frequency of the signals
    method : str, optional
        Method for removing the artefacts. The default is amplitude.
    signal_n : int
        Number of signal on which to look for artefacts
    
    Returns
    -------
    td_artefacts : dict / list of dict
        Dictionary of the identified artefacts.

    '''
    
    method = None
    signal_n = None
    threshold = None
    fields = None
    Fs = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'fields':
            fields = value
        elif key == 'Fs':
            Fs = value
        elif key == 'method':
            method = value
        elif key == 'signal_n':
            signal_n = value
        elif key == 'threshold':
            threshold = value
        else:
            print('WARNING: key "{}" not recognised by the identify_artefacts function...'.format(key))

    # Get a temporary copy of td
    td = _td.copy()

    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if fields != None:
        if type(fields) is str:
            fields = [fields]
        if type(fields) is not list:
            raise Exception('ERROR: _str must be a list of strings!')
        if signal_n == None:
            print('Number of signals with artefacts not specified. Setting to len(fields)-1 ...')
            signal_n = len(fields)-1
        
    if method == None:
        method = 'amplitude'
        print('Method for removing artifacts not specified: Selectd method is {}'.format(method))
        
    td_artefacts = []
    for td_tmp in td:
        if fields == None:
            fields = td_tmp['params']['signals']
            if signal_n == None:
                signal_n = len(fields)-1
        if Fs == None:
            Fs = td_tmp['params']['Fs']
        
        data = group_fields(td_tmp,fields)
        td_artefacts.append({'epochs':artefacts_removal(data, Fs, method, signal_n, threshold)})
    
    return td_artefacts

def combine_fields(_td, _fields, **kwargs):
    '''
    This function combines the fields in one dictionary

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    _fields : list of str
        Two fields to combine.
    method : str
        Method for combining the arrays. The default is subtract.
    remove_selected_fields : bool, optional
        Remove the fields before returning the dataset. The default is True.
    inplace : bool, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''

    method = 'subtract'
    remove_selected_fields = True
    save_name_to_params = False
    inplace = True

    # Check input variables
    for key,value in kwargs.items():
        if key == 'method':
            method = value
        elif key == 'remove_selected_fields':
            remove_selected_fields = value
        elif key == 'save_name_to_params':
            save_name_to_params = value
        elif key == 'inplace':
            inplace = value

    if inplace:
        td = _td
    else:
        td = _td.copy()

    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # check _fields input variable        
    if type(_fields) is not list:
        raise Exception('ERROR: _fields must be a list!')
    
    if type(_fields[0]) is not list:
        _fields = [_fields]
    
    for field in _fields:
        if len(field) != 2:
            raise Exception('ERROR: lists in _fields must cointain max 2 strings!')
    
    if method not in ['subtract','multuply','divide','add']:
        raise Exception('ERROR: specified method has not been implemented!')
    
    for td_tmp in td:
        signals_name = []
        for field in _fields:
            if len(field[0]) != len(field[1]):
                raise Exception('ERROR: the 2 arrays must have the same length!')
            
            if method == 'subtract':
                signal_name = '{}-{}'.format(field[0],field[1])
                td_tmp[signal_name] = bipolar(td_tmp[field[0]],td_tmp[field[1]])
            elif method == 'add':
                signal_name = '{}+{}'.format(field[0],field[1])
                td_tmp[signal_name] = add(td_tmp[field[0]],td_tmp[field[1]])
            elif method == 'multuply':
                raise Exception('Method must be implemented!')
            elif method == 'divide':
                raise Exception('Method must be implemented!')
            signals_name.append(signal_name)
            
        if save_name_to_params:
            td_tmp['params']['signals'] = signals_name
    
    if remove_selected_fields:
        remove_fields(td,flatten_list(_fields), exact_field = True, inplace = inplace)
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td


def convert_fields_to_numeric_array(_td, _fields, _vector_target_field, remove_selected_fields = True, inplace = True):
    '''
    This function converts the content of fields in a vector

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    _fields : str / list of str
        Fields from which collect the data to convert.
    _vector_target_field : str
        Field with the name of the array in td to compare to _fields vectors
    remove_selected_fields : bool, optional
        Remove the fields before returning the dataset. The default is True.
    inplace : bool, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''
    
    if inplace:
        td = _td
    else:
        td = _td.copy()

    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # check string input variable
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    if type(_vector_target_field) is not str:
        raise Exception('ERROR: _vector_target_field must be a string!')
    
    # Check that _signals are in the dictionary
    if not is_field(td,_fields):
        raise Exception('ERROR: Selected fields are not in the dict')
    
    for td_tmp in td:
        vector_compare = np.array(td_tmp[_vector_target_field])
        points = [np.array(td_tmp[field]) for field in _fields]
        vector_target = convert_points_to_target_vector(points, vector_compare)
        td_tmp['target'] = vector_target
           
    if remove_selected_fields:
        remove_fields(td,_fields, exact_field = False, inplace = True)
        
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td
    

def combine_dicts(_td1, _td2, inplace = True):
    """        
    This function combine 2 dicts.
    In case of same fields, the first dict dominates
    
    Parameters
    ----------
    _td1 : dict/list of dict
        dict of the trial data
    _td2 : dict/list of dict
        dict of the trial data
    inplace : string, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td_out : dict/list of dict
        trial data variable

    """
    if inplace:
        td1 = _td1
    else:
        td1 = _td1.copy()
    
    td2 = _td2.copy()
    
    
    input_dict = False
    if type(td1) is dict:
        input_dict = True
        td1 = [td1]
    
    if type(td2) is dict:
        td2 = [td2]
    
    # check that tds have the same dimension
    if len(td1) != len(td2):
        raise Exception('ERROR: The 2 tds have different dimension!')
    
    for td1_el, td2_el in zip(td1, td2):
        for k,v in td2_el.items():
            if k not in set(td1_el.keys()):
                td1_el[k] = v
    
    if input_dict:
        td1 = td1[0]
    
    if not inplace:
        return td1

def extract_dicts(_td, fields, inplace = True):
    """        
    This function extracts a dictionary containing only the fields in input.
    
    Parameters
    ----------
    td : dict/list of dict
        dict of the trial data
    fields : str/list of str
        Fields to combine in the new dict
    inplace : string, optional
        Perform operation on the input data dict. The default is False.

    Returns
    -------
    td_out : dict/list of dict
        trial data variable

    """
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    if type(fields) is str:
        fields = [fields]
    
    td_out = []
    for td_tmp in td:
        dict_tmp = dict()
        for field in fields:
            if type(td[field]) is dict:
                combine_dicts(dict_tmp, td[field], inplace = True)
            else:
                raise Exception('ERROR: td[{}] is not a dict!'.format(field))
        td_out.append(dict_tmp)
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td


def remove_fields(_td, _field, exact_field = False, inplace = True):
    '''
    This function removes fields from a dict.

    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data.
    _field : str / list of str
        Fields to remove.
    exact_field : bool, optional
        Look for the exact field name in the dict. The default is False.
    inplace : bool, optional
        Perform operaiton on the input data dict. The default is False.

    Returns
    -------
    td : dict/list of dict
        trial data dict with added items

    '''
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_field) is str:
        _field = [_field]
        
    if type(_field) is not list:
        raise Exception('ERROR: _field must be a list of strings!')
    
    for td_tmp in td:
        td_copy = td_tmp.copy()
        for iStr in _field:
            any_del = False
            for iFld in td_copy.keys():
                if exact_field:
                    if iStr == iFld:
                        del td_tmp[iFld]
                        any_del = True
                else:
                    if iStr in iFld:
                        del td_tmp[iFld]
                        any_del = True
            if not any_del:
                print('Field {} not found. I could not be removed...'.format(iStr))
    
    if input_dict:
        td = td[0]
        
    if not inplace:
        return td


def remove_all_fields_but(_td, _field, exact_field = False, inplace = True):
    '''
    This function removes all fields from a dict but the one selected.

    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data.
    _field : str / list of str
        Field to keep.
    exact_field : bool, optional
        Look for the exact field name in the dict. The default is False.
    inplace : bool, optional
        Perform operaiton on the input data dict. The default is False.

    Returns
    -------
    td : dict/list of dict
        trial data dict with added items

    '''
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_field) is str:
        _field = [_field]
        
    if type(_field) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    for td_tmp in td:
        td_copy = td_tmp.copy()
        for field in td_copy.keys():
            for field_name in _field:
                del_field = True
                if exact_field:
                    if field_name == field:
                        del_field = False
                        break
                else:
                    if field_name in field:
                        del_field = False
                        break
            if del_field == True:
                del td_tmp[field]
    
    if input_dict:
        td = td[0]
        
    if not inplace:
        return td


def add_field_from_dict(_td, _dict, inplace = True, verbose = False):
    '''
    This function adds fields in a dict to another dict.
    
    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data
    _dict : dict
        dict to add to trial data
    inplace : str, optional
        Perform operation on the input data dict. The default is False.
    verbose : bool, optional
        Describe what's happening in the code. The default is False.

    Returns
    -------
    td_out : dict/list of dict
        trial data dict with added items

    '''
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    # check dict input variable
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_dict) is not dict:
        raise Exception('ERROR: _dict input must be a dictionary!')
        
    for td_tmp in td:
        if set(_dict.keys()) in set(td_tmp.keys()):
            if verbose:
                print('Field {} already existing in td. NOT adding!')
        else:
            for k,v in _dict.items():
                td_tmp[k] = v
        
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td


def is_field(_td, _fields, verbose = False):
    '''
    This function checks whether fields are in a dict.
    
    Parameters
    ----------
    _td : dict / list of dict
        dict of trial data.
    _fields : string / list of strings
        Fields in the trial data dict.
    verbose : bool, optional
        Describe what's happening in the code. The default is False.

    Returns
    -------
    return_val : bool
        Return whether fields are in the trial data dict or not.

    '''
    
    return_val = True
    
    # check dict input variable
    if type(_td) is dict:
        _td = [_td]
        
    if type(_td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    # Flatten list of fields
    _fields = flatten_list(_fields)
    
    for idx, td_tmp in enumerate(_td):
        for field in _fields:
            if field not in td_tmp.keys():
                return_val = False
                if verbose:
                    print('Field {} not in dict #{}'.format(field, idx))
    
    return return_val
    # End of is_field function

def get_field(_td, _signals, save_signals_name = False):
    '''
    This function get fields from the trial data and place them in a another dict
    
    Parameters
    ----------
    _td : dict
        dict from which we collect the fields.
    _signals : str/list of str
        Fields to collect from the dict.
    save_signals_name : bool
        Add a field containing the name of the fileds. The default is False.
        
    Returns
    -------
    td_out : dict
        New dict containing the selected fields.

    '''
    td = _td.copy()
    td_out = []
    
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
    
    if type(_signals) == str:
        print('Signals input must be a list. You inputes a string --> converting to list...')
        _signals = [_signals]
    
    # Check that _signals are in the dictionary
    if not is_field(td,_signals):
        raise Exception('ERROR: Selected signals are not in the dict')
    
    # Loop over the trials
    for td_tmp in td:
        td_out_tmp = dict()
        # List of signals name
        signals_name = []
        # Loop over the signals
        for sgl in _signals:
            if type(sgl) == list:
                signal_name = '{} - {}'.format(sgl[0],sgl[1])
                signals_name.append(signal_name)
                td_out_tmp[signal_name] = np.array(td_tmp[sgl[0]]) - np.array(td_tmp[sgl[1]])
            else:
                signal_name = sgl
                signals_name.append(signal_name)
                td_out_tmp[signal_name] = np.array(td_tmp[sgl])
        
        if save_signals_name:
            td_out_tmp['signals_name'] = signals_name
        
        td_out.append(td_out_tmp)
    
    if input_dict:
        td_out = td_out[0]
        
    return td_out


def td_plot(td, y, **kwargs):
    '''
    This function plots signals from the td dict.

    Parameters
    ----------
    td : dict
        Trial data.
    y : str/list of str
        Signals to plot.
    x : str/list of str
        Signal to use for x axis.
    subplot : tuple
        Structure of the suvplots in the figure.
    events : str/list of str
        Gait events to plot.
    title : str
        Title of the plot.
    xlim : tuple
        x min and max.
    ylim : tuple
        y min and max.
    save : bool
        Flag for saving the figure.

    Example:
        td_plot(td,['LFP_BIP7','LFP_BIP9'], events = ['RFS','LFS'], subplot = (2,1))

    '''
    
    # Input variables
    subplot = ()
    events = None
    title = None
    x = None
    ylim = None
    xlim = None
    ylabel = None
    xlabel = None
    colours = None
    grid_plot = True
    axs_external = []
    maximise = False
    save_figure = False
    save_format = 'pdf'
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'events':
            events = value
        elif key == 'subplot':
            subplot = value
        elif key == 'x':
            x = value
        elif key == 'title':
            title = value
        elif key == 'grid_plot':
            grid_plot = value
        elif key == 'axs':
            axs_external = value
        elif key == 'ylabel':
            ylabel = value
        elif key == 'xlabel':
            xlabel = value
        elif key == 'ylim':
            ylim = value
        elif key == 'xlim':
            xlim = value
        elif key == 'colours':
            colours = value
        elif key == 'maximise':
            maximise = value
        elif key == 'save':
            save_figure = True
            save_name = value
        elif key == 'save_format':
            save_format = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))

    # Check input variables
    if type(td) is not dict:
        raise Exception('ERROR: td type must be a dict! It is a {}'.format(type(td)))
        
    if type(y) is str:
        y = [y]
        
    # Check whether _signals elements are in _td
    if not(is_field(td, y)):
        raise Exception('ERROR: _signals must be in _td!')

    # Check whether gait_field elements are in _td
    if events != None:
        if not(is_field(td, events)):
            raise Exception('ERROR: {} must be in _td!'.format(events))
    
    # Get min signal number in a plot
    min_signal_n = 1
    for signal in y:
        if type(signal) == list and len(signal) > min_signal_n:
            min_signal_n = len(signal)
    
    if type(save_format) is str:
        save_format = [save_format]
        
    if type(save_format) is not list:
        raise Exception('ERROR: type(save_format) is not a list!')
        
    # Figure
    if len(axs_external) == 0:
        # Check subplot dimension
        if not(subplot):
            subplot = (len(y),1)
            fig, axs = plt.subplots(nrows = subplot[0], ncols = subplot[1], sharex=True)
    else:
        axs = axs_external
    
    if len(axs) != len(y):
        raise Exception('ERROR: axs and y have different length: axs: {} != y: {}'.format(len(axs),len(y)))
    
    # Create figure
    if title != None:
        plt.suptitle(title)
    
    if maximise:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    
    if type(axs).__module__ != np.__name__:
        axs  = [axs]
    
    # Prepare plotting variables
    # Check x axis
    if x == None:
        x_list = [None] * len(y)
    elif x != None and type(x) == str:
        x_list = [x] * len(y)
    elif x != None and type(x) == list:
        if len(x) == len(y):
            x_list = x
        else:
            raise Exception('ERROR: x list have different length from y list!')
    else:
        raise Exception('ERROR: x must be None, str ot list of str!')
    
    # Check colours axis
    if colours == None:
        c_list = [[None] * min_signal_n] * len(y)
    elif colours != None and type(colours) == list and type(colours[0]) != list:
        c_list = [colours] * len(y)
    elif colours != None and type(colours) == list and type(colours[0]) == list:
        c_list = colours
    
    # print(x_list,'\n',y,'\n',axs,'\n',c_list)
    
    for x_ax, signal, ax, col in zip(x_list, y, axs, c_list):
        
        # Set plot title
        # print(signal)
        if type(signal) is list:
            signal_name = ' + '.join(signal)
        else:
            signal_name = signal
                
        ax.set_title(signal_name)
        
        # Set plot labels
        if xlabel != '':
            ax.set_xlabel('')
        if ylabel != '':
            ax.set_ylabel('')
        
        # Set axes limits
        if ylim != None:
            ax.set_ylim(ylim)
        else:
            if type(signal) is list:
                ylim_tmp = [+np.inf, -np.inf]
                for sig in signal:
                    if min(td[sig]) < ylim_tmp[0]:
                        ylim_tmp[0] = min(td[sig])
                    if max(td[sig]) > ylim_tmp[1]:
                        ylim_tmp[1] = max(td[sig])
            else:
                ylim_tmp = tuple([min(td[signal]), max(td[signal])])
            ax.set_ylim(ylim_tmp)
        if xlim != None:
            ax.set_xlim(xlim)
        
        # Set plot grid
        ax.grid(grid_plot)
        
        # Plot signal
        if type(signal) is list:
            for sig, c in zip(signal, col):
                if x_ax == None:
                    if c == None:
                        ax.plot(td[sig])
                    else:
                        ax.plot(td[sig],color = c)
                else:
                    if c == None:
                        ax.plot(td[x_ax],td[sig])
                    else:
                        ax.plot(td[x_ax],td[sig],color = c)
        else:
            if x_ax == None:
                if col[0] == None:
                    ax.plot(td[signal])
                else:
                    ax.plot(td[signal],color = col)
            else:
                if col[0] == None:
                    ax.plot(td[x_ax],td[signal])
                else:
                    ax.plot(td[x_ax],td[signal],color = col)
        
        # Plot gait events
        if events != None:
            for event in events:
                if 'R' in event:
                    line_style = event_linestyle.R.value
                else:
                    line_style = event_linestyle.L.value
                    
                if 'FS' in event:
                    col = event_color.FS.value
                else:
                    col = event_color.FO.value
                    
                for ev in td[event]:
                    ax.axvline(ev,ylim[0], ylim[1], color = col, linestyle = line_style)
    
    # Set tight plot
    plt.tight_layout()
    
    if save_figure:
        for form in save_format:
            if form == 'pickle':
                import pickle
                pickle.dump(fig, open(save_name +'.pickle', 'wb'))
            elif form == 'pdf':
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                fig.savefig(save_name + '.pdf', bbox_inches='tight')
            else:
                raise Exception('ERROR: wrong save format assignaed!')
    
    if len(axs_external) == 0:
        return fig, axs

def add_params(td, **kwargs):
    '''
    This function adds parameters to the trial data dictionary

    Parameters
    ----------
    td : dict / list of dict
        dict of trial data.
    **kwargs : 
        items to add to the td dict
        
    Returns
    -------
    td : dict / list of dict.
        dict of trial data.
    '''
    
    input_dict = False
    if type(td) == dict:
        input_dict = True
        td = [td]
        
    # Loop over the trials
    for td_tmp in td:
        td_tmp['params'] = {}
        # Check input variables
        for key,value in kwargs.items():
            td_tmp['params'][key] = value

    if input_dict:
        td = td[0]
        
    return td

def combine_epochs(td_1, td_2):
    '''
    This function combines the good epochs from 2 dict using the logic AND.

    Parameters
    ----------
    td_1 : dict / list of dict
        trial data with epoch field.
    td_2 : dict / list of dict
        trial data with epoch field.

    Returns
    -------
    td_epoch : dict / list of dict
        Dict aving as only field 'epochs', a binary array.

    '''
    td_epoch = []
    if len(td_1) != len(td_2):
        raise Exception('ERROR: artefacts and segments trials data have different length!')
    for art, seg in zip(td_1, td_2):
        if len(art['epochs']) != len(seg['epochs']):
            raise Exception('ERROR: artefacts[epochs] and segments [epochs] have different length!')
        td_epoch.append({'epochs': np.logical_and(np.array(art['epochs']), np.array(seg['epochs'])).astype('int')})
    
    return td_epoch

def compute_multitaper(_td, **kwargs):
    '''
    This function compute the multitapers spectal analysis for the td signals

    Parameters
    ----------
    _td : dict / list of dict
        trial data with epoch field.
    **kwargs : dict
        Additional information for computing multitaper.

    Returns
    -------
    td : dict / list of dict
        trial data with signals analysed based on the multitaper parameters.

    '''
    # Multitaper information
    window_size_sec = 0.25 # in seconds
    window_step_sec = 0.01 # in seconds
    freq_min = 10
    freq_max = 100
    NW = 4
    inplace = True
    verbose = False

    # Check input variables
    for key,value in kwargs.items():
        if key == 'wind_size':
            window_size_sec = value
        elif key == 'wind_step':
            window_step_sec = value
        elif key == 'freq_start':
            freq_min = value
        elif key == 'freq_stop':
            freq_max = value
        elif key == 'NW':
            NW = value
        elif key == 'inplace':
            inplace = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    # check dict input variable
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
        
    # Compute number of tapers
    tapers_n = np.floor(2*NW)-1
    # Get frequency range
    freq_range = [freq_min , freq_max]
    
    for td_tmp in td:
        Fs = td_tmp['params']['Fs']
        
        # Get window's info in samples
        window_size_smp = round(window_size_sec * Fs)
        window_step_smp = round(window_step_sec * Fs)
        
        for iSgl, signal in enumerate(td_tmp['params']['signals']):
            if verbose:
                print('Processing signal {}/{}'.format(iSgl+1, len(td_tmp['signal_names'])))
            td_tmp[signal], sfreqs, stimes = moving_pmtm(td_tmp[signal], Fs, window_size_smp, window_step_smp, freq_range, NW=NW, NFFT=None, verbose=verbose)
        
        # Update frequency info
        td_tmp['params']['Fs'] = 1/(stimes[1] - stimes[0])
        
        # Re-map target dataset
        if 'target' in set(td_tmp.keys()):
            target_new = np.zeros(stimes.shape)
            for ev in np.unique(td_tmp['target'])[1:]:
                target_new += np.histogram(np.where(td_tmp['target'] == ev), bins = stimes.shape[0], range = (0, len(td_tmp['target'])))[0]
            if (target_new > np.unique(td_tmp['target'])[-1]).any():
                raise Exception('WARNING: Multiple classes are falling in the mutitaping binning!')
            td_tmp['target'] = target_new
            
        td_tmp[td_tmp['params']['time']] = stimes
        td_tmp['params']['freq'] = 'freq'
        td_tmp['freq'] = sfreqs
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td


def compute_filter(_td, **kwargs):
    # Filtering information
    kind = None
    f_min = None
    f_max = None
    order = 5
    inplace = True
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'kind':
            kind = value
        elif key == 'f_min':
            f_min = value
        elif key == 'f_max':
            f_max = value
        elif key == 'order':
            order = value
        elif key == 'verbose':
            verbose = value
        elif key == 'inplace':
            inplace = value
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    if inplace:
        td = _td
    else:
        td = _td.copy()
    
    input_dict = False
    # check dict input variable
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    for td_tmp in td:
        Fs = td_tmp['params']['Fs']
        
        for iSgl, signal in enumerate(td_tmp['params']['signals']):
            if verbose:
                print('Filtering signal {}/{}'.format(iSgl+1, len(td_tmp['signal_names'])))
            if kind == 'bandpass':
                td_tmp[signal] = bpff(data = td_tmp[signal], lowcut = f_min, highcut = f_max, fs = Fs, order=order)
            elif kind == 'lowpass':
                td_tmp[signal] = lpff(data = td_tmp[signal], lowcut = f_min, fs = Fs, order=order)
            elif kind == 'highpass':
                td_tmp[signal] = hpff(data = td_tmp[signal], highcut = f_max, fs = Fs, order=order)
            else:
                raise Exception('ERROR: wrong kind of filter applied! Kind given is : {}'.format(kind))
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td

def load_pipeline(data_path, data_files, data_format, **kwargs):
    '''
    This funtion loads and organises data from 

    Parameters
    ----------
    data_path : str / list of str
        Path(s) of the folder(s) with the data.
    data_files : int / list of int
        Number of the files to load.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    target_load_dict = None
    remove_fields_dict = None
    remove_all_fields_but_dict = None
    convert_fields_to_numeric_array_dict = None
    params = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'trigger_file':
            target_load_dict = value
        elif key == 'remove_fields':
            remove_fields_dict = value
        elif key == 'remove_all_fields_but':
            remove_all_fields_but_dict = value
        elif key == 'convert_fields_to_numeric_array':
            convert_fields_to_numeric_array_dict = value
        elif key == 'params':
            params = value
            
    
    # Load data
    td = load_data_from_folder(folder = data_path,file_num = data_files,file_format = data_format)

    if remove_fields_dict != None:
        remove_fields(td, remove_fields_dict['fields'], exact_field = False, inplace = True)
    
    if remove_all_fields_but_dict != None:
        remove_all_fields_but(td, remove_all_fields_but_dict['fields'], exact_field = True, inplace = True)
    
    if target_load_dict != None:
        options = remove_fields(target_load_dict, ['path', 'files', 'file_format'], exact_field = False, inplace = False)
        td_target = load_data_from_folder(folder = target_load_dict['path'],
                                          file_num = target_load_dict['files'],
                                          file_format = target_load_dict['file_format'],
                                          **options)
        if 'fields' in options.keys():
            remove_all_fields_but(td_target,options['fields'],False,True)
    
        # Combine target data with the predictor data
        combine_dicts(td, td_target, inplace = True)
    
    if convert_fields_to_numeric_array_dict != None:
        convert_fields_to_numeric_array(td, _fields = convert_fields_to_numeric_array_dict['fields'], 
                                        _vector_target_field = convert_fields_to_numeric_array_dict['target_vector'],
                                        remove_selected_fields = True, inplace = True)    
    
    if params != None:
        add_params(td, **params)
    
    return td
    

def cleaning_pipeline(td, **kwargs):
    '''
    This function cleans the dataset by:
        combining signals
        removing artefacts
        segmenting the data

    Parameters
    ----------
    td : dict / list of dict
        Trial data.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict / list of dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    combine_fields_dict = None
    remove_artefacts_dict = None
    add_segmentation_dict = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'combine_fields':
            combine_fields_dict = value
        elif key == 'remove_artefacts':
            remove_artefacts_dict = value
        elif key == 'add_segmentation':
            add_segmentation_dict = value
    
    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    if combine_fields_dict != None:
        combine_fields(td, combine_fields_dict['fields'], **combine_fields_dict)
        
    if remove_artefacts_dict != None:
        td_artefacts = identify_artefacts(td, **remove_artefacts_dict)
        
    if add_segmentation_dict != None:
        td_segment = add_segmentation_dict['epochs']
    
    # Combine artefacts and segmentation
    if remove_artefacts_dict != None and add_segmentation_dict != None:
        td_epoch = combine_epochs(td_artefacts, td_segment)
    elif remove_artefacts_dict == None and add_segmentation_dict != None:
        td_epoch = td_segment
    elif remove_artefacts_dict != None and add_segmentation_dict == None:
        td_epoch = td_artefacts
    
    if remove_artefacts_dict != None or add_segmentation_dict != None:
        td = segment_data(td, td_epoch)
    
    return td


def preprocess_pipeline(td, **kwargs):
    '''
    This function preprocess the dataset.

    Parameters
    ----------
    td : dict / list of dict
        Trial data.
    **kwargs : dict
        Additional information for organising the data

    Returns
    -------
    td : dict / list of dict
        Trial data organised based on input requests.

    '''
    
    # Input variables
    operations = []
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'filter':
            operations.append((key,value))
        elif key == 'multitaper':
            operations.append((key,value))
        else:
            print('WARNING: key "{}" not recognised by the compute_multitaper function...'.format(key))
    
    for operation in operations:
        if operation[0] == 'filter':
            compute_filter(td, **operation[1])
        elif operation[0] == 'multitaper':
            compute_multitaper(td, **operation[1])
    
    return td


if __name__ == '__main__':
    td_test = {'test1': np.arange(10), 'test2': [1,2,3,4,5,6]}
    td_test2 = {'test1': np.arange(20), 'test3': [1,2,3,4,5,6]}
    td_test_list = [td_test,td_test]
    
    # Test combine_dicts
    td_comb = combine_dicts(td_test, td_test2, inplace = False)
    if set(['test1','test2','test3']) == td_comb.keys() and (td_comb['test1'] == td_test['test1']).all():
        print('Test combine_dicts passed!')
    else:
        raise Exception('ERROR: Test combine_dicts NOT passed!')
    
    # Test is_field
    if is_field(td_test,'test1') and is_field(td_test,'test3',False) == False and is_field(td_test_list,'test1') and is_field(td_test_list,'test3',False) == False:
        print('Test is_field passed!')
    else:
        raise Exception('ERROR: Test is_field NOT passed!')
    
    # Test get_field
    td_res = get_field(td_test, ['test1'])
    if 'test1' in td_res:
        print('Test get_field passed!')
    else:
        raise Exception('ERROR: Test get_field NOT passed!')
        
    # Test add_fields
    td_new = add_field_from_dict(td_test,{'test3': [1,3,4]}, inplace = False)
    if 'test3' in td_new.keys():
        print('Test add_field_from_dict passed!')
    else:
        raise Exception('ERROR: Test add_field_from_dict NOT passed!')
    
    # Test convert_fields_to_numeric_array
    td_test_cf2n = {'test1': np.arange(10), 'id1': [1,3,5], 'id2': [2,4]}
    td_new = convert_fields_to_numeric_array(td_test_cf2n, ['id1','id2'], 'test1', remove_selected_fields = True)
    if (td_new['target'] - np.array([0,0,1,2,1,2,1,0,0,0])>0.1).any() or is_field(td_new,['id1','id2']):
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test find_first passed!')
    
    
    td_new1 = remove_all_fields_but(td_test, 'test1', exact_field = True, inplace = False)
    td_new2 = remove_all_fields_but(td_test, 'test', exact_field = False, inplace = False)
    if set(td_new1.keys()) != set(['test1']) or set(td_new2.keys()) != set(['test1','test2']):
        raise Exception('ERROR: Test find_first NOT passed!')
    else:
        print('Test remove_all_fields_but passed!')
    
    # Test combine_fields
    td_test_combine = {'test1': np.arange(10), 'test2': np.arange(10)}
    td_new = combine_fields(td_test_combine,['test1','test2'],method = 'subtract')
    if 'test1-test2' in td_new.keys() and (td_new['test1-test2'] - np.zeros(10) < 0.1).all():
        print('Test combine_fields passed!')
    else:
        raise Exception('ERROR: Test combine_fields NOT passed!')
        
        
    
    del td_test, td_test2, td_test_list, td_comb, td_res, td_new, td_test_cf2n, td_new1, td_new2, td_test_combine
    print('All implemented tests passed!')
    
    