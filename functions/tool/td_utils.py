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

from utils import bipolar, add, flatten_list

# Plotting events characteristics
class event_color(Enum):
    FS = 'r'
    FO = 'c'

class event_linestyle(Enum):
    R = '-'
    L = '--'

# def flatten_dict(td):
#     # Stop field loop flag
#     is_not_all_list = True
                             
#     # Loop over fields.                    
#     while is_not_all_list:
#         # Loop internal check over single field
#         is_list = True
#         # Extract only the requested fields
#         for key, val in td.items():
#             if '__' not in key:
#                 # break
#                 print(key, val.dtype.char); print(' ')
#                 #  Check type of variable
#                 if type(val) == str and key not in td_out.keys():
#                     td_out[key] = val
#                 elif type(val) == int and key not in td_out.keys():
#                     td_out[key] = val
#                 elif type(val) == float and key not in td_out.keys():
#                     td_out[key] = val
#                 elif type(val) == list and key not in td_out.keys():
#                     td_out[key] = val
#                 elif type(val) == np.ndarray:
#                     val_red = reduce_mat_object(val)
                    
#                     if is_mat_struc(val_red):
#                         for key_struct, val_struct in zip(val_red.dtype.names, val_red.item()):
#                             val_struct_red = reduce_mat_object(val_struct)
#                             if is_mat_struc(val_struct_red):
#                                 # print(key_struct)
#                                 is_list = False
#                                 td_out[key + '_' + key_struct] = val_struct_red
#                             else:
#                                 td_out[key + '_' + key_struct] = reduce_list(np.ndarray.tolist(invert_to_column(val_struct_red)))
#                     else:
#                         td_out[key] = reduce_list(np.ndarray.tolist(invert_to_column(val_red)))
#         if is_list:
#             is_not_all_list = False
#         else:
#             # fields_name = td_out.keys()
#             td = td_out.copy()
#             td_out = {}

# =============================================================================
# Control fields
# =============================================================================
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

# =============================================================================
# Removing
# =============================================================================
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


# =============================================================================
# Adding
# =============================================================================
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
        if 'params' not in td_tmp.keys():
            td_tmp['params'] = {}
        # Check input variables
        for key,value in kwargs.items():
            td_tmp['params'][key] = value

    if input_dict:
        td = td[0]
        
    return td

# CONSIDER REMOVING
# def add_field_from_dict(_td, _dict, inplace = True, verbose = False):
#     '''
#     This function adds fields in a dict to another dict.
    
#     Parameters
#     ----------
#     _td : dict / list of dict
#         dict of the trial data
#     _dict : dict
#         dict to add to trial data
#     inplace : str, optional
#         Perform operation on the input data dict. The default is False.
#     verbose : bool, optional
#         Describe what's happening in the code. The default is False.

#     Returns
#     -------
#     td_out : dict/list of dict
#         trial data dict with added items

#     '''
    
#     if inplace:
#         td = _td
#     else:
#         td = _td.copy()
    
#     input_dict = False
#     # check dict input variable
#     if type(td) is dict:
#         input_dict = True
#         td = [td]
        
#     if type(td) is not list:
#         raise Exception('ERROR: _td must be a list of dictionaries!')
    
#     # check string input variable
#     if type(_dict) is not dict:
#         raise Exception('ERROR: _dict input must be a dictionary!')
        
#     for td_tmp in td:
#         if set(_dict.keys()) in set(td_tmp.keys()):
#             if verbose:
#                 print('Field {} already existing in td. NOT adding!')
#         else:
#             for k,v in _dict.items():
#                 td_tmp[k] = v
        
#     if input_dict:
#         td = td[0]
    
#     if not inplace:
#         return td

# =============================================================================
# Combine
# =============================================================================
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
  
    
def combine_dicts(_td, _td2copy, inplace = True):
    """        
    This function combine 2 dicts.
    In case of same fields, the first dict dominates
    
    Parameters
    ----------
    _td : dict/list of dict
        dict of the trial data
    _td2copy : dict/list of dict
        dict of the trial data
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
    
    td2copy = _td2copy.copy()
    
    
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
    
    if type(td2copy) is dict:
        td2copy = [td2copy]
    
    # check that tds have the same dimension
    if len(td) != len(td2copy):
        raise Exception('ERROR: The 2 tds have different dimension!')
    
    for td1_el, td2_el in zip(td, td2copy):
        for k,v in td2_el.items():
            if k not in set(td1_el.keys()):
                td1_el[k] = v
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td

# =============================================================================
# Extract
# =============================================================================
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
            td_out_tmp['params'] = dict()
            td_out_tmp['params']['signals'] = signals_name
        
        td_out.append(td_out_tmp)
    
    if input_dict:
        td_out = td_out[0]
        
    return td_out


def extract_dicts(_td, fields, **kwargs):
    """        
    This function extracts a dictionary containing only the fields in input.
    
    Parameters
    ----------
    td : dict/list of dict
        dict of the trial data
    fields : str/list of str
        Fields to combine in the new dict
    keep_name : str, optional
        Keep the name from the dict in the new keys (e.g. LFP: BIP1 --> LFP_BIP1). The default is True.
    inplace : str, optional
        Perform operation on the input data dict. The default is True.

    Returns
    -------
    td_out : dict/list of dict
        trial data variable

    """
    
    keep_name = True
    inplace = True
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'keep_name':
            keep_name = value
        if key == 'inplace':
            inplace = value
    
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
            if type(td_tmp[field]) is dict:
                dict_new = td_tmp[field]
                if keep_name:
                    dict2add = dict()
                    for k,v in dict_new.items():
                        dict2add[field+'_'+k] = v
                else:
                    dict2add = dict_new
                combine_dicts(dict_tmp, dict2add, inplace = True)
            else:
                raise Exception('ERROR: td[{}] is not a dict!'.format(field))
        td_out.append(dict_tmp)
    
    if input_dict:
        td = td_out[0]
    
    if not inplace:
        return td_out

# =============================================================================
# Plotting
# =============================================================================
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
    # td_new = add_field_from_dict(td_test,{'test3': [1,3,4]}, inplace = False)
    # if 'test3' in td_new.keys():
    #     print('Test add_field_from_dict passed!')
    # else:
    #     raise Exception('ERROR: Test add_field_from_dict NOT passed!')
    
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
    
    