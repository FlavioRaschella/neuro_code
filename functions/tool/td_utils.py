#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:48 2020

@author: raschell
"""

# Plot library
import matplotlib.pyplot as plt
# Enumerator library
from enum import Enum
# Numpy library
import numpy as np
# Processing
from processing import artefacts_removal, convert_points_to_target_vector
# Import loading functions
from loading_data import load_data_from_folder

from utils import bipolar, add, flatten_list, group_fields

# Plotting events characteristics
class event_color(Enum):
    FS = 'r'
    FO = 'c'

class event_linestyle(Enum):
    R = '-'
    L = '--'

def identify_artefacts(_td, _fields, _Fs, **kwargs):
    '''
    This function identifies the artefacts in the trial data dict(s)

    Parameters
    ----------
    _td : dict / list of dict
        Trial data.
    _fields : list of str
        Fields over which finding the artefacts
    _Fs : int
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
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'method':
            method = value
        if key == 'signal_n':
            signal_n = value
        if key == 'threshold':
            threshold = value

    # Get a temporary copy of td
    td = _td.copy()

    # check dict input variable
    if type(td) is dict:
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    # check string input variable
    if type(_fields) is str:
        _fields = [_fields]
        
    if type(_fields) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
        
    if method == None:
        method = 'amplitude'
        print('Method for removing artifacts not specified: Selectd method is {}'.format(method))
        
    if signal_n == None:
        print('Number of signals with artefacts not specified. Setting to len(_fields)-1 ...')
        signal_n = len(_fields)-1
        
    td_artefacts = []
    for td_tmp in td:
        data = group_fields(td_tmp,_fields)
        td_artefacts.append({'good_idx':artefacts_removal(data, _Fs, method, signal_n, threshold)})
    
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
        Perfrom operation on the input data dict. The default is False.

    Returns
    -------
    td : dict / list of dict
        Trial data.

    '''

    method = 'subtract'
    remove_selected_fields = True
    inplace = False

    # Check input variables
    for key,value in kwargs.items():
        if key == 'method':
            method = value
        elif key == 'remove_selected_fields':
            remove_selected_fields = value
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
          
    if remove_selected_fields:
        remove_fields(td,flatten_list(_fields), exact_field = True)
    
    if input_dict:
        td = td[0]
    
    if not inplace:
        return td


def convert_fields_to_numeric_array(_td, _fields, _vector_target_field, remove_selected_fields = True, inplace = False):
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
        Perfrom operation on the input data dict. The default is False.

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
    

def combine_dicts(_td1, _td2, inplace = False):
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
        Perfrom operation on the input data dict. The default is False.

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


def remove_fields(_td, _field, exact_field = False, inplace = False):
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
        Perfrom operaiton on the input data dict. The default is False.

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


def remove_all_fields_but(_td, _field, exact_field = False, inplace = False):
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
        Perfrom operaiton on the input data dict. The default is False.

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


def add_field(_td, _dict, inplace = False, verbose = False):
    '''
    This function adds fields to a dict.
    
    Parameters
    ----------
    _td : dict / list of dict
        dict of the trial data
    _dict : dict
        dict to add to trial data
    inplace : str, optional
        Perfrom operation on the input data dict. The default is False.
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


def is_field(_td, _str, verbose = False):
    '''
    This function checks whether fields are in a dict.
    
    Parameters
    ----------
    _td : dict / list of dict
        dict of trial data.
    _str : string / list of strings
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
    if type(_str) is str:
        _str = [_str]
        
    if type(_str) is not list:
        raise Exception('ERROR: _str must be a list of strings!')
    
    for idx,iDic in enumerate(_td):
        for iStr in _str:
            if type(iStr) is not list and iStr not in iDic.keys():
                return_val = False
                if verbose:
                    print('Field {} not in dict #{}'.format(iStr, idx))
            elif type(iStr) is list:
                for iSubStr in iStr:
                    if iSubStr not in iDic.keys():
                        return_val = False
                        if verbose:
                            print('Field {} not in dict #{}'.format(iSubStr, idx))
    
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


def td_plot(_td, _signals, **kwargs):
    '''
    This function plots signals from the td dict.
    
    This function plots signals from the td dictionary
    Input:
        _td: dict of trial data
        _signals: list of str or str with the signals to plot
        subplot: tuple with the size of the given signals
        gait_events: list of str or str with the gait_events to plot
        title: str with title of the plot
        ylim: tuple with y min and max
        xlim: tuple with x min and max
        save: bool for saving the figure
    
    Example:
        td_plot(td,['LFP_BIP7','LFP_BIP9'], gait_events = ['RFS','LFS'], subplot = (2,1))
    '''
    
    # Input variables
    subplot = ()
    gait_events = []
    title = ''
    save_figure = False
    ylim = ()
    xlim = ()
    ylabel = ''
    xlabel = ''
    grid_plot = True
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'gait_events':
            gait_events = value
        elif key == 'subplot':
            subplot = value
        elif key == 'title':
            title = value
        elif key == 'grid_plot':
            grid_plot = value
        elif key == 'ylabel':
            ylabel = value
        elif key == 'xlabel':
            xlabel = value
        elif key == 'ylim':
            ylim = value
        elif key == 'xlim':
            xlim = value
        elif key == 'save':
            save_figure = value

    # Check input variables
    if type(_td) is not dict:
        raise Exception('ERROR: td type must be a dict! It is a {}'.format(type(_td)))
        
    if type(_signals) is str:
        _signals = [_signals]
        
    # Check whether _signals elements are in _td
    if not(is_field(_td, _signals)):
        raise Exception('ERROR: _signals must be in _td!')

    # Check whether gait_field elements are in _td
    if len(gait_events) != 0:
        if not(is_field(_td, gait_events)):
            raise Exception('ERROR: {} must be in _td!'.format(gait_events))
        
    # Check subplot dimension
    if not(subplot):
        subplot = (len(_signals),1)
        
    # Plot
    fig, axs = plt.subplots(nrows = subplot[0], ncols = subplot[1], sharex=True)
    if title != '':
        fig.suptitle('File {}'.format(title), fontsize=10)
    
    if type(axs).__module__ != np.__name__:
        axs  = [axs]
    
    for signal, ax in zip(_signals,axs):
        
        # Set plot title
        # print(signal)
        ax.set_title(signal)
        
        # Set plot labels
        if xlabel != '':
            ax.set_xlabel('')
        if ylabel != '':
            ax.set_ylabel('')
        
        # Set axes limits
        if ylim:
            ax.set_ylim(ylim)
        else:
            ylim_tmp = tuple([min(_td[signal]), max(_td[signal])])
            ax.set_ylim(ylim_tmp)
        if xlim:
            ax.set_xlim(xlim)
        
        # Set plot grid
        ax.grid(grid_plot)
        
        # Plot signal
        ax.plot(_td[signal])
        
        # Plot gait events
        if len(gait_events) != 0:
            for event in gait_events:
                if 'R' in event:
                    line_style = event_linestyle.R.value
                else:
                    line_style = event_linestyle.L.value
                    
                if 'FS' in event:
                    col = event_color.FS.value
                else:
                    col = event_color.FO.value
                    
                for ev in _td[event]:
                    ax.axvline(ev,ylim[0], ylim[1], color = col, linestyle = line_style)


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
    segment_data_dict = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'combine_fields':
            combine_fields_dict = value
        elif key == 'remove_artefacts':
            remove_artefacts_dict = value
        elif key == 'segment_data':
            segment_data_dict = value
    
    # check dict input variable
    input_dict = False
    if type(td) is dict:
        input_dict = True
        td = [td]
        
    if type(td) is not list:
        raise Exception('ERROR: _td must be a list of dictionaries!')
    
    if combine_fields_dict != None:
        options = remove_fields(combine_fields_dict, ['fields'], exact_field = False, inplace = False)
        combine_fields(td, combine_fields_dict['fields'], **options)
    
    if remove_artefacts_dict != None:
        options = remove_fields(remove_artefacts_dict, ['fields','Fs'], exact_field = False, inplace = False)
        td_segment = identify_artefacts(td, remove_artefacts_dict['fields'], remove_artefacts_dict['Fs'], **options)
    
    if combine_fields_dict != None:
        pass
    
    if input_dict:
        td = td[0]
        
    return td


def preprocess_pipeline(td, **kwargs):
    pass
    # # Input variables
    # signals_to_use = None
    # filters_to_use = None
    
    # # Check input variables
    # for key,value in kwargs.items():
    #     if key == 'signals':
    #         signals_to_use = value
    #     elif key == 'filters':
    #         filters_to_use = value
    
    # # Get selected signals
    # td = get_field(td,signals_to_use)
    
    # for td_tmp in td:
        
    #     # Artefacts removal
    #     artefacts_removal(data, threshold = 300)
    
    #     # Remove unwanted parts
    
    
    #     # Data processing
    
    
    # return td




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
    td_new = add_field(td_test,{'test3': [1,3,4]}, inplace = False)
    if 'test3' in td_new.keys():
        print('Test add_field passed!')
    else:
        raise Exception('ERROR: Test add_field NOT passed!')
    
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
        
        
        
    print('All implemented tests passed!')
    
    