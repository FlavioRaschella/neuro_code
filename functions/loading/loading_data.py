#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:32:54 2020

@author: raschell
"""

# Import data structures
import numpy as np
# import pandas as pd
import h5py

# Import path library
import os, glob
# Import BlackRock library
from brpylib import NsxFile, NevFile

# Import td utils
from td_utils import remove_all_fields_but

# Library to open mat files
import scipy.io
# Enumerator library
from enum import Enum

class folmat_type(Enum):
    nev = '.nev'
    ns3 = '.ns3'
    ns6 = '.ns6'
    mat = '.mat'

def load_data_from_folder(folder, file_num, file_format, **kwargs):
    '''
    This function loads the data in the specified folder.

    Parameters
    ----------
    folder : str / list of str
        Path(s) where to find the data.
    file_num : list of int / list of list of int
        Number(s) of the dataset to load.
    file_format : str
        Format of the data to collect.
    fields_to_load : str / list of str, optional
        Name of the particular fields to load, trushing the others.
    pre_ext : str, optional
        Name that distinguish the file (between file_num and file_format)
    verbose : bool, optional
        Print what is happening on the function.

    Returns
    -------
    td : dict / list of dict
        Trial(s) data.

    Example
    -------
        folder = ['/Volumes/STN_CHINA/HH10/HH10_20190228', '/Volumes/STN_CHINA/HH10/HH10_20190228']
        file_num = [[],[1,2]]
        file_format = '.ns6'
        td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format, pre_ext = 'trial')

    '''
    # Input variables
    pre_ext = ''
    verbose = False
    
    # Function variables
    file_name = []
    folder_name = []
    td = []
    fields_to_load = None
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'fields':
            fields_to_load = value
        elif key == 'verbose':
            verbose = value
        elif key == 'pre_ext':
            pre_ext = value

    # Check input data
    if type(folder) is str:
        folder = [folder]
    
    if type(file_num) is list:
        if type(file_num[0]) is not(list):
            file_num = [file_num]
    else:
        raise Exception('file_num must be a list.')
    
    if len(file_num) != len(folder):
        raise Exception('Folder and File_num variables should have the same length!')
    
    # Check fields_to_load
    if fields_to_load != None:
        if type(fields_to_load) is str:
            fields_to_load = [fields_to_load]
            print('fields_to_load must be a list of string. You inputed a string. Converting to list...')
            
        if type(fields_to_load) is not list:
            raise Exception('ERROR: fields_to_load must be a list of strings!')
        
    # Check that format type is among the possible ones
    format_exist = False
    for fm_type in folmat_type:
        if fm_type.value == file_format:
            format_exist = True
            break
    
    if not(format_exist):
        raise Exception('Assigned format different from the implemented ones. \n Check the "format_type" enumerator.')
        
    # Get file(s) name
    for idx, fld in enumerate(folder):
        # print(idx, fld)
        if len(file_num[idx]) == 0:
            file_name.append([f for f in os.listdir(fld) if f.endswith(file_format)])
            folder_name.append([fld for f in os.listdir(fld) if f.endswith(file_format)])
        else:
            lst_tmp = []
            for fl_n in file_num[idx]:
                try:
                    file_tmp = glob.glob(os.path.join(fld,'*' + str(fl_n) + pre_ext + file_format))[0]
                    lst_tmp.append(file_tmp.split(os.sep)[-1])
                except:
                    raise Exception('File {} does not exist in folder {}'.format(str(fl_n) + file_format,fld))
            file_name.append(lst_tmp)
            folder_name.append([fld for cnt in file_name[idx] ])
    
    # Flatten lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    folder_name = flatten(folder_name)
    file_name = flatten(file_name)
    
    # Check that folder_name and file_name have the same length
    if len(folder_name) != len(file_name):
        raise Exception('Folder and File variables have different length!')
    
    # Print back information
    if verbose:
        print('Files to load...')
        for fld, fil in zip(folder_name, file_name):
            print(os.path.join(fld,fil))
    
    # Insert into dict lambda funtion
    insert = lambda _dict, obj, pos: {k: v for k, v in (list(_dict.items())[:pos] + list(obj.items()) + list(_dict.items())[pos:])}
    
    # Extract data from file
    for fld, fil in zip(folder_name, file_name):
        td_dict_tmp = load_data_from_file(fld,fil,file_format)
        td_dict_tmp = insert(td_dict_tmp,{'Folder':fld ,'File':fil},0)
        td.append(td_dict_tmp)
    
    if fields_to_load != None:
        remove_all_fields_but(td,fields_to_load,False,True)
    
    print('DATA LOADED!')
    return td
    # End of load_data_from_folder
    
"""
This function loads a file based on its format
""" 
def load_data_from_file(_folder,_file,_file_format):
    
    print('Opening file {}...'.format(os.path.join(_folder,_file)))
    if (_file_format == folmat_type.ns3.value or _file_format == folmat_type.ns6.value):
        # Open file
        try:
            nsx_file = NsxFile(os.path.join(_folder,_file))
        except Exception:
            raise Exception('File {} does not exist!'.format(os.path.join(_folder,_file)))
            
        
        # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
        data = nsx_file.getdata()
        # data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)
        
        # Close the nsx file now that all data is out
        nsx_file.close()
        
    elif _file_format == folmat_type.nev.value:
        # Open file
        try:
            nev_file = NevFile(os.path.join(_folder,_file))
        except Exception:
            raise Exception('File {} does not exist!'.format(os.path.join(_folder,_file)))
        
        # Extract data and separate out spike data
        # Note, can be simplified: spikes = nev_file.getdata(chans)['spike_events'], shown this way for general getdata() call
        data = nev_file.getdata()
        # spikes   = data['spike_events']
        
        # Close the nev file now that all data is out
        nev_file.close()
        
    elif _file_format == folmat_type.mat.value:
        data = {}
        # Try loading matlab files        
        f = load_mat_file(os.path.join(_folder,_file))
        
        # Clean data if it is a dict
        if type(f) is dict:
            data = from_matstruct_to_pydict(f)
        else:
            raise Exception('{} MAT file is not saved as a dict!'.format(os.path.join(_folder,_file)))
        
    return data
    # End of load_data_from_file

def load_mat_file(_file):
    for fun in h5py.File, scipy.io.loadmat:
        try:
            return fun(_file)
        except:
            pass
    raise Exception('File {} does not exist!'.format(_file))

"""
This function remove list enclosure
e.g. [[[]]] --> []
""" 
def reduce_list(_var):
    
    while type(_var) is list and len(_var) == 1:
        _var = _var[0]
    
    return _var

"""
This function transpose an column array to a raw array
e.g. (n,1) --> (1,n)
""" 
def invert_to_column(_var):
    if type(_var) and _var.shape != (1,):
        if _var.shape[0] > _var.shape[1]:
            _var = _var.T
        
    return _var

"""
This function recognise whether the input variable is a matlab struct
""" 
def is_mat_struc(_var):
    is_struct = False
    if _var.dtype.names is not None:
        is_struct = True
        
    return is_struct

"""
This function reduce a matlab struct to its core, in order to extract its fields
""" 
def reduce_mat_object(_var):
    
    while _var.dtype is np.dtype('O'):
        _var = _var[0]
        
    return _var
    
"""
This function tranform a matlab struct to a python dictionary
""" 
def from_matstruct_to_pydict(_td, **kwargs):
    """
    Get a matlab struct and convert it to a tidy dict
    _td: dict
    fields: list of fields
    """
    td_out = {}
    # fields_name = []
    
    # # Check input variables
    # for key,value in kwargs.items():
    #     if key == 'fields':
    #         fields_name = value
    
    # if fields_name == None:
    #     fields_name = []
    
    # if type(fields_name) is str:
    #     fields_name = [fields_name]
    #     print('fields_name must be a list of string. You inputed a string. Converting to list...')
    
    # # Check correct input variables
    # if type(fields_name) is not list:
    #     raise Exception('Field variable must be a list type!')
    
    # Stop field loop flag
    is_not_all_list = True
    
    # Populate td_out dict
    # if len(fields_name) == 0:                              
    # Loop over fields.                    
    while is_not_all_list:
        # Loop internal check over single field
        is_list = True
        # Extract only the requested fields
        for key, val in _td.items():
            if '__' not in key:
                # print(key, val); print(' ')
                #  Check type of variable
                if type(val) == str and key not in td_out.keys():
                    td_out[key] = val
                elif type(val) == int and key not in td_out.keys():
                    td_out[key] = val
                elif type(val) == float and key not in td_out.keys():
                    td_out[key] = val
                elif type(val) == list and key not in td_out.keys():
                    td_out[key] = val
                elif type(val) == np.ndarray:
                    val_red = reduce_mat_object(val)
                    
                    if is_mat_struc(val_red):
                        for key_struct, val_struct in zip(val_red.dtype.names, val_red.item()):
                            val_struct_red = reduce_mat_object(val_struct)
                            if is_mat_struc(val_struct_red):
                                # print(key_struct)
                                is_list = False
                                td_out[key + '_' + key_struct] = val_struct_red
                            else:
                                td_out[key + '_' + key_struct] = reduce_list(np.ndarray.tolist(invert_to_column(val_struct_red)))
                    else:
                        td_out[key] = reduce_list(np.ndarray.tolist(invert_to_column(val_red)))
        if is_list:
            is_not_all_list = False
        else:
            # fields_name = td_out.keys()
            _td = td_out.copy()
            td_out = {}
    # else:
    #     while is_not_all_list:
    #         # Loop internal check over single field
    #         is_list = True
    #         # Extract only the requested fields
    #         for key in fields_name:
    #             # Check that variable exists
    #             try:
    #                 val = _td[key]
    #             except:
    #                 raise Exception('Field {} does not exist!'.format(key))
    #             #  Check type of variable
    #             if type(val) == str and key not in td_out.keys():
    #                 td_out[key] = val
    #             elif type(val) == int and key not in td_out.keys():
    #                 td_out[key] = val
    #             elif type(val) == float and key not in td_out.keys():
    #                 td_out[key] = val
    #             elif type(val) == list and key not in td_out.keys():
    #                 td_out[key] = val
    #             elif type(val) == np.ndarray:
    #                 val_red = reduce_mat_object(val)
                    
    #                 if is_mat_struc(val_red):
    #                     for key_struct, val_struct in zip(val_red.dtype.names, val_red.item()):
    #                         val_struct_red = reduce_mat_object(val_struct)
    #                         if is_mat_struc(val_struct_red):
    #                             # print(key_struct)
    #                             is_list = False
    #                             td_out[key + '_' + key_struct] = val_struct_red
    #                         else:
    #                             td_out[key + '_' + key_struct] = reduce_list(np.ndarray.tolist(invert_to_column(val_struct_red)))
                                
    #                 else:
    #                     td_out[key] = reduce_list(np.ndarray.tolist(invert_to_column(val_red)))
    #         if is_list:
    #             is_not_all_list = False
    #         else:
    #             fields_name = td_out.keys()
    #             _td = td_out.copy()
    #             td_out = {}

    return td_out
    # End of from_matstruct_to_pydict