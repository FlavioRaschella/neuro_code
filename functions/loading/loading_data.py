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

# Import utils
from utils import flatten_list

# Library to open mat files
import scipy.io
# Enumerator library
from enum import Enum

class folmat_type(Enum):
    nev = '.nev'
    ns3 = '.ns3'
    ns6 = '.ns6'
    mat = '.mat'

def load_data_from_folder(folders, **kwargs):
    '''
    This function loads the data in the specified folders.

    Parameters
    ----------
    folders : str / list of str
        Path(s) where to find the data.
    files_number : list of int / list of list of int
        Number(s) of the dataset to load.
    files_format : str
        Format of the data to collect.
    pre_ext : str, optional
        Name that distinguish the file (between file_num and files_format)
    verbose : bool, optional
        Print what is happening on the function.

    Returns
    -------
    td : dict / list of dict
        Trial(s) data.

    Example
    -------
        folders = ['/Volumes/STN_CHINA/HH10/HH10_20190228', '/Volumes/STN_CHINA/HH10/HH10_20190228']
        file_num = [[],[1,2]]
        files_format = '.ns6'
        td = load_data_from_folder(folders = folders,file_num = file_num,files_format = files_format, pre_ext = 'trial')

    '''
    # Input variables
    files_number = None
    files_name = None
    files_format = '.mat'
    pre_ext = ''
    verbose = False
    reduce_lists = True
    
    # Function variables
    folders_list = []
    files_list = []
    td = []
    
    # Check input variables
    for key,value in kwargs.items():
        if key == 'files_number':
            files_number = value
        elif key == 'files_name':
            files_name = value
        elif key == 'files_format':
            files_format = value
        elif key == 'pre_ext':
            pre_ext = value
        elif key == 'reduce_lists':
            reduce_lists = value
        elif key == 'verbose':
            verbose = value

    # Check input data
    if type(folders) is str:
        folders = [folders]
    
    if files_number == None and files_name == None:
        raise Exception('ERROR: you must assign either files_number or files_name!')
    if files_number != None and files_name != None:
        raise Exception('ERROR: you can assign either files_number or files_name, not both!')
    
    if files_number != None:
        if type(files_number) is int or type(files_number) is float:
            files_number = [files_number]
        if type(files_number) is list:
            if type(files_number[0]) is not list:
                files_number = [files_number]
        else:
            raise Exception('ERROR: files_number must be a list.')
            
        if len(files_number) != len(folders):
            raise Exception('ERROR: Folder and files_number variables must have the same length!')
            
    if files_name != None:
        if type(files_name) is str:
            files_name = [files_name]
        if type(files_name) is list:
            if type(files_name[0]) is not list:
                files_name = [files_name]
        else:
            raise Exception('ERROR: files_name must be a list.')
            
        if len(files_name) != len(folders):
            raise Exception('ERROR: folders and files_name variables must have the same length!')
    
    # Check that format type is among the possible ones
    format_exist = False
    for fm_type in folmat_type:
        if fm_type.value == files_format:
            format_exist = True
            break
    if not format_exist:
        print('WARNING: You did not assign a file format!\nFile format is then: "{}"'.format(files_format))
    
    # Get file id
    if files_number != None:
        for folder, files in zip(folders, files_number):
            files_tmp = []
            for file in files:
                try:
                    file_tmp = glob.glob(os.path.join(folder,'*' + str(file) + pre_ext + files_format))[0]
                    files_tmp.append(file_tmp.split(os.sep)[-1])
                except:
                    raise Exception('ERROR: File {} does not exist in folder {}!'.format(str(file) + files_format,folder))
            files_list.append(files_tmp)
            folders_list.append([folder for cnt in files_tmp ])
    elif files_name != None:
        for folder, files in zip(folders, files_name):
            files_tmp = []
            for file in files:
                try:
                    file_tmp = glob.glob(os.path.join(folder,'*' + file + pre_ext + files_format))[0]
                    files_tmp.append(file_tmp.split(os.sep)[-1])
                except:
                    raise Exception('ERROR: File {} does not exist in folder {}!'.format(str(file) + files_format,folder))
            files_list.append(files_tmp)
            folders_list.append([folder for cnt in files_tmp ])
    
    # Flatten lists
    # flatten = lambda l: [item for sublist in l for item in sublist]
    folders_list = flatten_list(folders_list, False, False)
    files_list = flatten_list(files_list, False, False)
    
    # Check that folder_name and file_name have the same length
    if len(folders_list) != len(files_list):
        raise Exception('ERROR: Folder and File variables have the same length!')
    
    # Print back information
    if verbose:
        print('Files to load...')
        for folder, files in zip(folders_list, files_list):
            print(os.path.join(folder,files))
    
    # Insert into dict lambda funtion
    insert = lambda _dict, obj, pos: {k: v for k, v in (list(_dict.items())[:pos] + list(obj.items()) + list(_dict.items())[pos:])}
    
    # Extract data from file
    for folder, file in zip(folders_list, files_list):
        # break
        td_dict_tmp = load_data_from_file(folder,file,files_format)
        td_dict_tmp = insert(td_dict_tmp,{'Folder':folder ,'File':file},0)
        if reduce_lists:
            td_dict_tmp = reduce_one_element_list(td_dict_tmp)
        td.append(td_dict_tmp)
    
    print('DATA LOADED!')
    return td
    # End of load_data_from_folder
    
"""
This function loads a file based on its format
""" 
def load_data_from_file(folder,file,file_format):
    '''
    This function loads the data from a file.

    Parameters
    ----------
    folders : str
        Path where to find the data.
    files : str
        Name of the file to load
    files_format : str
        Format of the data to collect.
    
    Returns
    -------
    data : dict
        Dictionary of the data contained in the file.

    '''
    print('Opening file {}...'.format(os.path.join(folder,file)))
    if (file_format == folmat_type.ns3.value or file_format == folmat_type.ns6.value):
        # Open file
        try:
            nsx_file = NsxFile(os.path.join(folder,file))
        except Exception:
            raise Exception('ERROR: File {} does not exist!'.format(os.path.join(folder,file)))
            
        
        # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
        data = nsx_file.getdata()
        # data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)
        
        # Close the nsx file now that all data is out
        nsx_file.close()
        
    elif file_format == folmat_type.nev.value:
        # Open file
        try:
            nev_file = NevFile(os.path.join(folder,file))
        except Exception:
            raise Exception('ERROR: File {} does not exist!'.format(os.path.join(folder,file)))
        
        # Extract data and separate out spike data
        # Note, can be simplified: spikes = nev_file.getdata(chans)['spike_events'], shown this way for general getdata() call
        data = nev_file.getdata()
        # spikes   = data['spike_events']
        
        # Close the nev file now that all data is out
        nev_file.close()
        
    elif file_format == folmat_type.mat.value:
        data = {}
        # Try loading matlab files        
        f = load_mat_file(os.path.join(folder,file))
        
        # Clean data if it is a dict
        if type(f) is dict:
            data = process_mat_dict(f)
        else:
            raise Exception('ERROR: {} MAT file is not saved as a dict!'.format(os.path.join(folder,file)))
        
    return data
    # End of load_data_from_file

def load_mat_file(_file):
    for fun in h5py.File, scipy.io.loadmat:
        try:
            return fun(_file)
        except:
            pass
    raise Exception('ERROR: File {} does not exist!'.format(_file))

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

def reduce_mat_array(val,kind = None):
    if val.ndim == 2 and 1 in val.shape:
        if val.shape == (1,1):
            val = val[0]
        else:
            val = np.squeeze(val)
    
    if val.size == 1:
        if kind in ['U']: # String
            val = np.str(val[0])
        elif kind in ['B','b']: # byte
            val = np.float(val)
        elif kind in ['H','u','i']: # int
            val = np.int(val)
    
    return val
    
def process_mat_struct(key,val):
    val = reduce_mat_array(val, kind = 'V')
    
    if type(val) == np.ndarray:
        tmp_dict = dict()
        # print(val.dtype.names)
        for name in val.dtype.names:
            # print(val[name].tolist())
            tmp_dict[name] = val[name].tolist()
        
        for k, v in tmp_dict.items():
            # break
            for idx, el in enumerate(v):
                # break
                tmp = process_mat_dict({'el':el})
                tmp_dict[k][idx] = tmp['el']
    else:
        raise Exception('ERROR: new data type in structure disassembling! Update the code...')

    return tmp_dict

def process_mat_cell(key,val):
    val = reduce_mat_array(val, kind = 'O')
    
    if type(val) == np.ndarray:
        tmp_dict = dict()
        # print(val.dtype.names)
        tmp_dict[key] = val.tolist()
        
        for k, v in tmp_dict.items():
            # break
            for idx, el in enumerate(v):
                # break
                tmp = process_mat_dict({'el':el})
                tmp_dict[k][idx] = tmp['el']
    else:
        raise Exception('ERROR: new data type in structure disassembling! Update the code...')

    return tmp_dict

"""
This function tranform a matlab struct to a python dictionary
""" 
def process_mat_dict(td):
    """
    Get a matlab struct and convert it to a tidy dict
    _td: dict
    
    This is the basic translation that is adopted:
        matlab array -> python numpy array
        matlab cell array -> python list
        matlab structure -> python dict
    """
    td_out = {}
    for key, val in td.items():
        if '__' not in key:
            # break; key = 'gait_time'; val = td[key]; key = 'fileComments'; val = td[key];            
            # print(val.dtype.char)
            
            if val.dtype.char in ['U']: # String
                td_out[key] = reduce_mat_array(val, 'U')
            elif val.dtype.char in ['B','b']: # byte
                td_out[key] = reduce_mat_array(val, 'B')
            elif val.dtype.char in ['H','u','i']: # int
                td_out[key] = reduce_mat_array(val, 'H')
            elif val.dtype.char in ['d']: # numpy.ndarray
                td_out[key] = reduce_mat_array(val,'d')
            elif val.dtype.char in ['V']: # struct
                td_out[key] = process_mat_struct(key,val)
            elif val.dtype.char in ['O']: # cell
                td_out[key] = process_mat_cell(key,val)
            else:
                raise Exception('ERROR: mat type "{}" not implemented! Update the code...'.format(val.dtype.char))
    return td_out


def reduce_one_element_list(_td):
    '''
    This function takes the loaded list of dictionaries and converts the list 
    attributes of one element into that element.

    Parameters
    ----------
    td : dict / list of dict
        Trial(s) data.

    Returns
    -------
    td : dict / list of dict
        Trial(s) data.

    '''
    
    if type(_td) is not dict:
        raise Exception('ERROR: The type(td) must be a dict! It is a {}'.format(type(_td)))
    
    for k,v in _td.items():
        if type(v) is list:
            if len(v) == 1 and type(v[0]) is not dict:
                _td[k] = v[0]
            elif len(v) == 1 and type(v[0]) is dict:
                _td[k] = reduce_one_element_list(v[0])
        elif type(v) is dict:
            _td[k] = reduce_one_element_list(v)
    
    return _td

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
    
    
    
    
    
    
    