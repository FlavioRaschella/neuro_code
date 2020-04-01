#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:32:54 2020

@author: raschell
"""

# Import data structures
import numpy as np

# Import file reading libs
import h5py
import csv
from itertools import islice

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
    csv = '.csv'
    txt = '.txt'

def load_data_from_folder(folders, **kwargs):
    '''
    This function loads the data in the specified folders.

    Parameters
    ----------
    folders : str / list of str, len (_folders)
        Path(s) where to find the data.
        
    files_number : int, list of int / list of list of int, optional
        Number(s) of the files to load. If list of list, it refers to the case
        with loading from multiple folders (e.g. [[1,2,3],[4,5,6]]).
        
    files_name : str / list of str / list of list of str, optional
        Names of the files to load. If list of list, it refers to the case
        with loading from multiple folders (e.g. [['file1'],['file1','file2']]).
        
    files_format : str, optional
        Format of the files to collect. Possible values are: '.txt','.csv','.mat'.
        The default is '.mat'.
        
    pre_ext : str, optional
        Part of the name that distinguish the file (between file_num and 
        files_format). Example: in 'file1_gait.mat' --> pre_ext = '_gait'.
        The default is ''.
        
    start_line : int, optional
        Starting line for reading file. This is used for .csv and .txt files.
        The default is 0.
        
    fields : str / list of str, optional
        Fields to read in file. This is used for .csv and .txt files.
        The default is None.
        
    delimiter : str, optional
        The string used to separate values. This is used for .csv and .txt files.
        The default is '\t'.
        
    dtype : dtype, optional
        Data type of the resulting array. If None, the dtypes will be 
        determined by the contents of each column, individually.
        This is used for .txt files. The default is None.
        Example: dtype=[('myint','i8'),('myfloat','f8'),('mystring','S5')]
        
    reduce_lists : bool, optional
        Reduce lists with one element to the element itself. The default is True.
        
    verbose : bool, optional
        Narrate what is happening on the function. The default is False.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        Trial(s) data.

    Example
    -------
        folders = ['/Volumes/PATH1/PATH2', '/Volumes/PATH1/PATH3']
        files_number = [[4],[1,2]]
        files_format = '.mat'
        td = load_data_from_folder(folders = folders,
                                   files_number = files_number,
                                   files_format = files_format,
                                   pre_ext = 'trial')
        
        folders = '/Volumes/PATH1/PATH2'
        files_name = ['FILE1','FILE2']
        files_format = '.txt'
        delimiter = '\t'
        fields = ['time','field1','filed2']
        start_line = 26
        td = load_data_from_folder(folders = folders,
                                   files_name = files_name,
                                   files_format = files_format,
                                   fields = fields,
                                   delimiter = delimiter,
                                   start_line = start_line)
        
        folders = '/Volumes/PATH1/PATH2'
        files_name = 'FILE1'
        files_format = '.csv'
        fields = ['time','field1','filed2']
        td = load_data_from_folder(folders = folders,
                                   files_name = files_name,
                                   files_format = files_format,
                                   fields = fields,
                                   verbose = True)
    '''
    
    # Input variables
    files_number = None
    files_name = None
    files_format = '.mat'
    pre_ext = ''
    verbose = False
    reduce_lists = True
        
    # Input for csv & txt
    start_line_ = 0
    fields_ = None
    # Extensions for txt
    delimiter_ = '\t'
    dtype_ = None
    
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
        elif key == 'start_line':
            start_line_ = value
        elif key == 'delimiter':
            delimiter_ = value
        elif key == 'dtype':
            dtype_ = value
        elif key == 'fields':
            fields_ = value
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
        print(' ')
    
    # Insert into dict lambda funtion
    insert = lambda _dict, obj, pos: {k: v for k, v in (list(_dict.items())[:pos] + list(obj.items()) + list(_dict.items())[pos:])}
    
    # Extract data from file
    for folder, file in zip(folders_list, files_list):
        
        if (files_format == folmat_type.ns3.value or files_format == folmat_type.ns6.value):
            raise Exception('Implemented but never tested... To debug...')
            
        elif files_format == folmat_type.nev.value:
            raise Exception('Implemented but never tested... To debug...')
            
        elif files_format == folmat_type.mat.value:
            td_dict_tmp = load_data_from_file(folder,file,files_format)
            
        elif files_format == folmat_type.csv.value:
            td_dict_tmp = load_data_from_file(folder,file,files_format, 
                                              start_line = start_line_, 
                                              fields = fields_, 
                                              verbose = verbose)
            
        elif files_format == folmat_type.txt.value:
            td_dict_tmp = load_data_from_file(folder,file,files_format,
                                              start_line = start_line_,
                                              fields = fields_,
                                              delimiter = delimiter_,
                                              dtype = dtype_, 
                                              verbose = verbose)
            
        td_dict_tmp = insert(td_dict_tmp,{'Folder':folder ,'File':file},0)
        if reduce_lists:
            td_dict_tmp = reduce_one_element_list(td_dict_tmp)
        td.append(td_dict_tmp)
    
    print('\nDATA LOADED!')
    return td
    

def load_data_from_file(folder,file,file_format, **kwargs):
    '''
    This function loads the data from a file depending on their format.

    Parameters
    ----------
    folders : str
        Path where to find the data.
        
    files : str
        Name of the file to load.
        
    files_format : str
        Format of the files to collect. Possible values are: '.txt','.csv','.mat'.
        
    start_line : int, optional
        Starting line for reading file. This is used for .csv and .txt files.
        The default is 0.
        
    fields : str / list of str, optional
        Fields to read in file. This is used for .csv and .txt files.
        The default is None.
        
    delimiter : str, optional
        The string used to separate values. This is used for .csv and .txt files.
        The default is '\t'.
        
    dtype : dtype, optional
        Data type of the resulting array. If None, the dtypes will be 
        determined by the contents of each column, individually.
        This is used for .txt files. The default is None.
        Example: dtype=[('myint','i8'),('myfloat','f8'),('mystring','S5')]
        
    reduce_lists : bool, optional
        Reduce lists with one element to the element itself. The default is True.
        
    verbose : bool, optional
        Narrate what is happening on the function. The default is False.
    
    Returns
    -------
    data : dict
        Dictionary of the data contained in the file.

    '''
    # Input variables
    start_line_ = 0
    fields_ = None
    delimiter_ = '\t'
    dtype_ = None
    verbose = False
    
    # Check input variables
    for key,value in kwargs.items():
        key = key.lower()
        if key == 'start_line':
            start_line_ = value
        elif key == 'fields':
            fields_ = value
        elif key == 'delimiter':
            delimiter_ = value
        elif key == 'dtype':
            dtype_ = value
        elif key == 'verbose':
            verbose = value
    
    if verbose: print('Opening file {}...'.format(os.path.join(folder,file)))
    if (file_format == folmat_type.ns3.value or file_format == folmat_type.ns6.value):
        # Load nsx
        nsx_file = NsxFile(os.path.join(folder,file))
        
        # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
        data = nsx_file.getdata()
        # data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample)
        
        # Close the nsx file now that all data is out
        nsx_file.close()
        
    elif file_format == folmat_type.nev.value:
        # Load nev
        nev_file = NevFile(os.path.join(folder,file))
        
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
        
    elif file_format == folmat_type.csv.value:
        # Load csv files        
        data = load_csv_file(filename = os.path.join(folder,file),
                             fields = fields_,
                             start_line = start_line_,
                             verbose = verbose)
        if type(data) is not dict:
            raise Exception('ERROR: {} CSV file is not saved as a dict!'.format(os.path.join(folder,file)))
        
    elif file_format == folmat_type.txt.value:
        # Load txt files        
        data = load_txt_file(filename = os.path.join(folder,file),
                             fields = fields_,
                             delimiter = delimiter_, 
                             dtype = dtype_,
                             start_line = start_line_)
        if type(data) is not dict:
            raise Exception('ERROR: {} CSV file is not saved as a dict!'.format(os.path.join(folder,file)))
        
    return data

# =============================================================================
# Loading functions for each format
# =============================================================================

# CSV files
def load_csv_file(filename, fields = None, start_line = 0, verbose = False):
    
    table = []
    with open(filename, 'r') as f:
        reader = csv.reader( [line.replace('\0','') for line in f] )
        try:
            for row in islice(reader, start_line, None):
               table.append(row)
        except csv.Error as e:
            raise Exception('ERROR: file {}, line {}: {}'.format(filename, reader.line_num, e))
    
    if fields != None:
        output = dict()
        for field in fields:
            output[field] = []
        table_n = len(table)
        for iR, row in enumerate(table):
            if verbose and iR%10000 == 0: print('Creating dict. Row: {}/{}'.format(iR+1,table_n))
            if len(row) == len(fields):
                for row_el, field in zip(row,fields):
                    output[field].append(row_el)
    else: 
        output = {'data': table}
        
    return output

# TXT files
def load_txt_file(filename, fields, delimiter = '\t', dtype = None, start_line = 0):
    # Collect data in txt file
    table = np.genfromtxt(filename, delimiter = delimiter , skip_header = start_line,
                      dtype = dtype, names = fields)
    # Store data in dict
    output = dict()
    for key in table.dtype.names:
        output[key] = table[key]
                
    return output

# MAT files
def load_mat_file(_file):
    # Load mat file using either h5py or scipy.io.loadmat
    for fun in h5py.File, scipy.io.loadmat:
        try:
            return fun(_file)
        except:
            pass
    raise Exception('ERROR: File {} does not exist!'.format(_file))

# =============================================================================
# Utility functions for importing a .mat file
# =============================================================================

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
            # break; key = 'RFS'; val = td[key]; key = 'fileComments'; val = td[key];            
            # print(val.dtype.char)
            
            if val.dtype.char in ['U']: # String
                td_out[key] = reduce_mat_array(val, 'U')
            elif val.dtype.char in ['B','b']: # byte
                td_out[key] = reduce_mat_array(val, 'B')
            elif val.dtype.char in ['H','u','i']: # int
                td_out[key] = reduce_mat_array(val, 'H')
            elif val.dtype.char in ['d','l']: # numpy.ndarray
                td_out[key] = reduce_mat_array(val,'d')
            elif val.dtype.char in ['V']: # struct
                td_out[key] = process_mat_struct(key,val)
            elif val.dtype.char in ['O']: # cell
                td_out[key] = process_mat_cell(key,val)
            else:
                raise Exception('ERROR: "{}" key value has mat type "{}" not implemented! Update the code...'.format(key, val.dtype.char))
    return td_out


def reduce_one_element_list(_td):
    '''
    This function takes the loaded list of dictionaries and converts the list 
    attributes of one element into that element.

    Parameters
    ----------
    td : dict / list of dict, len (n_td)
        Trial(s) data.

    Returns
    -------
    td : dict / list of dict, len (n_td)
        Trial(s) data where list elements ot len == 1 are reduced to the element itself.

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
    
# EOF