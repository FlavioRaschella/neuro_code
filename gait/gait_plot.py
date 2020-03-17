#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:47:21 2020

@author: raschell
"""

#%% Import libraries
# Import data management libraries
import numpy as np
# Import plotting libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Import loading functions
from loading_data import load_data_from_folder
# Import td utilities
from td_utils import is_field
# Flatten list
from utils import flatten_list

def stick_plot(td, kinematics, **kwargs):
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
    coordinates : list of strings
        Coordinates as ordered in the kinematics dictionary.
    step_plot : int/float, optional
        Step between one representation and the next. The default value is 1.
    pause : int/float, optional
        Pause between one representation and the next. It is in seconds. The default value is .1.

    '''
    
    coordinates = ['x','y','z']
    step_plot = 10
    pause = .1
    idx_start = 0
    idx_stop = 0
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
        elif key == 'verbose':
            verbose = value
        else:
            print('WARNING: key "{}" not recognised by the td_plot function...'.format(key))
    
    #%% Check input variables
    body_parts = ['leg_r','leg_l','arm_r','arm_l','head','trunk','other']
    for body_part in kinematics.keys():
        if body_part not in body_parts:
            raise Exception('ERROR: Possible body parts are "leg_r","leg_l","arm_r","arm_l","head","trunk","other". You inserted "{}" !'.format(body_part))

    signals = []
    for k,v in kinematics.items():
        signals += v
    signals = flatten_list(signals)
    
    if not is_field(td, signals, True):
        raise Exception('ERROR: Some fields are missing from the trial data!!')
    
    #%% Collect points in the space
    
    # Get data len
    signals_len = [len(td[signal]) for signal in signals]
    if (np.diff(signals_len) > 0.1).any():
        raise Exception('ERROR: signals have different length! Not possible...')
    else:
        signals_len = signals_len[0]
    
    # Chack idx_start and idx_stop
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
    
    #%% Correct the data using stickplot
                
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

#%% Main function
if __name__ == '__main__':
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
    stick_plot(td, kin_info, coordinates = ['x','z','y'], step_plot = 10, pause = .1, verbose = True)


# EOF