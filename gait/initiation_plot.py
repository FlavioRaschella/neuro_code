#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:26:20 2020

@author: raschell
"""

import numpy as np
from os import path
import pickle
import matplotlib.pyplot as plt
from stats import confidence_interval

folder = '/Volumes/MK_EPIOS/PD/Initiation/Data/Patient6_May2019/PostSx_Day2'
file_num_short = [3,4,5]
file_num_big   = [6,7]

save_name_short = path.join(folder,'td_' + '_'.join(list(map(str, file_num_short))))
save_name_big   = path.join(folder,'td_' + '_'.join(list(map(str, file_num_big))))

#%% Load data
pickle_in = open(save_name_short + '.pickle',"rb")
_, _, _, td_lt_s, td_rt_s = pickle.load(pickle_in)

pickle_in = open(save_name_big + '.pickle',"rb")
_, _, _, td_lt_b, td_rt_b  = pickle.load(pickle_in)

#%% Normalise data to same lenght
from processing import interpolate1D

# Collect average init length
data_info = [('KIN','KIN_interval'),
             ('EMG','EMG_interval'),
             ('LFP','LFP_interval'),
             ('KIN_a','KIN_interval_a'),
             ('EMG_a','EMG_interval_a'),
             ('LFP_a','LFP_interval_a')]

td_interval_rt = dict()
td_interval_lt = dict()
for info in data_info:
    td_interval_rt[info[0]] = []
    td_interval_lt[info[0]] = []

# Set intervals
for td_rt_s_tmp, td_lt_s_tmp in zip(td_rt_s,td_lt_s):
    for info in data_info:
        td_interval_rt[info[0]].extend(np.array([len(interval) for interval in td_rt_s_tmp[info[1]]]))
        td_interval_lt[info[0]].extend(np.array([len(interval) for interval in td_lt_s_tmp[info[1]]]))
for td_rt_b_tmp, td_lt_b_tmp in zip(td_rt_b,td_lt_b):
    for info in data_info:
        td_interval_rt[info[0]].extend(np.array([len(interval) for interval in td_rt_b_tmp[info[1]]]))
        td_interval_lt[info[0]].extend(np.array([len(interval) for interval in td_lt_b_tmp[info[1]]]))

for info in data_info:
    td_interval_rt[info[0]] = np.array(td_interval_rt[info[0]]).mean().round().astype('int')
    td_interval_lt[info[0]] = np.array(td_interval_lt[info[0]]).mean().round().astype('int')

# Normalise dataset
data_info = [('KIN_name', 'KIN'),
             ('EMG_name', 'EMG'),
             ('LFP_lbp_name', 'LFP'),
             ('LFP_hbp_name', 'LFP'),
             ('KIN_name_a','KIN_a'),
             ('EMG_name_a','EMG_a'),
             ('LFP_lbp_name_a','LFP_a'),
             ('LFP_hbp_name_a','LFP_a')]

for td_rt_s_tmp, td_lt_s_tmp in zip(td_rt_s,td_lt_s):
    # Right short step
    for info in data_info:
        signal_name = []
        for signal in td_rt_s_tmp[info[0]]:
            signal_name.append(signal + '_nor')
            signal_new= []
            for sig in td_rt_s_tmp[signal]:
                signal_new.append(interpolate1D(sig, td_interval_rt[info[1]]))
            td_rt_s_tmp[signal + '_nor'] = signal_new
        td_rt_s_tmp[info[0] + '_nor'] = signal_name
    
    # Left short step
    for info in data_info:
        signal_name = []
        for signal in td_lt_s_tmp[info[0]]:
            signal_name.append(signal + '_nor')
            signal_new= []
            for sig in td_lt_s_tmp[signal]:
                signal_new.append(interpolate1D(sig, td_interval_lt[info[1]]))
            td_lt_s_tmp[signal + '_nor'] = signal_new
        td_lt_s_tmp[info[0] + '_nor'] = signal_name

for td_rt_b_tmp, td_lt_b_tmp in zip(td_rt_b,td_lt_b):
    # Right big step
    for info in data_info:
        signal_name = []
        for signal in td_rt_b_tmp[info[0]]:
            signal_name.append(signal + '_nor')
            signal_new= []
            for sig in td_rt_b_tmp[signal]:
                signal_new.append(interpolate1D(sig, td_interval_rt[info[1]]))
            td_rt_b_tmp[signal + '_nor'] = signal_new
        td_rt_b_tmp[info[0] + '_nor'] = signal_name
    
    # Left big step
    for info in data_info:
        signal_name = []
        for signal in td_lt_b_tmp[info[0]]:
            signal_name.append(signal + '_nor')
            signal_new= []
            for sig in td_lt_b_tmp[signal]:
                signal_new.append(interpolate1D(sig, td_interval_lt[info[1]]))
            td_lt_b_tmp[signal + '_nor'] = signal_new
        td_lt_b_tmp[info[0] + '_nor'] = signal_name

#%% Plot data

def col_scale(n,shade = 'gray'):
    if shade == 'gray':
        col = np.array([np.linspace(0,0.8,n), np.linspace(0,0.8,n), np.linspace(0,0.8,n)]).T
    elif shade == 'r':
        col = np.array([np.linspace(0,0.8,n), np.linspace(0,0,n), np.linspace(0,0,n)]).T
    elif shade == 'g':
        col = np.array([np.linspace(0,0,n), np.linspace(0,0.8,n), np.linspace(0,0,n)]).T
    elif shade == 'b':
        col = np.array([np.linspace(0,0,n), np.linspace(0,0,n), np.linspace(0,0.8,n)]).T
    return col

def join_lists(list1,list2):
    if len(list1) != len(list2):
        raise Exception('ERROR: lists have different length.')
    
    lists = []
    for list1_el, list2_el in zip(list1,list2):
        lists.append(np.concatenate((list1_el,list2_el), axis = 0))
    return lists


lfp_R_lb_name = td_lt_s[0]['LFP_lbp_name_nor'][:2]
lfp_R_lb_a_name = td_lt_s[0]['LFP_lbp_name_a_nor'][:2]
lfp_R_hb_name = td_lt_s[0]['LFP_hbp_name_nor'][:2]
lfp_R_hb_a_name = td_lt_s[0]['LFP_hbp_name_a_nor'][:2]

lfp_L_lb_name = td_lt_s[0]['LFP_lbp_name_nor'][2:]
lfp_L_lb_a_name = td_lt_s[0]['LFP_lbp_name_a_nor'][2:]
lfp_L_hb_name = td_lt_s[0]['LFP_hbp_name_nor'][2:]
lfp_L_hb_a_name = td_lt_s[0]['LFP_hbp_name_a_nor'][2:]

# Left side all trials
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_R_lb_name,lfp_R_hb_name,lfp_R_lb_a_name,lfp_R_hb_a_name):
    # break
    data_l_s = dict()
    data_l_b = dict()
    
    data_plot_l = [(lfp_h_name[:17] + '_high',lfp_h_a_name,lfp_h_name,'LFP_a'),
                   (lfp_l_name[:17] + '_low',lfp_l_a_name,lfp_l_name,'LFP_a'),
                   ('EMG_vl','EMG_LVL_a_nor','EMG_LVL_nor','EMG_a'),
                   ('KIN_knee','KIN_L_angle_knee_a_nor','KIN_L_angle_knee_nor','KIN_a'),
                   ('KIN_foot','KIN_LeftFoot_P_y_a_nor','KIN_LeftFoot_P_y_nor','KIN_a')]
    
    for data_l in data_plot_l:
        data_l_s[data_l[0]] = []
        data_l_b[data_l[0]] = []
        
    for data_l in data_plot_l:
        for td_lt_s_tmp in td_lt_s:
            data_l_s[data_l[0]].extend( join_lists(td_lt_s_tmp[data_l[1]],td_lt_s_tmp[data_l[2]]) )
        for td_lt_b_tmp in td_lt_b:
            data_l_b[data_l[0]].extend( join_lists(td_lt_b_tmp[data_l[1]],td_lt_b_tmp[data_l[2]]) )
            
    print('# samples left short: {}\n# samples left big: {}'.format(len(data_l_s[data_plot_l[0][0]]),len(data_l_b[data_plot_l[0][0]])))
    
    # Plot
    fig, ax = plt.subplots(len(data_plot_l),1)
    plt.suptitle('RIGHT INIT')
    
    for iCount, data_plot in enumerate(data_plot_l):
        m_s, dw_s, up_s = confidence_interval(np.array(data_l_s[data_plot[0]]).T)
        ax[iCount].fill_between(np.arange(m_s.shape[0]),dw_s, up_s, alpha=0.1,color='k')
        ax[iCount].plot(m_s, color='k', linewidth=2)
        m_b, dw_b, up_b = confidence_interval(np.array(data_l_b[data_plot[0]]).T)
        ax[iCount].fill_between(np.arange(m_b.shape[0]),dw_b, up_b, alpha=0.1,color='b')
        ax[iCount].plot(m_b, color='b', linewidth=2)
        ax[iCount].set_title(data_plot[0] + '. k : short, b : big')
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        ax[iCount].vlines(td_interval_lt[data_plot[3]],min([dw_s.min(),dw_b.min()]),max([up_s.max(),up_b.max()]),'k')

    plt.tight_layout()
    # Save figure
    fig_name = path.join(folder,lfp_h_name[:17]+'short_big_steps_together_left')
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')    
    
    
# Right side all trials
for lfp_l_name, lfp_h_name, lfp_l_a_name, lfp_h_a_name in zip(lfp_L_lb_name,lfp_L_hb_name,lfp_L_lb_a_name,lfp_L_hb_a_name):
    # break
    data_r_s = dict()
    data_r_b = dict()
    
    data_plot_r = [(lfp_h_name[:19] + '_high',lfp_h_a_name,lfp_h_name,'LFP_a'),
                   (lfp_l_name[:19] + '_low',lfp_l_a_name,lfp_l_name,'LFP_a'),
                   ('EMG_vl','EMG_RVL_a_nor','EMG_RVL_nor','EMG_a'),
                   ('KIN_knee','KIN_R_angle_knee_a_nor','KIN_R_angle_knee_nor','KIN_a'),
                   ('KIN_foot','KIN_RightFoot_P_y_a_nor','KIN_RightFoot_P_y_nor','KIN_a')]
    
    for data_r in data_plot_r:
        data_r_s[data_r[0]] = []
        data_r_b[data_r[0]] = []
    
    for data_r in data_plot_r:
        for td_rt_s_tmp in td_rt_s:
            data_r_s[data_r[0]].extend( join_lists(td_rt_s_tmp[data_r[1]],td_rt_s_tmp[data_r[2]]) )
        for td_rt_b_tmp in td_rt_b:
            data_r_b[data_r[0]].extend( join_lists(td_rt_b_tmp[data_r[1]],td_rt_b_tmp[data_r[2]]) )
    
    print('# samples right short: {}\n# samples right big: {}'.format(len(data_r_s[data_plot_r[0][0]]),len(data_r_b[data_plot_r[0][0]])))
    
    # Plot
    fig, ax = plt.subplots(len(data_plot_r),1)
    plt.suptitle('RIGHT INIT')
    
    for iCount, data_plot in enumerate(data_plot_r):
        m_s, dw_s, up_s = confidence_interval(np.array(data_r_s[data_plot[0]]).T)
        ax[iCount].fill_between(np.arange(m_s.shape[0]),dw_s, up_s, alpha=0.1,color='k')
        ax[iCount].plot(m_s, color='k', linewidth=2)
        m_b, dw_b, up_b = confidence_interval(np.array(data_r_b[data_plot[0]]).T)
        ax[iCount].fill_between(np.arange(m_b.shape[0]),dw_b, up_b, alpha=0.1,color='b')
        ax[iCount].plot(m_b, color='b', linewidth=2)
        ax[iCount].set_title(data_plot[0] + '. k : short, b : big')
        ax[iCount].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        
        ax[iCount].vlines(td_interval_rt[data_plot[3]],min([dw_s.min(),dw_b.min()]),max([up_s.max(),up_b.max()]),'k')

    plt.tight_layout()
    # Save figure
    fig_name = path.join(folder,lfp_h_name[:19]+'short_big_steps_together_right')
    pickle.dump(fig, open(fig_name +'.pickle', 'wb'))
    fig.savefig(fig_name + '.pdf', bbox_inches='tight')    
    
# EOF