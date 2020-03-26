#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:58:34 2020

@author: raschell
"""

import numpy as np
from utils import transpose, find_values, convert_list_to_array

def get_features(data, events, template):
    events_features = np.empty( (len(events), data.shape[1]*len(template)) )
    events_features[:] = np.nan
    
    for iEv, event in enumerate(events):
        if event + np.min(template) > 0 and event + np.max(template) < data.shape[0]:
            events_features[iEv,:] = data[(event + template).astype('int'),:].reshape(1,data.shape[1]*len(template))
    
    events_features_zeros = events_features.copy()
    events_features_zeros[np.isnan(events_features).any(axis=1),:] = 0
    
    return events_features[~np.isnan(events_features).any(axis=1),:], events_features_zeros


def extract_features(data, event_data, dead_win, no_event_n, features_n, template):
    '''
    This function extracts the features from a dataset.

    Parameters
    ----------
    data : np.array / list of np.array
        Dataset of the signals from which we need to extract the features.
        Every element in the list can be either 1d or 2d.
    event_data : np.array
        Array (nxk) where k are the different events signals and n = data.shape[0].
    dead_win : int
        Length of the dead window around the event. dead_win is in samples. 
    no_event_n : int
        Length of the no_events in the current dataset. neg_train is in samples.
    features_n : int
        Number of features to be retreived for each event.
    template : np.ndarray
        Array containing the distance in samples between time events.

    Returns
    -------
    features : np.array
        Array of features.
    labels : np.array
        Array of labels. Remember that 0 is no_event class.
        
    '''
    
    # Check data input and transpose it to column vector        
    if type(data) is list:
        data = convert_list_to_array(data, axis = 1)
            
    if type(data) is np.ndarray:
        data = transpose(data, 'column')
    else:
        raise Exception('ERROR: data input is not a list nor np.ndarray!')
    
    # Transpose event_data to column vector
    event_data = transpose(event_data, 'column')
    
    # Check whether input data are correct
    if data.shape[0] != event_data.shape[0]:
        raise Exception('ERROR: Data and event data have different length!')
    
    data_n = data.shape[1]
    template_n = len(template)
    if features_n != template_n * data_n:
        raise Exception('# of Features ({}) != from template_n ({}) * data_n ({})! Please check...'.format(features_n, template_n, data_n))
    
    # Get number of data/events
    event_data_n = event_data.shape[1]
    
    # Dead window around the events 
    bad_idx = []
    for iEv in range(event_data_n):
        events = find_values(event_data[:,iEv], 1)
        for event in events:
            bad_idx.append(np.arange(event-dead_win,event+dead_win+1).reshape(2*dead_win+1,))
    bad_idx = np.concatenate(bad_idx)
    
    # Collect no event class indexes
    no_events_idx = np.setdiff1d(np.arange(event_data.shape[0]), bad_idx.astype('int'))
    if no_events_idx.shape[0] > no_event_n:
        rdm_idx = np.random.permutation(no_events_idx.shape[0])
        no_events_idx = no_events_idx[ rdm_idx[:no_event_n]].astype('int')
    else:
        no_event_n = no_events_idx.shape[0]
    
    # Collect features
    features = np.array([]).reshape(0,features_n)
    labels = np.array([]).reshape(0,1)
    for iEv in range(event_data_n):
        features_tmp, _ = get_features(data, find_values(event_data[:,iEv], 1), template)
        
        features = np.vstack([features, features_tmp])
        labels = np.vstack([ labels, (iEv+1)*np.ones((features_tmp.shape[0],1)) ])
    
    if no_events_idx.size != 0:
        features_tmp, _ = get_features(data, no_events_idx, template)
        
        features = np.vstack([features, features_tmp])
        labels = np.vstack([ labels, np.zeros((features_tmp.shape[0],1)) ])
            
    return features, np.squeeze(labels)

# EOF