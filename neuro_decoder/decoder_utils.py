#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:58:34 2020

@author: raschell
"""

# Import numpy
import numpy as np
# Import utility libs
from utils import transpose, find_values, convert_list_to_array, flatten_list
from td_utils import is_field, combine_dicts
import copy
# Import plotting lib
import matplotlib.pyplot as plt
from coolors import parula
# Import decoders lib
from rLDA import rLDA
# Import stats lib
from stats import boostrap, confidence_interval_sample

# =============================================================================
# Grid search decoder
# =============================================================================
def grid_search_cv_adapt_shift(estimator, data, events, shifts_for_events, cv_blocks, win_tol, params_clf, params_data):
    
    # Check decoder parameters
    if not check_decoder_params(estimator, params_clf, convert = True):
        raise Exception('ERROR in check_decoder_params!')
    
    # ========================================================================
    # Shift the events
    if events[0].ndim != len(shifts_for_events):
        raise Exception('ERROR: events dimension {} is different from the number of shifts_for_events {}'.format(events[0].ndim, len(shifts_for_events)))

    events_shifted = copy.deepcopy(events)
    
    # Adjust dimension to allow for loop later
    if events_shifted[0].ndim == 1:
        for event in events_shifted:
            event = np.expand_dims(event, axis = 1)
            
    # Remove event happening before the shift
    for iEv, event in enumerate(events_shifted):
        for ev in range(event.shape[1]):
            ev_idx = np.where(np.logical_or(find_values(event[:,ev],1,'equal')+shifts_for_events[ev]<0, find_values(event[:,ev],1,'equal')+shifts_for_events[ev]> event[:,ev].shape[0]))[0]
            if len(ev_idx) != 0:
                print('event removed in block {}, target {}'.format(iEv,ev))
                event[find_values(event[:,ev],1,'equal')[ev_idx],ev] = 0
    
    # Shift events
    for event in events_shifted:
        for ev in range(event.shape[1]):
            event[:,ev] = np.roll(event[:,ev],shifts_for_events[ev])
    
    # ========================================================================
    # Build the decoders
    raise Exception('ERROR: This function is a work in progress! The idea is \
                    to automatically adapt the shifts. Hopefully soon ready! \
                    Hopefully...')
    pass


def grid_search_cv_band(estimator, data, events, events_shifted, cv_blocks, win_tol, params_clf, params_data, band_min = 0, band_max = 0, zscore = False):
    
    # Check decoder parameters
    if not check_decoder_params(estimator, params_clf, convert = True):
        raise Exception('ERROR: in check_decoder_params!')
        
    # Check input data
    freqs = []
    for dt in data:
        if dt.ndim !=3:
            raise Exception('ERROR: data are not 3d!')
        freqs.append(dt.shape[2])
    
    if (np.diff(freqs) > 0.1).any():
        raise Exception('ERROR: data have different shape[2]!')
    
    if band_max == 0:
        band_max = freqs[0]
    
    if band_min > freqs[0]:
        raise Exception('ERROR: band_min > data.shape[2]!')
    if band_max > freqs[0]:
        raise Exception('ERROR: band_max > data.shape[2]!')
    if band_max < 1:
        raise Exception('ERROR: band_max < 1')
    if band_min > band_max:
        raise Exception('ERROR: band_min > band_max')
    
    # ========================================================================
    # Build the decoders for the several bands
    models = []
    band_combinations = np.sum(np.arange(band_max-band_min+1))
    band_combinations_count = 0
    for low_band in range(band_min, band_max):
        for high_band in range(low_band, band_max):
            band_combinations_count += 1
            print('Band combination {}/{}'.format(band_combinations_count,band_combinations))
            # Select the data between band_min and band_max
            data_band = []
            for dt in data:
                data_band.append(np.mean(dt[:,:,range(low_band,high_band+1)],axis = 2))
            
            # zscore the data
            data_decode = copy.deepcopy(data_band)
            if zscore:
                data_all = np.concatenate(data_decode)
                data_mean = np.mean(data_all, axis = 0)
                data_std = np.std(data_all, axis = 0)
                
                for iDt, dt in enumerate(data_decode):
                        data_decode[iDt] = (dt - np.tile(data_mean,(dt.shape[0],1)))/np.tile(data_std,(dt.shape[0],1))
                    
            
            # For parameters build decoders
            keys = list(set(params_clf.keys()))
            n_keys = [len(params_clf[key]) for key in keys]
            n_features = np.prod(n_keys)
            
            # For simplicity, now use a for loop over the parameters. Later, I can make 
            # a loop to build all the possible parameters combinations and then loop 
            # through the params_clf
            n_features_count = 0
            for regr_coeff in params_clf['regression_coeff']:
                for th in params_clf['threshold_detect']:
                    for refr_per in params_clf['refractory_period']:
                        n_features_count +=1
                        print('Building decoder {}/{}'.format(n_features_count,n_features))
                        
                        # Model                
                        param_clf = {'regression_coeff': regr_coeff, 'threshold_detect': th, 'refractory_period': refr_per}
                        models_tmp = build_decoder(estimator, data_decode, events, events_shifted, cv_blocks, param_clf, params_data, win_tol)
                        
                        for model in models_tmp:
                            model['params']['band'] = [low_band, high_band]
                        
                        # Update list
                        models.append(models_tmp)
                
    models = flatten_list(models, unique = False)
    # ========================================================================
    # Find the best model
    score = -np.inf
    for model in models:
        if (model['score']>score).any():
            score = np.max(model['score'])
            best_model = model
    
    # grid_search = {'models': models, 'best_model': best_model}
    return models, best_model
    

def grid_search_cv(estimator, data, events, events_shifted, cv_blocks, win_tol, params_clf, params_data, zscore = False):
    
    # Check decoder parameters
    if not check_decoder_params(estimator, params_clf, convert = True):
        raise Exception('ERROR: in check_decoder_params!')
       
    # Check data to be column vectors
    for dt in data:
        dt = transpose(dt,'column')
    
    # ========================================================================
    # Build the decoders
    
    # For parameters build decoders
    keys = list(set(params_clf.keys()))
    n_keys = [len(params_clf[key]) for key in keys]
    n_features = np.prod(n_keys)
    
    # For simplicity, now use a for loop over the parameters. Later, I can make 
    # a loop to build all the possible parameters combinations and then loop 
    # through the params_clf
    models = []
    n_features_count = 0
    for regr_coeff in params_clf['regression_coeff']:
        for th in params_clf['threshold_detect']:
            for refr_per in params_clf['refractory_period']:
                n_features_count +=1
                print('Building decoder {}/{}'.format(n_features_count,n_features))
                
                # zscore the data
                data_decode = copy.deepcopy(data)
                if zscore:
                    data_all = np.concatenate(data_decode)
                    data_mean = np.mean(data_all, axis = 0)
                    data_std = np.std(data_all, axis = 0)
                    
                    for iDt, dt in enumerate(data_decode):
                        data_decode[iDt] = (dt - np.tile(data_mean,(dt.shape[0],1)))/np.tile(data_std,(dt.shape[0],1))
                                
                # Model                
                param_clf = {'regression_coeff': regr_coeff, 'threshold_detect': th, 'refractory_period': refr_per}
                models.append(build_decoder(estimator, data_decode, events, events_shifted, cv_blocks, param_clf, params_data, win_tol))
                
    models = flatten_list(models,  unique = False)
    # ========================================================================
    # Find the best model
    score = -np.inf
    for model in models:
        if (model['score']>score).any():
            score = np.max(model['score'])
            best_model = model
    
    # grid_search = {'models': models, 'best_model': best_model}
    return models, best_model
    

# =============================================================================
# Build decoder
# =============================================================================
def build_decoder(estimator, data, events, events_shifted, cv_blocks, params_clf, params_data, win_tol, plot = False):
    
    # Check decoder parameters
    if not check_decoder_params(estimator, params_clf, convert = False):
        raise Exception('ERROR in check_decoder_params!')
        
    # Check data parameters
    if not check_data_params(params_data):
        raise Exception('ERROR in check_data_params!')
    
    # Check input data
    n_channels = []
    for dt in data:
        if dt.ndim !=2:
            raise Exception('ERROR: data are not 2d!')
        n_channels.append(dt.shape[1])
    
    if (np.diff(n_channels) > 0.1).any():
        raise Exception('ERROR: data have different number of channels!')
        
    # Build the decoder
    if estimator == 'rLDA':
        clf = rLDA(regression_coeff = params_clf['regression_coeff'], 
                   threshold_detect = 0.798,
                   refractory_period = params_clf['refractory_period'])
    else:
        raise Exception('ERROR: no other decoder implemented for the moment!')
    
    # Get number of events
    n_events = events[0].shape[1]

    # Number of decoders
    n_keys = []
    for key, val in params_data.items():
        if key not in ['shifts','win_tol']:
            n_keys.append(len(val))
    n_dataset = np.prod(n_keys)
    
    # Set models
    models = []
    n_dataset_count = 0
    for time_n in params_data['time_n']:
        for feature_win in params_data['feature_win']:
            for dead_win in params_data['dead_win']:
                for no_event in params_data['no_event']:
                    n_dataset_count +=1
                    print('Dataset {}/{}'.format(n_dataset_count,n_dataset))
                    
                    # Data blocks
                    X = []
                    y = []
                    
                    # ========================================================
                    # Extract features for all the blocks
                    
                    # Actual number of no_event to use for each td
                    # no_event_dt = np.round(no_event/len(data)).astype('int')
                    no_event_dt = np.round(no_event).astype('int')
                    
                    for dt, event in zip(data, events_shifted):
                        dt = transpose(dt,'column')
                        event = transpose(event,'column')
                        
                        if dt.ndim == 1:
                            dt = np.expand_dims(dt, axis = 1)
                        
                        # Get features
                        features_n = dt.shape[1] * time_n
                        template = np.round(np.linspace(0,-feature_win,time_n)).astype('int')
                
                        X_tmp, y_tmp = extract_features(dt, event, int(dead_win), no_event_dt, features_n, template)
                        
                        # Append features
                        X.append(X_tmp)
                        y.append(y_tmp)
                    
                    # ========================================================
                    # Build and test decoder
                    n_cv = len(cv_blocks)
                    blocks = np.concatenate(cv_blocks)
                    conf_matrix = np.zeros((len(win_tol), n_events+1, n_events+1))
                    for iCv_train, cv_block in enumerate(cv_blocks):
                        train_blocks = np.setdiff1d(blocks,cv_block)
                        test_blocks = cv_block
                        
                        X_train = np.concatenate([X[blk] for blk in train_blocks], axis = 0)
                        y_train = np.concatenate([y[blk] for blk in train_blocks], axis = 0)
                        
                        # Train decoder
                        clf.fit(X_train,y_train)
                        
                        # ====================================================
                        # Test decoder
                        for iCv_test, cv_blk in enumerate(test_blocks):
                            n_data = data[cv_blk].shape[0]
                            X_test = get_features_data(data[cv_blk], template)
                            y_test = events[cv_blk][-np.min(template):,:]
                            
                            # Predict
                            classes = clf.predict(X_test, probs = False)
                            
                            # Set predictive class            
                            y_pred = np.zeros(y_test.shape).astype('int')
                            for iEv in range(n_events):
                                y_pred[:,iEv][np.where(classes == iEv+1)[0]] = 1

                            # Compute confusion matrix for mutual information
                            events_test = []
                            events_pred = []
                            for iEv in range(n_events):
                                events_test.append(np.where(y_test[:,iEv] == 1)[0] -np.min(template))
                                events_pred.append(np.where(y_pred[:,iEv] == 1)[0] -np.min(template))
                            # print('cut' + str(iCv_test))
                            # print(events_pred)
                            
                            if plot:
                                # Compute class probabilities
                                classes_prob = clf.predict(X_test, probs = True)
                                # Plot probabilities
                                plt.figure()
                                # Plot events
                                colors = ['b','k','r','g','m','y']
                                title = 'Train {}/{}; Test {}/{}\n'.format(iCv_train+1, n_cv,iCv_test+1, len(test_blocks))
                                for iEv, (class_prob, ev_test, ev_pred, col) in enumerate(zip(classes_prob[:,1:].T, events_test,events_pred,colors)):
                                    plt.plot(class_prob, color = col)
                                    plt.vlines(ev_pred+np.min(template),0,1,color = col, linestyle = '--')
                                    plt.scatter(ev_test+np.min(template),1.1*np.ones(ev_test.shape), marker = 'v',color = col)
                                    title += 'Event {}, col: {};'.format(iEv+1,col)
                                plt.title(title)
                            
                            conf_matrix += compute_hetero_kardinality(events_test,events_pred,n_data,win_tol)
                        
                    # Compute mutual information
                    mutual_information = compute_mutual_information_prior_norm(conf_matrix)
                    
                    # Combine params structures
                    params_model = {'time_n': time_n, 'feature_win': feature_win, 'dead_win':dead_win, 
                                    'no_event':no_event, 'win_tol': win_tol, 'shifts': params_data['shifts']}
                    params = combine_dicts((params_clf, params_model), inplace = False)
                    # Save model
                    models.append({'clf': clf, 'params': params, 'score': mutual_information, 'conf_matrix': conf_matrix})
    
    return models


def build_decoder_freq_band(estimator, data, events, events_shifted, cv_blocks, params_clf, params_data, win_tol):
    
    # Check decoder parameters
    if not check_decoder_params(estimator, params_clf, convert = False):
        raise Exception('ERROR in check_decoder_params!')
        
    # Check data parameters
    if not check_data_params(params_data):
        raise Exception('ERROR in check_data_params!')
    
    # Check that data is made of 3d matrixes
    for dt in data:
        if dt.ndim != 3:
            raise Exception('ERROR: data is not made of 3d arrays!')
    
    # Build the decoder
    if estimator == 'rLDA':
        clf = rLDA(regression_coeff = params_clf['regression_coeff'], 
                   threshold_detect = params_clf['threshold_detect'],
                   refractory_period = params_clf['refractory_period'])
    else:
        raise Exception('ERROR: no other decoder implemented for the moment!')
    
    # Get number of events
    n_events = events[0].shape[1]

    # Number of decoders
    n_keys = [len(val) for val in params_data.values()]
    n_dataset = np.prod(n_keys)
    
    # Set models
    models = []
    n_dataset_count = 0
    for time_n in params_data['time_n']:
        for feature_win in params_data['feature_win']:
            for dead_win in params_data['dead_win']:
                for no_event in params_data['no_event']:
                    n_dataset_count +=1
                    print('Dataset {}/{}'.format(n_dataset_count,n_dataset))
                    
                    # Data blocks
                    X = []
                    y = []
                    
                    # ========================================================
                    # Extract features for all the blocks
                    
                    # Actual number of no_event to use for each td
                    no_event_dt = np.round(no_event/len(data)).astype('int')
                    # no_event_dt = 100000
                    for dt, event in zip(data, events_shifted):
                        dt = transpose(dt,'column')
                        event = transpose(event,'column')
                        
                        if dt.ndim == 1:
                            dt = np.expand_dims(dt, axis = 1)
                        
                        # Get features
                        features_n = dt.shape[1] * time_n
                        template = np.round(np.linspace(0,-feature_win,time_n)).astype('int')
                
                        X_tmp, y_tmp = extract_features(dt, event, int(dead_win), no_event_dt, features_n, template)
                        
                        # Append features
                        X.append(X_tmp)
                        y.append(y_tmp)
                    
                    # ========================================================
                    # Build and test decoder
                    blocks = np.concatenate(cv_blocks)
                    conf_matrix = np.zeros((len(win_tol), n_events+1, n_events+1))
                    for cv_block in cv_blocks:
                        train_blocks = np.setdiff1d(blocks,cv_block)
                        test_blocks = cv_block
                        
                        X_train = np.concatenate([X[blk] for blk in train_blocks], axis = 0)
                        y_train = np.concatenate([y[blk] for blk in train_blocks], axis = 0)
                        
                        # Train decoder
                        clf.fit(X_train,y_train)
                        
                        # ====================================================
                        # Test decoder
                        for cv_blk in test_blocks:
                            n_data = data[cv_blk].shape[0]
                            X_test = get_features_data(data[cv_blk], template)
                            y_test = events[cv_blk][-np.min(template):,:]
                            
                            # Predict
                            classes = clf.predict(X_test, probs = False)
                            # classes_prob = clf.predict(X_test, probs = True)
                            # plt.figure();plt.plot(classes_prob)
                            # Set predictive class            
                            y_pred = np.zeros((n_data,n_events))
                            for iEv in range(n_events):
                                y_pred[:,iEv][np.where(classes == iEv+1)[0]] = 1
                                
                            # Compute confusion matrix for mutual information
                            events_test = []
                            events_pred = []
                            for iEv in range(n_events):
                                events_test.append(np.where(y_test[:,iEv] == 1)[0] -np.min(template))
                                events_pred.append(np.where(y_pred[:,iEv] == 1)[0] -np.min(template))
                            
                            conf_matrix += compute_hetero_kardinality(events_test,events_pred,n_data,win_tol)
                        
                    # Compute mutual information
                    mutual_information = compute_mutual_information_prior_norm(conf_matrix)
                    
                    # Combine params structures
                    params_model = {'time_n': time_n, 'feature_win': feature_win, 'dead_win':dead_win, 
                                    'no_event':no_event, 'win_tol': win_tol, 'shifts': params_data['shifts']}
                    params = combine_dicts((params_clf, params_model), inplace = False)
                    # Save model
                    models.append({'clf': clf, 'params': params, 'score': mutual_information})
                        
    return models


# =============================================================================
# Feature extraction
# =============================================================================
def get_features_data(data, template):
    events_features = np.empty( (data.shape[0]+np.min(template), data.shape[1]*len(template)) )
    events_features[:] = np.nan
    
    for iEl in range(-np.min(template), data.shape[0]):
        events_features[iEl+np.min(template),:] = data[(iEl + template).astype('int'),:].reshape(1,data.shape[1]*len(template))
        
    return events_features[~np.isnan(events_features).any(axis=1),:]


def get_features(data, events, template):
    events_features = np.empty( (len(events), data.shape[1]*len(template)) )
    events_features[:] = np.nan
    
    for iEv, event in enumerate(events):
        if event + np.min(template) > 0 and event + np.max(template) < data.shape[0]:
            events_features[iEv,:] = data[(event + template).astype('int'),:].reshape(1,data.shape[1]*len(template))
    
    events_features_zeros = events_features.copy()
    events_features_zeros[np.isnan(events_features).any(axis=1),:] = 0
    
    return events_features[~np.isnan(events_features).any(axis=1),:], events_features_zeros


def extract_features(data, event_data, dead_win, no_event_n, features_n, template, order = 'sequential'):
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
    order : str, optional
        Set the order for the feature to be extracted. It can either be
        "sequencial" or "byfeature". The default is "sequencial".

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
    
    # Set inverval with no detections
    no_detect_interval = -template[-1]
    
    # Collect no event class indexes
    no_events_idx = np.setdiff1d(np.arange(no_detect_interval,event_data.shape[0]), bad_idx.astype('int'))
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


# =============================================================================
# Mutual information
# =============================================================================

def compute_Hetero_Hist_Edges(triggers,vectLength,tolWin):
    '''
    This function cuts intervals around each trigger (histEdges) depending on 
    the tolerance window

    Parameters
    ----------
    triggers : list of np.ndarray, len(n_classes),  [shape(n_events,),...]
        List of events for each class.
    vectLength : int / float
        DESCRIPTION.
    tolWin : int/float / list of int/float
        Size of the window around the events.

    Returns
    -------
    histEdges :
        intervals around each trigger.
    hitBins : 
        Index of the edge for each class.
    negLen : 
        Total size of negative "samples", i.e. time which is not in the bins
    '''

    if type(triggers) is np.ndarray:
        triggers = [triggers]
    if type(triggers) is not list:
        raise Exception('ERROR: trigger must be a list. You inputed a "{}".'.format(type(triggers)))
    
    if type(tolWin) is int or type(tolWin) is float:
        tolWin = [tolWin]
    if type(tolWin) is np.ndarray:
        tolWin = tolWin.tolist()
    if type(tolWin) is not list:
        raise Exception('ERROR: tolWin must be a list. You inputed a "{}".'.format(type(tolWin)))
    
    # Set number of classes
    n_classes = len(triggers)
      
    hitBins = []
    for iTr, trigger in enumerate(triggers):
        hitBins.append(np.zeros((len(tolWin),len(trigger))).astype('int'))
        
    all_triggers = np.concatenate([trigger for trigger in triggers]).astype('int')
    all_labels = np.concatenate([(iTr*np.ones(trigger.shape)) for iTr, trigger in enumerate(triggers)]).astype('int')

    sort_index = np.argsort(all_triggers)
    all_triggers = all_triggers[sort_index]
    all_labels = all_labels[sort_index]
    
    histEdges = []
    negLen = []
    
    for iTw, tlw in enumerate(tolWin):
        halfWin = np.floor(tlw / 2)
        if len(all_triggers) != 0:
            extTrig = np.array([- halfWin] + all_triggers.tolist() + [vectLength + halfWin + 1]).astype('int')
        else:
            extTrig = np.array([- halfWin] + [vectLength + halfWin + 1]).astype('int')
            
        # histEdges_tmp = np.zeros(np.floor(vectLength / tlw).astype('int'),)
        histEdges_tmp = []
        negLen_tmp = 0
        edgeCount = 0
        trigCount = np.zeros(n_classes,).astype('int')
        for trig in range(1,len(extTrig)):
            spaceSize = extTrig[trig] - extTrig[trig - 1] - 2 * halfWin - 1
            if spaceSize <= 0:
                # histEdges_tmp[edgeCount] = np.mean([extTrig[trig] - extTrig[trig - 1]])
                histEdges_tmp.append(np.mean([extTrig[trig], extTrig[trig - 1]]))
                edgeCount += 1;
            else:
                histEdges_tmp.append(extTrig[trig - 1] + halfWin + 0.5)
                histEdges_tmp.append(extTrig[trig] - halfWin - 0.5)
                edgeCount += 2
                negLen_tmp += spaceSize
            
            if trig < len(extTrig)-1:
                hitBins[all_labels[trig - 1]][iTw,trigCount[all_labels[trig - 1]]] = edgeCount -1
                trigCount[all_labels[trig - 1]] += 1
            
        # histEdges.append(histEdges_tmp[:edgeCount])
        histEdges.append(histEdges_tmp)
        negLen.append(negLen_tmp/ (2 * halfWin + 1))
        
    return histEdges, hitBins, negLen


def compute_hetero_kardinality(triggers,triggers_dec,vectLength,tolWin):
    '''
    This function computs the confusion matrix kardinality for each tolerance window

    Parameters
    ----------
    triggers : TYPE
        DESCRIPTION.
    triggers_dec : TYPE
        DESCRIPTION.
    tolWin : TYPE
        DESCRIPTION.
    histEdges : TYPE
        DESCRIPTION.
    hitBins : TYPE
        DESCRIPTION.
    negLen : TYPE
        DESCRIPTION.

    Returns
    -------
    kardinality : TYPE
        DESCRIPTION.

    '''
    
    histEdges, hitBins, negLen = compute_Hetero_Hist_Edges(triggers,vectLength,tolWin)
    
    n_classes = len(triggers) + 1;
    n_tol_win = len(tolWin)
    kardinality = np.empty((n_tol_win,n_classes,n_classes))
    kardinality[:] = np.nan
    
    for iTw in range(n_tol_win):
        hit_bins_tmp = [hitBins[trig][iTw,:] for trig in range(n_classes - 1)]
        hit_bins_tmp.append(np.setdiff1d(np.arange(len(histEdges[iTw])), np.concatenate(hit_bins_tmp)))
        
        # Count the number of times each detected event falls into the
        # intervals around the bin.
        count = []
        for trig in range(n_classes - 1):
            if len(triggers_dec[trig]) != 0:
                hist, _ = np.histogram(triggers_dec[trig], bins=histEdges[iTw])
                count.append(np.concatenate((hist,[np.sum(histEdges[iTw][-1] == triggers_dec[trig])])))
            else:
                count.append(np.zeros(len(histEdges[iTw]),))
        count = np.array(count)
        # Compute the confusion matrix
        for trig in range(n_classes - 1): # trigger's class
            for clDec in range(n_classes - 1): # decoded class
                if trig == clDec:
                    count_tr = count[np.setdiff1d(np.arange(n_classes-1),trig),hit_bins_tmp[clDec]]
                    if count_tr.ndim == 1:
                        noFalseHits = count_tr == 0
                    else:
                        noFalseHits = np.sum(count_tr,axis = 0) == 0
                    
                    kardinality[iTw,trig,clDec] = np.sum(np.logical_and(count[trig,hit_bins_tmp[clDec]] > 0, noFalseHits))
                else:
                    kardinality[iTw,trig,clDec] = np.sum(count[trig,hit_bins_tmp[clDec]] > 0)
            
            kardinality[iTw,trig,n_classes-1] = np.sum(count[trig, hit_bins_tmp[n_classes-1]])
        
        # and the true negatives as well
        for clDec in range(n_classes - 1):
            n_hits = np.sum(count[:,hit_bins_tmp[clDec]],axis = 0) == 0
            kardinality[iTw,n_classes-1,clDec] = np.sum(n_hits)
        
        kardinality[iTw,n_classes-1,n_classes-1] = np.max([0,negLen[iTw] - np.sum(kardinality[iTw,range(n_classes - 1),-1])]);
    
    return kardinality


def compute_mutual_information_prior_norm(all_kardinality):
    '''
    

    Parameters
    ----------
    all_kardinality : np.ndarray, shape (N,n_classes,n_classes)
        DESCRIPTION.
        
    Returns
    -------
    grpMutInfo : TYPE
        DESCRIPTION.

    '''
    mat_dim = all_kardinality.ndim
    n_classes = all_kardinality.shape[-1]
    if all_kardinality.shape[-1] != all_kardinality.shape[-2]:
        raise ValueError('ERROR: kardinality matrix is not squared!')
        
    prior_dim = mat_dim-2
    posterior_dim = mat_dim-1
    
    if mat_dim == 2:
        sum_kardinality = np.tile(np.nansum(np.nansum(all_kardinality,axis = -1),axis = -1),(n_classes,n_classes))
    elif mat_dim == 3:
        sum_kardinality = np.moveaxis(np.tile(np.nansum(np.nansum(all_kardinality,axis = -1),axis = -1),(n_classes,n_classes,1)), -1, 0)
        
    norm_kardinality = joint_prob = all_kardinality/sum_kardinality
    
    prior = np.nansum(norm_kardinality,prior_dim)
    posterior = np.nansum(norm_kardinality,posterior_dim)    
    mutInfo = joint_prob * np.log2( joint_prob / (np.moveaxis(np.tile(prior,(n_classes,1,1)),1,0)*np.moveaxis(np.tile(posterior,(n_classes,1,1)),0,-1)) )
    mutInfo = np.nansum(np.nansum(mutInfo,axis = -1),axis = -1)
    
    prior_entropy = prior*np.log2(prior)
    prior_entropy = -np.nansum(prior_entropy,prior_dim)
    
    if type(prior_entropy) is not np.ndarray:
        prior_entropy = np.array([prior_entropy])
    
    grpMutInfo = np.zeros(mutInfo.shape)
    good_idx = np.where(prior_entropy != 0)[0]
    bad_idx = np.where(prior_entropy == 0)[0]
    if len(good_idx) != 0:
        grpMutInfo[good_idx] = mutInfo[good_idx] / prior_entropy[good_idx]
    
    if len(bad_idx) != 0:
        mutInfo[bad_idx] = 0
    
    return grpMutInfo


# =============================================================================
# Data separation
# =============================================================================
def prepare_cv_uniform(data, events, deviation = 0.2, cv_division = 4, blocks_start = [], blocks_stop = []):
    '''
    This function prepare the data in uniform blocks to perfom CV.
    It assumes uniform distribution of events

    Parameters
    ----------
    data : list of np.ndarray, shape(n_samples, n_channels, n_frequencies)
        DESCRIPTION.
    events : list of np.ndarray, shape(n_samples, n_events)
        DESCRIPTION.
    deviation : TYPE
        DESCRIPTION.
    cv_division : TYPE
        DESCRIPTION.
    blocks_start : TYPE
        DESCRIPTION.
    blocks_stop : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    # Check input data
    if type(data) is np.ndarray:
        data = [transpose(data,'column')]
        
    if type(events) is np.ndarray:
        events = [transpose(events,'column')]
    
    if type(data) is not list:
        raise Exception('ERROR: data must be a list!')
        
    if type(events) is not list:
        raise Exception('ERROR: events must be a list!')
    
    # Prepare data
    is_parted = False
    while not is_parted:
        # Distribute the cuts throughout the cv_blocks
        n_blocks = len(data)
        len_blocks = []
        for dt in data:
            len_blocks.append(dt.shape[0])
            # disp('TESTING: prepare_cv_uniform uses data length based on events!!!'); % [TESTING ONLY] Determine data length based on actual trigger positions
            # len_blocks(cut) = max(cell2mat(events(cut, :)')) - min(cell2mat(events(cut, :)')) + 1;
        len_blocks = np.array(len_blocks)
        if len(blocks_start)==0 or len(blocks_stop)==0:
            blocks_stop = np.cumsum(len_blocks)-1
            blocks_start = np.insert(blocks_stop[:-1]+1,0,0)
        
        # Sort lengths in descending order
        sort_index = np.argsort(len_blocks)
        sort_index = sort_index[::-1]
        len_blocks_sort = len_blocks[sort_index]
    
        # Set divisions lengths
        targetLen = np.sum(len_blocks) / cv_division
        
        cv_blocks = []
        for div in range(cv_division):
            cv_blocks.append(np.array([],dtype = 'int'))
            
        go_pos = True
        currentInd = -1
        while len(len_blocks_sort) != 0:
            # Loop over the cuts in the following way: [1 ... cv_division cv_division ... 1 ...]
            if go_pos:
                if currentInd == cv_division-1:
                    go_pos = False
                else:
                    currentInd += 1
            else:
                if currentInd == 0:
                    go_pos = True
                else:
                    currentInd -= 1;
            
            # If this fold is already above target length, go to next iteration
            if len(cv_blocks[currentInd]) != 0 and np.sum(len_blocks[cv_blocks[currentInd]]) > targetLen:
                continue
            
            # Loop over cut by decreasing lengths and attribute the first that  
            # keeps the below target length condition
            for currentCutInd in range(len(len_blocks_sort)):
                if len(cv_blocks[currentInd]) != 0:
                    if (np.sum(len_blocks[cv_blocks[currentInd]]) + len_blocks_sort[currentCutInd]) < targetLen:
                        cv_blocks[currentInd] = np.insert(cv_blocks[currentInd],0,sort_index[currentCutInd])
                        sort_index = np.delete(sort_index,currentCutInd)
                        len_blocks_sort = np.delete(len_blocks_sort,currentCutInd)
                        break
                elif len_blocks_sort[currentCutInd] < targetLen:
                    cv_blocks[currentInd] = np.insert(cv_blocks[currentInd],0,sort_index[currentCutInd])
                    sort_index = np.delete(sort_index,currentCutInd)
                    len_blocks_sort = np.delete(len_blocks_sort,currentCutInd)
                    break
                        
                if currentCutInd == len(len_blocks_sort)-1:
                    cv_blocks[currentInd] = np.insert(cv_blocks[currentInd],0,sort_index[currentCutInd])
                    sort_index = np.delete(sort_index,currentCutInd)
                    len_blocks_sort = np.delete(len_blocks_sort,currentCutInd)
    
        #% Test whether the distribution is within the tolerance
        partLen = []
        for cvMember in cv_blocks:
            cvMember.sort()
            partLen.append(np.sum(len_blocks[cvMember]))
        partLen = np.array(partLen)
        
        # If the blocks have the right dimensions, ok. Otherwise cut the largest block in two
        if np.sum(np.logical_and(partLen > targetLen*(1 - deviation), partLen < targetLen * (1 + deviation))) == cv_division:
            is_parted = True
        else:
            sort_index = np.argsort(len_blocks)
            sort_index = sort_index[::-1]
            cut = sort_index[0]
            
            # Cut
            events_cut = find_values(events[cut],1)
            if len(events_cut) == 0:
                cutPoint = np.round(data[cut].shape[0] / 2).astype('int')
            elif len(events_cut) == 1:
                if (data[cut].shape[0] - events_cut) > events_cut:
                    cutPoint = np.round((data[cut].shape[0] + events_cut) / 2).astype('int')
                else:
                    cutPoint = np.round(events_cut / 2).astype('int')
            else:
                halfTrigInd = np.floor(len(events_cut) / 2).astype('int')
                cutPoint = np.round((events_cut[halfTrigInd-1]+events_cut[halfTrigInd]) / 2).astype('int')
            
            # Rearrange events
            events.insert(cut+1, events[cut][cutPoint+1:,:])
            events[cut] = events[cut][:cutPoint+1,:]
    
            # Rearrange data
            if data[cut].ndim == 2:
                data.insert(cut+1, data[cut][cutPoint+1:,:])
                data[cut] = data[cut][:cutPoint+1,:]
            elif data[cut].ndim == 3:
                data.insert(cut+1, data[cut][cutPoint+1:,:,:])
                data[cut] = data[cut][:cutPoint+1,:,:]
            else:
                raise Exception('ERROR: data have dimension > 3! This has never been implemented!')
            
            # Rearrange start stop indexes
            blocks_start = np.insert(blocks_start, cut+1, blocks_start[cut]+cutPoint+1)
            
            blocks_stop[cut] = blocks_start[cut]+cutPoint
            blocks_stop = np.insert(blocks_stop, cut+1, blocks_start[cut+1]+data[cut+1].shape[0]-1)

    return cv_blocks, data, events, n_blocks, blocks_start, blocks_stop


# =============================================================================
# Stats
# =============================================================================
def bootstrap_MInorm_simple(conf_matrix, no_straps = 10000):
    
    if conf_matrix.ndim != 2:
        raise Exception('ERROR: conf_matrix dimension > 2!')
    
    boot = dict()
    
    n_data = np.sum(conf_matrix)
    boot_n_freedom = n_data - 1
    boot_data = boostrap(conf_matrix, no_straps)
    
    boot_mi = compute_mutual_information_prior_norm(np.moveaxis(boot_data,-1,0))
    real_mi = float(compute_mutual_information_prior_norm(conf_matrix))
    
    boot_std = np.std(boot_mi)
    boot_mean, boot_95conf_low, boot_95conf_high = confidence_interval_sample(boot_mi, confidence=0.95)
    
    boot_95percentile = np.percentile(boot_mi,[2.5, 97.5]) - boot_mean + real_mi

    # Save in boot dict
    boot['n_freedom'] = boot_n_freedom
    boot['boot_mi'] = boot_mi
    boot['real_mi'] = real_mi
    boot['boot_mean'] = boot_mean
    boot['boot_std'] = boot_std
    boot['boot_95conf_low'] = boot_95conf_low
    boot['boot_95conf_high'] = boot_95conf_high
    boot['boot_95percentile'] = boot_95percentile

    return boot

# =============================================================================
# Parameters configuration
# =============================================================================
def check_decoder_params(estimator, params, convert = True):
    
    # Necessary parameters in params
    if estimator == 'rLDA':
        params_fields = ['regression_coeff','threshold_detect','refractory_period']
    
    if type(params) is not dict:
        raise Exception('ERROR: params must be a dict! You inputed a "{}".'.format(type(params)))
    
    if (estimator == 'rLDA') and (not is_field(params,params_fields)):
        raise Exception('ERROR: some fields are missing from params!')
    
    if convert:
        for key,val in params.items():
            if type(val) is int or type(val) is float:
                params[key] = [val]
                
            if type(val) is np.ndarray:
                params[key] = val.tolist()
            
            if type(params[key]) is not list:
                raise Exception('ERROR: key: {}. val must be a list! You inputed a "{}".'.format(key,type(val)))
    
    return True

def check_data_params(params):
    
    # Necessary parameters in params
    params_fields = ['time_n','feature_win','dead_win','no_event','shifts']
    
    if type(params) is not dict:
        raise Exception('ERROR: params must be a dict! You inputed a "{}".'.format(type(params)))
    
    if not is_field(params,params_fields):
        raise Exception('ERROR: some fields are missing from params!')
    
    for key,val in params.items():
        if type(val) is int or type(val) is float:
            params[key] = [int(val)]
            
        if (type(val) is list) and (len(val) != 0):
            params[key] = [int(x) for x in val]
            
        if type(val) is np.ndarray:
            params[key] = val.tolist()
            
        if type(params[key]) is not list:
            raise Exception('ERROR: key: {}. val must be a list! You inputed a "{}".'.format(key,type(val)))
    
    return True


# =============================================================================
# Plotting
# =============================================================================
def plot_MI_2d_tol(models, Xval, Yval, tol, kind, return_matrix = False, xticks = [], yticks = []):
    '''
    This function computes and plots the MI matrix for different shifts applied
    to the events. This function is limited to one decoder type with multiple 
    shifts.

    Parameters
    ----------
    models : list
        Models you used for decoding.
    Xval : np.ndarray, shape(n_xval,)
        X values where the MI was computed.
    Yval : np.ndarray, shape(n_yval,)
        Y values where the MI was computed.
    tol : np.ndarray, shape(n_tol,)
        Tolerances where the MI was computed.
    kind : str
        Type of information to be plotted, stores in the models. It can either
        be "band" or "shifts".
    return_matrix : bool, optional
        Return the MI matrix. The default is False.

    Returns
    -------
    mi_2d : np.ndarray, shape(n_xval,n_yval)
        Matrix with MI values computed over the X Y values.

    '''
    
    # ========================================================================
    # Check input variables
    if type(models) is dict:
        models = [models]
    if type(models) is not list:
        raise Exception('ERROR: models is not a list!')
    
    if type(Xval) is int or type(Xval) is float:
        Xval = np.array([Xval])
    if type(Xval) is list:
        Xval = np.array(Xval)
    if type(Xval) is not np.ndarray:
        raise Exception('ERROR: Xval is not a np.ndarray! You inputed a "{}"'.format(type(Xval)))
    
    if type(Yval) is int or type(Yval) is float:
        Yval = np.array([Yval])
    if type(Yval) is list:
        Yval = np.array(Yval)
    if type(Yval) is not np.ndarray:
        raise Exception('ERROR: Yval is not a np.ndarray! You inputed a "{}"'.format(type(Yval)))
    
    if type(tol) is int or type(tol) is float:
        tol = np.array([tol])
    if type(tol) is list:
        tol = np.array(tol)
    if type(tol) is not np.ndarray:
        raise Exception('ERROR: tol is not a np.ndarray!')
    
    for t in tol:
        if not (np.abs(models[0]['params']['win_tol']-t)<0.1).any():
            raise Exception('ERROR: tol is not in models win_tol!')
    
    if kind not in ['band','shifts']:
        raise Exception('ERROR: kind can either be "band" or "shifts"! You inputed "{}"'.format(kind))
        
    # ========================================================================
    # Compute the matrix
    mi_2d = np.empty((len(Xval),len(Yval),len(tol)))
    mi_2d[:] = np.nan
    for iT, t in enumerate(tol):
        for iX, xval in enumerate(Xval):
            for iY, yval in enumerate(Yval):
                shifts_found = False
                for model in models:
                    if kind == 'shifts':
                        if model['params'][kind] == [xval, yval]:
                            tol_of_interest = np.where(np.abs((model['params']['win_tol'] - t)) < 0.1)[0]
                            mi_2d[iX,iY,iT] = model['score'][tol_of_interest]
                            shifts_found = True
                            break
                    elif kind == 'band':
                        if model['params'][kind] == [xval, yval] or model['params'][kind] == [yval,xval]:
                            tol_of_interest = np.where(np.abs((model['params']['win_tol'] - t)) < 0.1)[0]
                            mi_2d[len(Xval)-iX-1,iY,iT] = model['score'][tol_of_interest]
                            shifts_found = True
                            break
                if shifts_found == False:
                    raise Exception('ERROR: Values [{}, {}] not found!'.format(xval,yval))
    
    if kind == 'band':
        for iT, t in enumerate(tol):
            for iY in range(len(Yval)):
                for iX in range(iY+1,len(Xval)):
                        mi_2d[iX,len(Yval)-iY-1,iT] = 0
    
    # Plot
    centers = [Xval.min(),Xval.max(),Yval.max(),Yval.min()]
    dx, = np.diff(centers[:2])/(mi_2d.shape[1]-1)
    dy, = -np.diff(centers[2:])/(mi_2d.shape[0]-1)
    extent = [centers[0]-dx/2, centers[1]+dx/2, centers[2]+dy/2, centers[3]-dy/2]
    
    for iT, t in enumerate(tol):
        fig, ax = plt.subplots(1,1)
        cs = ax.imshow(mi_2d[:,:,iT], interpolation=None, extent=extent, aspect='auto', cmap = parula)
        
        if kind == 'band':
            plt.xlabel('Frequency band start [Hz]')
            plt.ylabel('Frequency band stop [Hz]')
            if len(xticks) == 0:
                plt.xticks(np.arange(centers[0], centers[1]+dx, dx))
            else:
                plt.xticks(np.arange(centers[0], centers[1]+dx, dx), labels = np.around(xticks, decimals=2), rotation=90)
            if len(yticks) == 0:
                plt.yticks(np.arange(centers[3], centers[2]+dy, dy),labels = np.arange(centers[2], centers[3]-dy, -dy).astype('int'))
            else:
                plt.yticks(np.arange(centers[3], centers[2]+dy, dy),labels = np.around(yticks, decimals=2), rotation=0)
        elif kind == 'shifts':
            plt.xlabel('2nd shift')
            plt.ylabel('1st shift')
            plt.xticks(np.arange(centers[0], centers[1]+dx, dx))
            plt.yticks(np.arange(centers[3], centers[2]+dy, dy))
        
        plt.title('Decoder MI for tolerance window +-{} ms'.format(int(t)))
        
        fig.colorbar(cs, ax = ax)
    
    if return_matrix:
        return mi_2d
    
    
    
    
# EOF