#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:58:34 2020

@author: raschell
"""

import numpy as np
from utils import transpose, find_values, convert_list_to_array

# =============================================================================
# Feature extraction
# =============================================================================

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


def compute_hetero_kardinality(triggers,finalDetPos,vectLength,tolWin):
    '''
    This function computs the confusion matrix kardinality for each tolerance window

    Parameters
    ----------
    triggers : TYPE
        DESCRIPTION.
    finalDetPos : TYPE
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
            if len(finalDetPos[trig]) != 0:
                hist, _ = np.histogram(finalDetPos[trig], bins=histEdges[iTw])
                count.append(np.concatenate((hist,[np.sum(histEdges[iTw][-1] == finalDetPos[trig])])))
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

    mat_dim = all_kardinality.ndim
    n_classes = all_kardinality.shape[-1]
    if all_kardinality.shape[-1] != all_kardinality.shape[-2]:
        raise ValueError('ERROR: kardinality matrix is not squared!')
        
    prior_dim = mat_dim-2
    posterior_dim = mat_dim-1
    
    sum_kardinality = np.moveaxis(np.tile(np.nansum(np.nansum(all_kardinality,axis = -1),axis = -1),(n_classes,n_classes,1)), -1, 0)
    norm_kardinality = joint_prob = all_kardinality/sum_kardinality
    
    prior = np.nansum(norm_kardinality,prior_dim)
    posterior = np.nansum(norm_kardinality,posterior_dim)    
    mutInfo = joint_prob * np.log2( joint_prob / (np.moveaxis(np.tile(prior,(n_classes,1,1)),1,0)*np.moveaxis(np.tile(posterior,(n_classes,1,1)),0,-1)) )
    mutInfo = np.nansum(np.nansum(mutInfo,axis = -1),axis = -1)
    
    prior_entropy = prior*np.log2(prior)
    prior_entropy = -np.nansum(prior_entropy,prior_dim)
    
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
    data : TYPE
        DESCRIPTION.
    blocks_start : TYPE
        DESCRIPTION.
    blocks_stop : TYPE
        DESCRIPTION.
    events : TYPE
        DESCRIPTION.
    deviation : TYPE
        DESCRIPTION.
    cv_division : TYPE
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
            len_blocks.append(len(dt))
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
                print('WARNING: data dimension = 3! Never tested before. Check the data!')
            else:
                raise Exception('ERROR: data have dimension > 3! This has never been implemented!')
            
            # Rearrange start stop indexes
            blocks_start = np.insert(blocks_start, cut+1, blocks_start[cut]+cutPoint+1)
            
            blocks_stop[cut] = blocks_start[cut]+cutPoint
            blocks_stop = np.insert(blocks_stop, cut+1, blocks_start[cut+1]+data[cut+1].shape[0]-1)

    return cv_blocks, data, events, n_blocks, blocks_start, blocks_stop

# EOF