#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:19:56 2020

@author: raschell
"""

# Import data management libraries
import numpy as np

# Import loading functions
from loading_data import load_data_from_folder
# Import data processing
from td_utils import remove_fields, is_field, combine_fields, td_plot

folder = ['/Volumes/MK_EPIOS/HUMANS/from ED']
file_num = [[3,4,5]]

file_format = '.mat'

#%% Load data  
# Load LFP
td = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format)

# Remove fields from td
td = remove_fields(td,['EMG','KIN','EV'])

# Load gait events
td_gait = load_data_from_folder(folder = folder,file_num = file_num,file_format = file_format, pre_ext = '_B33_MANUAL_gaitEvents')

td = combine_fields(td, td_gait)

#%% Prepare data for each dataset
    
plot_data = False
signal_threshold = 300

dataset_len = len(td)

# signal_to_use = ['LFP_BIP7','LFP_BIP8','LFP_BIP9','LFP_BIP10','LFP_BIP11','LFP_BIP12',
#                   ['LFP_BIP7','LFP_BIP8'],['LFP_BIP8','LFP_BIP9'],['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]
# signal_to_use = [['LFP_BIP11','LFP_BIP12']]
signal_to_use = ['LFP_BIP10','LFP_BIP11','LFP_BIP12',
                  ['LFP_BIP10','LFP_BIP11'],['LFP_BIP11','LFP_BIP12']]

# events_to_use = ['MANUAL_EV_RFS_time','MANUAL_EV_RFO_time','MANUAL_EV_LFS_time','MANUAL_EV_LFO_time']
# events_label = ['RFS','RFO','LFS','LFO']
# events_to_use = ['MANUAL_EV_RFS_time','MANUAL_EV_LFS_time']
events_to_use = ['MANUAL_EV_RFS_time']
# events_label = ['RFS','LFS']
events_label = ['RFS']
events_type = 'time'

signal_n = len(signal_to_use)
if not is_field(td, signal_to_use):
    raise Exception('Missing fields in td list!')

if len(events_to_use) != len(events_label):
    raise Exception('Different number of events selected!')

# Decoder list
td_decoder = []

# Loop over the files
for iTd, td_tmp in enumerate(td):
    print('Preparing file {}: {}/{}'.format(td_tmp['File'], iTd+1, dataset_len))
    # Len dataset 
    if type(signal_to_use[0]) is list:
        signal_len = len(td_tmp[signal_to_use[0][0]])
    else:
        signal_len = len(td_tmp[signal_to_use[0]])
    
    signal_FS = td_tmp['LFP_Fs']
    junkOffset = np.ceil(signal_FS/2).astype('int')
    # Set an array for good indexes
    goodIdx = np.zeros((signal_len,), dtype=int)
    
    # Create temporary data
    signal_tmp = np.array([]).reshape(0,signal_len)
    # Collect signals name
    signal_names = []
    
    # Loop over the signals
    for sgl in signal_to_use:
        if type(sgl) == list:
            signal_tmp = np.vstack([signal_tmp, np.array(td_tmp[sgl[0]]) - np.array(td_tmp[sgl[1]]) ])
            signal_names.append('{} - {}'.format(sgl[0],sgl[1]))
        else:
            signal_tmp = np.vstack([signal_tmp, np.array(td_tmp[sgl])])
            signal_names.append(sgl)
        goodIdx += np.logical_and(signal_tmp[-1,:] > -signal_threshold, signal_tmp[-1,:] < signal_threshold).astype('int')
    
    
    # Collect good indexes
    goodIdx = (goodIdx > signal_n-1).astype('int')
    
    # Separate dataset in good epochs
    good_start = np.where(np.logical_and(goodIdx[:-1]==False ,goodIdx[1:]==True))
    if goodIdx[0] == 1:
        good_start = np.insert(good_start,0,0)
        
    good_stop = np.where(np.logical_and(goodIdx[:-1]==True ,goodIdx[1:]==False))
    if goodIdx[-1] == 1:
        good_stop = np.append(good_stop,signal_len)
    
    # Check that good_start and good_stop have same length
    if len(good_start) != len(good_stop):
        raise Exception('Start and Stop epochs have different length!')
    
    good_epoch = np.where((good_stop - good_start + 1) > 2*signal_FS)
    good_start = good_start[good_epoch]
    good_stop = good_stop[good_epoch]
    print('Start: {}; Stop: {}'.format(good_start, good_stop))
    
    # Conver events from time based to sample based
    if events_type == 'time':
        for event_lbl, event in zip(events_label, events_to_use):
            td_tmp[event_lbl] = [np.where(np.array(td_tmp['LFP_time']) >= ev)[0][0] for ev in td_tmp[event]]
                    
    # Divide ephocs
    epochs_len = len(good_start)
    for idx_ep, (ep_start, ep_stop) in enumerate(zip(good_start,good_stop)):
        if ep_start == 0:
            ep_start_junk = 0
        else:
            ep_start_junk = ep_start+junkOffset
        if ep_stop == signal_len:
            ep_stop_junk = signal_len
        else:
            ep_stop_junk = ep_stop-junkOffset
        take_idx = np.arange(ep_start_junk, ep_stop_junk)
        
        gait_events_tmp = {}
        gait_events_empty = np.zeros((len(events_label), ), dtype=bool)
        for idx_ev, event in enumerate(events_label):
            gait_events_tmp[event] = np.array(td_tmp[event])[np.where(np.logical_and(td_tmp[event] >= take_idx[0],td_tmp[event] <= take_idx[-1]))] - take_idx[0] + 1
            if gait_events_tmp[event].size == 0:
                gait_events_empty[idx_ev] = True
        
        # Update variables
        if not all(gait_events_empty):
            td_decoder_tmp = {}
            td_decoder_tmp['file_names'] = td_tmp['File']
            td_decoder_tmp['Fs'] = signal_FS
            
            td_decoder_tmp['signal_names'] = signal_names
            td_decoder_tmp['gait_events_names'] = events_label
            
            for iSgl, signal in enumerate(signal_names):
                td_decoder_tmp[signal] = signal_tmp[iSgl,take_idx]
            for event in events_label:
                td_decoder_tmp[event] = gait_events_tmp[event]
            
            td_decoder.append(td_decoder_tmp)
        else:
            print('Epoch #{}/{} removed! No gait events in it.'.format(idx_ep+1, epochs_len))
    

#%% Plot data

if plot_data:
    import matplotlib.pyplot as plt
    from td_process import event_color, event_linestyle

    signal_to_plot = ['LFP_BIP7','LFP_BIP8','LFP_BIP9']
    events_to_plot = ['RFS','LFS']
    subplot = (len(signal_to_plot),1)
        

    for td_tmp in td_decoder: 
        
        fig, axs = plt.subplots(nrows = subplot[0], ncols = subplot[1])
        fig.suptitle('File {}'.format(td_tmp['file_names']), fontsize=10)
        
        for idx, ax in enumerate(np.ndarray.tolist(axs.reshape(len(signal_to_plot),1))):
            
            ax[0].set_title(signal_to_plot[idx])
            
            if idx % 2 == 0:
                ax[0].set_ylabel('mV')
            
            if idx+1 % 5 != 0:
                ax[0].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
                            
            ax[0].set_xlabel('')
            ax[0].grid(True)
            ax[0].set_ylim((-50,50))
            ax[0].plot(td_tmp[signal_to_plot[idx]])
            
            for event in events_to_plot:
                if 'R' in event:
                    line_style = event_linestyle.R.value
                else:
                    line_style = event_linestyle.L.value
                    
                if 'FS' in event:
                    col = event_color.FS.value
                else:
                    col = event_color.FO.value
                    
                for ev in td_tmp[event]:
                    ax[0].axvline(ev,-50, 50, color = col, linestyle = line_style)
        
#%% Process data blocks 

from power_estimation import moving_pmtm

# freq_bands = [[13,20],[20,30]]
# filt_order = 3

blocks_len = len(td_decoder)

# Multitaper information
window_size_sec = 0.25 # in seconds
window_step_sec = 0.01 # in seconds
freq_range = [10,100]
NW = 4
tapers_n = np.floor(2*NW)-1

window_size_smp = round(window_size_sec*td_decoder[0]['Fs'])
window_step_smp = round(window_step_sec*td_decoder[0]['Fs'])

for iTd, td_tmp in enumerate(td_decoder):
    print(' ');
    print('Processing block {}/{}'.format(iTd+1, blocks_len))
    
    # Multitaper
    Fs = td_tmp['Fs']

    signal_names_spmt = []
    for iSgl, signal in enumerate(td_tmp['signal_names']):
        print('Processing signal {}/{}'.format(iSgl+1, len(td_tmp['signal_names'])))
        td_tmp[signal+'_spmt'], sfreqs, stimes = moving_pmtm(td_tmp[signal], Fs, window_size_smp, window_step_smp, freq_range, NW=NW, NFFT=None, verbose=False)
        signal_names_spmt.append(signal+'_spmt')

    td_tmp['Fs_spmt'] = 1/(stimes[1] - stimes[0])
    
    gait_events_name_spmt = []
    for iEv, event in enumerate(events_label):
        gait_events_name_spmt.append('{}_spmt'.format(event))
        td_tmp[gait_events_name_spmt[iEv]] = [np.where(stimes >= ev/Fs)[0][0] for ev in td_tmp[event]]
        
    td_tmp['gait_events_name_spmt'] = gait_events_name_spmt
    td_tmp['signal_names_spmt'] = signal_names_spmt
    td_tmp['frequencies_spmt'] = sfreqs
    td_tmp['time_spmt'] = stimes
    
print(' ')
print('DATA PROCESSED!')
#%% Plot multitaper
import matplotlib.pyplot as plt
from td_process import event_color, event_linestyle
from matplotlib import cm

if plot_data:    
    iTd = 0
    td_tmp = td_decoder[iTd]
    signal_name = 'LFP_BIP11 - LFP_BIP12_spmt'
    
    x = np.linspace(0, td_tmp[signal_name].shape[1], 100)
    y = np.linspace(0, td_tmp[signal_name].shape[0], td_tmp[signal_name].shape[0])
    
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(td_tmp[signal_name], extent = [0,td_tmp[signal_name].shape[1],0,td_tmp[signal_name].shape[0]], aspect='auto') # left, right, bottom, top
    # plt.imshow(np.flip(mt_spectrogram.T, axis=0), extent = [0,1,0,1] ) # left, right, bottom, top
    # plt.axes().set_aspect('equal', 'datalim')
    plt.xticks(x, np.around(np.linspace(td_tmp['time_spmt'][0], td_tmp['time_spmt'][-1], 100),decimals = 1))
    plt.yticks(y, td_tmp['frequencies_spmt'])
    
    event = 'RFS_spmt'
    if 'R' in event:
        line_style = event_linestyle.R.value
    else:
        line_style = event_linestyle.L.value
        
    if 'FS' in event:
        col = event_color.FS.value
    else:
        col = event_color.FO.value
        
    for ev in td_tmp[event]:
        plt.axvline(ev,-50, 50, color = col, linestyle = line_style)
    
    
    # Contourmap
    x = td_tmp['time_spmt']
    y = td_tmp['frequencies_spmt']
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, np.flip(td_tmp[signal_name],axis=0), cmap=cm.inferno)
    cbar = fig.colorbar(cs)
    
    event = 'RFS'
    if 'R' in event:
        line_style = event_linestyle.R.value
    else:
        line_style = event_linestyle.L.value
        
    if 'FS' in event:
        col = event_color.FS.value
    else:
        col = event_color.FO.value
        
    for ev in td_tmp[event]:
        plt.axvline(ev/Fs,-50, 50, color = 'b', linestyle = line_style)


#%% Extract the features

def extract_features(events,data,template):
    events_features = np.empty( (len(events), data.shape[1]*len(template)) )
    events_features[:] = np.nan
    
    for iEv, event in enumerate(events):
        if event + np.min(template) > 0 and event + np.max(template) < data.shape[0]:
            events_features[iEv,:] = data[(event + template).astype('int'),:].reshape(1,data.shape[1]*len(template))
    
    events_features_zeros = events_features.copy()
    events_features_zeros[np.isnan(events_features).any(axis=1),:] = 0
    
    return events_features[~np.isnan(events_features).any(axis=1),:], events_features_zeros


# time_n = [5,7,10] #10
# feature_win_sec = [0.3, 0.5] # 0.5
# dead_win_sec = [0.01, 0.02, 0.04]
# neg_train_sec = [100, 200, 300]
# regularization_coeff = [0.01, 0.1, 0.5] #0.01
# refractory_sec = 0.5

time_n = [10] #10
feature_win_sec = [0.5] # 0.5
dead_win_sec = [0.02]
neg_train_sec = [1000]
regularization_coeff = [0.01] #0.01
refractory_sec = 0.5

blocks_n = len(td_decoder)
sample_rate = td_decoder[0]['Fs_spmt']
channels_n = len(td_decoder[0]['signal_names_spmt'])
freq_n = td_decoder[0][td_decoder[0]['signal_names_spmt'][0]].shape[0]
models_n = len(time_n)*len(feature_win_sec)*len(dead_win_sec)*len(neg_train_sec)*len(regularization_coeff)
models = []

counter = 1
for iTime in time_n:
    for iFeat in feature_win_sec:
        for iDeadWin in dead_win_sec:
            for iNegEv in neg_train_sec:
                for iReg in regularization_coeff:
                    print('#################################################')
                    print('Building model {}/{}'.format(counter, models_n))
                    
                    # Initialise model dictionary
                    model = {}
                    
                    neg_train_smp = np.round(iNegEv*sample_rate).astype('int')
                    dead_win_smp = np.round(iDeadWin*sample_rate).astype('int')
                    feature_win_smp = np.round(iFeat*sample_rate).astype('int')
                    refractory_smp = np.round(refractory_sec*sample_rate).astype('int')
                    neg_train_smp = 100000 
                    
                    features_n = channels_n * iTime * freq_n
                    
                    template = np.round(np.linspace(0,-feature_win_smp,iTime)).astype('int')
                        
                    params = {'sample_rate': sample_rate,
                             'time_n': iTime,
                             'channels_n': channels_n,
                             'freq_n': freq_n,
                             'regularization_coeff': iReg,
                             'neg_train_sec': iNegEv,
                             'dead_win_sec': iDeadWin,
                             'feature_win_sec': iFeat,
                             'refractory_sec': refractory_sec,
                             'neg_train_smp': neg_train_smp,
                             'dead_win_smp': dead_win_smp,
                             'feature_win_smp': feature_win_smp,
                             'refractory_smp': refractory_smp,
                             'features_n': features_n,
                             'template': template}
                    
                    blocks_len = len(td_decoder)
                    
                    # Collect signals for features in one 2darray
                    for td_tmp in td_decoder:
                        signal_features = np.array([]).reshape(td_tmp[td_tmp['signal_names_spmt'][0]].shape[1],0)
                        for signal in td_tmp['signal_names_spmt']:
                            signal_features = np.hstack([signal_features, td_tmp[signal].T])
                        td_tmp['signal_features'] = signal_features
                            
                    # Set no-events samples
                    no_events_idx = np.array([]).reshape(0,2)
                    # Loop over the blocks
                    for iTd, td_tmp in enumerate(td_decoder):      
                        bad_idx = []
                        # Collect events features
                        for event_type in td_tmp['gait_events_name_spmt']:
                            # print(len(event[event_type]))
                            for ev in td_tmp[event_type]:
                                bad_idx.append(np.arange(ev-dead_win_smp,ev+dead_win_smp+1).reshape(2*dead_win_smp+1,))
                        
                        bad_idx = np.concatenate(bad_idx)
                        no_events_tmp = np.setdiff1d(np.arange(0,signal_features.shape[0]), np.squeeze(bad_idx.astype('int')))
                        no_events_idx = np.vstack([no_events_idx, np.vstack([no_events_tmp, iTd*np.ones((len(no_events_tmp),)) ]).T  ])
                     
                    if no_events_idx.shape[0] > neg_train_smp:
                        rdm_idx = np.random.permutation(no_events_idx.shape[0])
                        no_events_idx = no_events_idx[ rdm_idx[np.arange(neg_train_smp)], :].astype('int')
                    else:
                        neg_train_smp = no_events_idx.shape[0]
                        params['neg_train_smp'] = neg_train_smp
                    
                    # Collect features
                    features = np.array([]).reshape(0,features_n)
                    labels = np.array([]).reshape(0,1)
                    features_name_spmt = []
                    for event_type in td_decoder[0]['gait_events_name_spmt']:
                        model['features_{}'.format(event_type)] = []
                        features_name_spmt.append('features_{}'.format(event_type))
                    
                    model['features_no_event'] = []
                    features_name_spmt.append('features_no_event')
                    
                    # Loop over the blocks
                    for iTd, td_tmp in enumerate(td_decoder):    
                        print('Collecting features from block {}/{}'.format(iTd+1, blocks_len))
                        # Collect events features
                        
                        for iEv, event_type in enumerate(td_tmp['gait_events_name_spmt']):
                            features_tmp, _ = extract_features(td_tmp[event_type],td_tmp['signal_features'],template)
                            model['features_{}'.format(event_type)].append(features_tmp)
                            
                            if iTd != blocks_n-1:
                                features = np.vstack([features, features_tmp])
                                labels = np.vstack([ labels, (iEv+1)*np.ones((features_tmp.shape[0],1)) ])
                        
                        no_events_idx_tmp = np.where(no_events_idx[:,1] == iTd)[0]
                        if no_events_idx_tmp.size != 0:
                            features_tmp, _ = extract_features(no_events_idx[no_events_idx_tmp,0],td_tmp['signal_features'],template)
                            model['features_no_event'].append(features_tmp)
                            
                            if iTd != blocks_n-1:
                                features = np.vstack([features, features_tmp])
                                labels = np.vstack([ labels, np.zeros((features_tmp.shape[0],1)) ])
                        
                    model['features_name'] = features_name_spmt
                       
                    print(' ')
                    for feat_name in model['features_name']:
                        for iBk, feat in enumerate(model[feat_name]):
                            print('Block {}. Event {}: #{}'.format(iBk+1, feat_name, feat.shape[0]))
                         
                    model['features'] = features
                    model['labels'] = np.squeeze(labels)
                    model['params'] = params
                    
                    # Create a decoder
                    print(' ')
                    print('CREATING DECODER...')
                    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                    
                    train_X = model['features']
                    train_y = model['labels']
                    regularization_coeff = model['params']['regularization_coeff']
                    clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
                    # clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage = regularization_coeff, store_covariance=True)
                    # clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage = 'auto', store_covariance=True)
                    clf.fit(train_X, train_y)
                    
                    # Store classifier
                    model['clf'] = clf
                    
                    print('MODEL BUILT!')
                    
                    # Append model
                    models.append(model)
                    # Increment counter
                    counter += 1
  
#%% Plot features
if plot_data:
    import matplotlib.pyplot as plt
    from td_process import event_color, event_linestyle
    from sklearn.manifold import TSNE
    
    X = []
    y = []
    X_feat = {}
    for iFeat, feat_name in enumerate(model['features_name']):
        X_feat[feat_name] = []
        for feat in model[feat_name]:
            X_feat[feat_name].append(feat)
            X.append(feat)
            y.append(iFeat * np.ones(feat.shape[0]) )
        
        X_feat[feat_name] = np.concatenate(X_feat[feat_name])
    
    X = np.concatenate(X)
    y = np.concatenate(y)
    
    # TSNE
    # X_embedded = TSNE(n_components=3, init='pca', random_state=0).fit_transform(X)
    # X_embedded.shape
    # ax = fig.add_subplot(111)
    # ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:,2], c=y, cmap=plt.cm.Spectral)
    # ax.set_title('tSNE')
    # ax.axis('tight')
    
    fig, axs = plt.subplots(nrows = len(model['features_name']), ncols = 1)
    for iEv, feat_name in enumerate(model['features_name']):
        axs[iEv] = plt.plot(X_feat[feat_name].T, '.')
    
    
    # signal_to_plot = ['LFP_BIP7_ampl_13_20','LFP_BIP7_ampl_20_30']
    # events_to_plot = ['RFS']
    # subplot = (len(signal_to_plot),1)
        
    # signal_idx = [idx for idx, name in enumerate(td_tmp['signal_names_ampl']) if name in signal_to_plot]

    # for td_tmp in td_decoder:    
    #     fig, axs = plt.subplots(nrows = subplot[0], ncols = subplot[1])
    #     fig.suptitle('File {}'.format(td_tmp['file_names']), fontsize=10)
        
    #     for idx, ax in enumerate(np.ndarray.tolist(axs.reshape(len(signal_to_plot),1))):
            
    #         ax[0].set_title(signal_to_plot[idx])
            
    #         if idx % 2 == 0:
    #             ax[0].set_ylabel('mV')
            
    #         if idx+1 % 5 != 0:
    #             ax[0].tick_params(
    #                 axis='x',          # changes apply to the x-axis
    #                 which='both',      # both major and minor ticks are affected
    #                 bottom=False,      # ticks along the bottom edge are off
    #                 top=False,         # ticks along the top edge are off
    #                 labelbottom=False) # labels along the bottom edge are off
                            
    #         ax[0].set_xlabel('')
    #         ax[0].grid(True)
    #         ax[0].set_ylim((-50,50))
    #         ax[0].plot(td_tmp[signal_to_plot[idx]])
            
    #         for event in events_to_plot:
    #             if 'R' in event:
    #                 line_style = event_linestyle.R.value
    #             else:
    #                 line_style = event_linestyle.L.value
                    
    #             if 'FS' in event:
    #                 col = event_color.FS.value
    #             else:
    #                 col = event_color.FO.value
                    
    #             for iEv, ev in enumerate(td_tmp[event]):
    #                 ax[0].axvline(ev,-50, 50, color = col, linestyle = line_style)
    #                 for i in np.arange(10):
    #                     ax[0].scatter(ev+model['params']['template'][i],td_tmp['features_'+event][iEv, len(model['params']['template'])*signal_idx[idx]+i ],color = col)
                  
#%% Test decoder on data
blocks_len = len(td_decoder)

for model in models:
    for iTd, td_tmp in enumerate(td_decoder):
        print('Computing probabilities on block {}/{}'.format(iTd+1, blocks_len))
        _, test_X = extract_features(np.arange(0,td_tmp['signal_features'].shape[0]),td_tmp['signal_features'],model['params']['template'])
        td_tmp['decoding_prob'] = model['clf'].predict_proba(test_X)

#%% Plot data
import matplotlib.pyplot as plt
from td_process import event_color, event_linestyle

for td_tmp in td_decoder:
    fig = plt.figure()
    # plt.plot(td_tmp['decoding_prob'][:,0], color = 'b')
    plt.plot(td_tmp['decoding_prob'][:,1], color = 'r')
    # plt.plot(td_tmp['decoding_prob'][:,2], color = 'c')
    for event in td_tmp['gait_events_name_spmt']:
        if 'R' in event:
            col = event_color.FS.value
        else:
            col = event_color.FO.value      
            
        for ev in td_tmp[event]:
            plt.plot(ev,1.1, marker='v', color = col)

#%% Separation
            
#%% Save
import pickle

pickle_out = open('/Users/raschell/Desktop/test/td.pickle','wb')
pickle.dump(td, pickle_out)
pickle_out.close()

#%% Load
import pickle

pickle_in = open('/Users/raschell/Desktop/test/td.pickle',"rb")
td = pickle.load(pickle_in)

#%% Random plot 

fig = plt.figure()
plt.plot(td_decoder['signals'][0][1,:], color = 'b')
plt.plot(td_decoder['high_beta'][0][1,:], color = 'r', linewidth=1)
plt.plot(td_decoder['high_beta_amp'][0][1,:], color = 'c', linewidth=2)
for event in events_label:
    if 'R' in event:
        line_style = event_linestyle.R.value
    else:
        line_style = event_linestyle.L.value
        
    if 'FS' in event:
        col = event_color.FS.value
    else:
        col = event_color.FO.value
        
    for ev in gait_events[0][event]:
        plt.axvline(ev,-50, 50, color = col, linestyle = line_style)


data = signal[2,:]
fig = plt.figure()
plt.psd(data, int(10*signal_FS), signal_FS)
plt.ylim((-20,20))
plt.xlim((-10,500))

fig = plt.figure()
f, Pxx_den = welch(data, signal_FS, nperseg=signal_FS)
plt.semilogy(f, Pxx_den)
plt.ylim([0.5e-3, 1])
plt.xlim((-10,500))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


