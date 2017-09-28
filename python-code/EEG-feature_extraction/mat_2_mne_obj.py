import numpy as np
import scipy.io
import mne


#
# Load Data
#
data_path = "/Users/leonardrychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/Code/eeg_feature_extraction/B01T.mat"
mat = scipy.io.loadmat(data_path)['data']	

# Separate experiments in .mat file
exp_arr = [[] for _ in range(mat.shape[1])]
exp_arr[0] = mat[0,0]
exp_arr[1] = mat[0,1]
exp_arr[2] = mat[0,2]

### Parameters
trial_len = 4    # sec
sample_freq = 250    # Hz


### Data description
# X:		raw eeg data 					- mat(n_timepoints, n_channels)
# trial:	sample points of trial start	- array(n_trials)
# y:		corresponding labels to trials 	- array(n_trials)

# save experiments to dicts
exp_dict = [[] for _ in range(mat.shape[1])]
#								exp_arr[exp_num][0][0][data]
exp_dict[0] = {'X': 			exp_arr[0][0][0][0].T,
			   'trial': 		exp_arr[0][0][0][1],
			   'y': 			exp_arr[0][0][0][2],
			   'fs':			exp_arr[0][0][0][3].flatten(),
			   'classes':		exp_arr[0][0][0][4].flatten(),
			   'artifacts':		exp_arr[0][0][0][5],
			   'gender':		exp_arr[0][0][0][6],
			   'age':			exp_arr[0][0][0][7].flatten()
			   }

exp_dict[1] = {'X': 			exp_arr[1][0][0][0].T,
			   'trial': 		exp_arr[1][0][0][1],
			   'y': 			exp_arr[1][0][0][2],
			   'fs':			exp_arr[1][0][0][3].flatten(),
			   'classes':		exp_arr[1][0][0][4].flatten(),
			   'artifacts':		exp_arr[1][0][0][5],
			   'gender':		exp_arr[1][0][0][6],
			   'age':			exp_arr[1][0][0][7].flatten()
			   }

exp_dict[2] = {'X': 			exp_arr[2][0][0][0].T,
			   'trial': 		exp_arr[2][0][0][1],
			   'y': 			exp_arr[2][0][0][2],
			   'fs':			exp_arr[2][0][0][3].flatten(),
			   'classes':		exp_arr[2][0][0][4].flatten(),
			   'artifacts':		exp_arr[2][0][0][5],
			   'gender':		exp_arr[2][0][0][6],
			   'age':			exp_arr[2][0][0][7].flatten()
			   }


### Remove Artifacts
artifact_idx = exp_dict[0]['artifacts'].nonzero()[0]
# from trials
exp_dict[0]['trial'] = np.delete(exp_dict[0]['trial'], artifact_idx)
exp_dict[0]['y'] = np.delete(exp_dict[0]['y'], artifact_idx)


#
#	Create MNE Objects
#

### MNE Info Object
info = [[] for _ in range(mat.shape[1])]
info[0] = mne.create_info(ch_names	  = ['eeg_1', 'eeg_2', 'eeg_3', 'eog_4', 
									     'eog_5', 'eog_6'],
						  ch_types 	  = ['eeg','eeg','eeg','eog','eog','eog'],
						  sfreq 		  = exp_dict[0]['fs'],
#					      subject_info = {'gender': exp0_dict['gender'],
#					     				  'age':    exp0_dict['age']}
						 )
info[1] = mne.create_info(ch_names	  = ['eeg_1', 'eeg_2', 'eeg_3', 'eog_4', 
										 'eog_5', 'eog_6'],
						  ch_types 	  = ['eeg','eeg','eeg','eog','eog','eog'],
						  sfreq 		  = exp_dict[1]['fs']
#					      subject_info = {'gender': exp1_dict['gender'],
#					   					  'age':    exp1_dict['age']}
					   	 )
info[2] = mne.create_info(ch_names	  = ['eeg_1', 'eeg_2', 'eeg_3', 'eog_4', 
										 'eog_5', 'eog_6'],
						 ch_types 	  = ['eeg','eeg','eeg','eog','eog','eog'],
					     sfreq 		  = exp_dict[2]['fs']
#					     subject_info = {'gender': exp2_dict['gender'],
#					   					 'age':    exp2_dict['age']}
					   	 )


### MNE RawArray Object, shape=(n_channels, n_times)
raw = [[] for _ in range(mat.shape[1])]
raw[0] = mne.io.RawArray(data = exp_dict[0]['X'],
					    info = info[0])
raw[1] = mne.io.RawArray(data = exp_dict[1]['X'],
					    info = info[1])
raw[2] = mne.io.RawArray(data = exp_dict[2]['X'],
					    info = info[2])


### MNE Epoch Array
# Events, shape=(time_point, 0, event_id)
events = [[] for _ in range(mat.shape[1])]
event_ids = np.arange(3)

events[0] = np.zeros((len(exp_dict[0]['trial']), 3)).astype(int)
events[0][:,2] = event_ids[0]     # add event numbers to event mat
events[0][:,0] = exp_dict[0]['trial'].flatten()    # add trials to event mat

events[1] = np.zeros((len(exp_dict[1]['trial']), 3)).astype(int)
events[1][:,2] = event_ids[1]
events[1][:,0] = exp_dict[1]['trial'].flatten()

events[2] = np.zeros((len(exp_dict[2]['trial']), 3)).astype(int)
events[2][:,2] = event_ids[2]
events[2][:,0] = exp_dict[2]['trial'].flatten()

# Epoch Data, shape=(n_epochs, n_channels, n_times)
epochs_data = [[] for _ in range(mat.shape[1])]
epochs_data[0] = np.zeros((len(exp_dict[0]['trial'].flatten()),    # n_epochs
						   exp_dict[0]['X'].shape[0],    # n_channels
						   trial_len * sample_freq))    # n_times
epochs_data[1] = np.zeros((len(exp_dict[1]['trial'].flatten()),
						   exp_dict[1]['X'].shape[0],
						   trial_len * sample_freq))
epochs_data[2] = np.zeros((len(exp_dict[2]['trial'].flatten()),
						   exp_dict[2]['X'].shape[0],
						   trial_len * sample_freq))

# iterate through trial_start_points
for i, trial_start in enumerate(events[0][:,0]):     
	epochs_data[0][i,:,:] = exp_dict[0]['X'].T[trial_start : trial_start
														     +trial_len
														     *sample_freq].T
for i, trial_start in enumerate(events[1][:,0]):     
	epochs_data[1][i,:,:] = exp_dict[1]['X'].T[trial_start : trial_start
															 +trial_len
															 *sample_freq].T
for i, trial_start in enumerate(events[2][:,0]):     
	epochs_data[2][i,:,:] = exp_dict[2]['X'].T[trial_start : trial_start
															 +trial_len
															 *sample_freq].T

# Epochs
epochs = [[] for _ in range(mat.shape[1])]
epochs[0] = mne.EpochsArray(data=epochs_data[0], info=info[0], events=events[0])
epochs[1] = mne.EpochsArray(data=epochs_data[1], info=info[1], events=events[1])
epochs[2] = mne.EpochsArray(data=epochs_data[2], info=info[2], events=events[2])



#
# Plot EEG Signals
#

# Plot with auto-compute scalings
if 0:
	scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
	raw[0].plot(n_channels=4, scalings=scalings, 
			   title='Auto-scaled Data from arrays',
	           show=True, block=True)


#
# Save
#
# raw
if 0:
	raw[0].save('exp_0-raw.fif', overwrite=True)
	raw[1].save('exp_1-raw.fif', overwrite=True)
	raw[2].save('exp_2-raw.fif', overwrite=True)

# epoch_array
if 0:
	epochs[0].save('exp_0-epo.fif', overwrite=True)
	epochs[1].save('exp_1-epo.fif', overwrite=True)
	epochs[2].save('exp_2-epo.fif', overwrite=True)
















