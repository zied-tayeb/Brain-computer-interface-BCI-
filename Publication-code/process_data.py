import sys
import numpy as np
import scipy.io
import pandas as pd
from scipy.signal import butter, iirnotch, lfilter
from sklearn.preprocessing import normalize, StandardScaler


"""
The EEG trials returnd by this function have the following format:

    ===========================================> time
        |                                   |
    trial_start                         trial_end
        |<------------trial_len------------>|
                        |<---MotorImager--->|

returns:    raw_eeg     (n_samples, n_channels)  return all channels
            trials      (,n_trials)
            labels      (,n_labels)
            trial_len   scalar
            fs          scalar
            mi_interval [mi_start, mi_end] within the trial in seconds

"""

def read_data(subject_str):
    """ Input:  (str) that defines what data to load
        Output: (np.array) of size (n_samples, n_channels)
    """
    # Init variables
    data_type = ""
    file_dir = ""
    raw_data = np.zeros(1)
    labels = np.zeros(1)
    trials = np.zeros(1)
    mi_interval = []
    freq_s = 0
    trial_t = 0

    folder_dir = "/Users/leonardrychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/Data/eeg/"

    if subject_str == "NST Alona": # Without Feed Back
        file_dir1 = "NST-datasets/Alona-data/s1.mat"
        file_dir2 = "NST-datasets/Alona-data/s2.mat"
        file_dir3 = "NST-datasets/Alona-data/s3.mat"
        data_type = 'nst'
    elif subject_str == "NST Christoph": # Without Feed Back
        file_dir1 = "NST-datasets/Christoph-data/s1.mat"
        file_dir2 = "NST-datasets/Christoph-data/s2.mat"
        file_dir3 = "NST-datasets/Christoph-data/s3.mat"
        data_type = 'nst'
    elif subject_str == "NST Emec": # Without Feed Back
        file_dir1 = "NST-datasets/Emec-data/s1.mat"
        file_dir2 = "NST-datasets/Emec-data/s2.mat"
        file_dir3 = "NST-datasets/Emec-data/s3.mat"
        data_type = 'nst'
    elif subject_str == 'NST Leonard': # Without Feed Back
        file_dir1 = "NST-datasets/Leonard-data/s1.mat"
        file_dir2 = "NST-datasets/Leonard-data/s2.mat"
        file_dir3 = "NST-datasets/Leonard-data/s3.mat"
        data_type = 'nst'
    elif subject_str == "NST Lukas": # Without Feed Back
        file_dir1 = "NST-datasets/Lukas-data/s1.mat"
        file_dir2 = "NST-datasets/Lukas-data/s2.mat"
        file_dir3 = "NST-datasets/Lukas-data/s3.mat"
        data_type = 'nst'
    elif subject_str == "Graz B1":      # Without Feed Back
        file_dir1 = "Graz_data_B/B01T.mat"
        file_dir2 = "Graz_data_B/B01E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B2":      # Without Feed Back
        file_dir1 = "Graz_data_B/B02T.mat"
        file_dir2 = "Graz_data_B/B02E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B3":      # With Feed Back
        file_dir1 = "Graz_data_B/B03T.mat"
        file_dir2 = "Graz_data_B/B03E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B4":      # With Feed Back
        file_dir1 = "Graz_data_B/B04T.mat"
        file_dir2 = "Graz_data_B/B04E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B5":      # With Feed Back
        file_dir1 = "Graz_data_B/B05T.mat"
        file_dir2 = "Graz_data_B/B05E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B6":      # With Feed Back
        file_dir1 = "Graz_data_B/B06T.mat"
        file_dir2 = "Graz_data_B/B06E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B7":      # With Feed Back
        file_dir1 = "Graz_data_B/B07T.mat"
        file_dir2 = "Graz_data_B/B07E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B8":      # With Feed Back
        file_dir1 = "Graz_data_B/B08T.mat"
        file_dir2 = "Graz_data_B/B08E.mat"
        data_type = 'graz'
    elif subject_str == "Graz B9":      # With Feed Back
        file_dir1 = "Graz_data_B/B09T.mat"
        file_dir2 = "Graz_data_B/B09E.mat"
        data_type = 'graz'
    
    else:
        print("No such subject available! Task Stops!\n")
        sys.exit()

    # Read NST data
    if data_type == 'nst':
        # Prameter
        trial_len = 6   # sec (length of a trial after trial_sample)
        trial_offset = 5    # idle period prior to trial start [sec]
        trial_total = trial_offset + trial_len    # total length of trial
        mi_interval = [trial_offset, trial_offset+trial_len]    # interval of motor imagery within trial_t [sec]
        # Read data
        mat1 = scipy.io.loadmat(folder_dir + file_dir1) 
        mat2 = scipy.io.loadmat(folder_dir + file_dir2) 
        mat3 = scipy.io.loadmat(folder_dir + file_dir3)
        freq_s = mat1['Fs'].flatten()[0]
        raw_data1 = mat1['X'][:,1:]
        raw_data2 = mat2['X'][:,1:]
        raw_data3 = mat3['X'][:,1:]
        labels1 = mat1['Y'].flatten() - 1
        labels2 = mat2['Y'].flatten() - 1
        labels3 = mat3['Y'].flatten() - 1
        trials1 = mat1['trial'].flatten() - freq_s*trial_offset
        trials2 = mat2['trial'].flatten() - freq_s*trial_offset
        trials3 = mat3['trial'].flatten() - freq_s*trial_offset
        trials2 += raw_data1.T.shape[1]
        trials3 += raw_data1.T.shape[1] + raw_data2.T.shape[1]
        #combine matrices together 
        raw_data = np.concatenate((raw_data1, raw_data2, raw_data3))
        labels = np.concatenate((labels1,labels2,labels3))
        trials = np.concatenate((trials1, trials2, trials3))
        # Remove class 3
        c3_idxs = np.where(labels==2)[0]
        labels = np.delete(labels, c3_idxs)
        trials = np.delete(trials, c3_idxs)
        

    if data_type == "graz":
        # Parameter
        freq_s = 250    # Hz (Sampling Frequency)
        trial_len = 8   # length of trial after trial_idx [sec] 
        mi_interval = [4,7]    # interval of motor imagery within trial_t [sec]
        trial_offset = 0     # idle period prior to trial start [sec]
        trial_total = trial_len    # total length of trial
        # read data
        mat1 = scipy.io.loadmat(folder_dir + file_dir1)['data']    
        mat2 = scipy.io.loadmat(folder_dir + file_dir2)['data']    
        # dict_keys(['__header__', '__globals__', '__version__', 'data'])
        
        # Load Test Data
        data_bt = []
        labels_bt = []
        trials_bt = []
        n_experiments = 3
        for i in range(n_experiments):
            data      = mat1[0,i][0][0][0]
            trials    = mat1[0,i][0][0][1]
            labels    = mat1[0,i][0][0][2] - 1
            fs        = mat1[0,i][0][0][3].flatten()[0]
            if fs != freq_s:
                print("ERROR: Sampling Frequencies don't concide!")
            artifacts = mat1[0,i][0][0][5]
            # remove artivacts
            artifact_idxs = np.where(artifacts == 1)[0]
            trials = np.delete(trials, artifact_idxs)
            labels = np.delete(labels, artifact_idxs)
            # add data to files
            data_bt.append(data)
            labels_bt.append(labels)
            trials_bt.append(trials)
        # add length of previous data set to adjust trial start points
        trials_bt[1] += data_bt[0].shape[0]
        trials_bt[2] += data_bt[0].shape[0] + data_bt[1].shape[0]
        # concatenate all data mat, trials, and labels
        data_bt = np.concatenate((data_bt[0], data_bt[1], data_bt[2]))
        trials_bt = np.concatenate((trials_bt[0], trials_bt[1], trials_bt[2]))
        labels_bt = np.concatenate((labels_bt[0], labels_bt[1], labels_bt[2]))

        ## Load Evaluation Data
        #data_be = mat2[0][0][0][0][0]
        #trials_be = mat2[0][0][0][0][1]
        #labels_be = mat2[0][0][0][0][2] - 1
        #artifacts_be = mat2[0][0][0][0][5]
        ## remove artefacts
        #artifact_be_idxs = np.where(artifacts_be == 1)[0]
        #trials_be = np.delete(trials_be, artifact_be_idxs)
        #labels_be = np.delete(labels_be, artifact_be_idxs)

        # Interpolate NaN values
        #raw_data = pd.DataFrame(raw_data).interpolate().values

        raw_data = data_bt[:,:3]
        trials = trials_bt
        labels = labels_bt
 
    return (raw_data, trials, labels, trial_total, freq_s, mi_interval, 
            subject_str)





def extract_trials(raw_data, trials, labels, trial_total, fs):
    """
    raw_data:       Raw EEG data                (n_samples,n_channels)
    trials:         Starting sample of a trial  (n_trials,)
    labels:         Corresponding label         (n_labels,)
    trial_total:    Total length of trial [sec] scalar
    fs:             Sampling frequency in [Hz]  scalar
    """
    # get class indecis
    class1_idxs = np.where(labels == 0)[0]
    class2_idxs = np.where(labels == 1)[0]
    # init data lists for each class 
    #                     (n_trials,          n_samples,      n_channels        )
    class1_data = np.zeros((len(class1_idxs), trial_total*fs, raw_data.shape[1]))
    class2_data = np.zeros((len(class2_idxs), trial_total*fs, raw_data.shape[1]))

    # split data class 1
    for i, c1_idx in enumerate(class1_idxs):    # iterate over trials
        trial = raw_data[trials[c1_idx] : trials[c1_idx]+trial_total*fs]    # (n_samples, n_channels)
        class1_data[i,:,:] = trial
    # split data class 2
    for i, c2_idx in enumerate(class2_idxs):    # iterate over trials
        trial = raw_data[trials[c2_idx] : trials[c2_idx]+trial_total*fs]    # (n_samples, n_channels)
        class2_data[i,:,:] = trial

    return class1_data, class2_data


def bandpass_filter(data, lowcut, highcut, fs, order=6):
    """
    data:   (n_samples, n_channels)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    try:    
        raw_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):    # iterate over number of channels
            raw_filt[:,i] = lfilter(b, a, data[:,i]) 
        return raw_filt
    except:    # if 1D array
        return lfilter(b, a, data)
        

def bandstop_filter(data, w0, Q):
    """
    data:   (n_samples, n_channels)
    """
    b, a = iirnotch(w0, Q)
    try:
        raw_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):    # iterate over number of channels
            raw_filt[:,i] = lfilter(b, a, data[:,i])
        return raw_filt
    except: # if 1D array
        return lfilter(b, a, data)
     

def normalize_data(data, type="norm"):
    if type == "norm":
        return normalize(data)
    elif type == "std":
        return StandardScaler().fit_transform(data) 
    elif type == "nan":
        mean = np.nanmean(data)
        std = np.nanstd(data)
        return (data - mean) / std


def subBP_features(raw_data, trials, labels, fs):
    """
    Read:
    - raw_data:   (n_samples, n_channels)
    - trials:     (n_trials, )
    - labels:     (n_trials, )
    Return:  
    - X:          (n_trials, n_features)
    - Y:          (n_trials)
    """

    ### Filter data in sub-bands
    sub_bands = np.array([np.arange(7,12), np.arange(8,13)]).T
    # array([[ 7,  8],
    #        [ 8,  9],
    #        ...
    #        [23, 24],
    #        [24, 25]])

    filt_subbands = []
    for low, high in sub_bands:
        print('Filter from', low, 'to', high)
        filt_subbands.append(bandpass_filter(data, low, high, fs))


    ### Extract Features

























