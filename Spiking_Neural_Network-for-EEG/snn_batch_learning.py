import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import eeg_clf_test_snn
import eeg_clf_train_snn
import pylab
import os

# - FUNCTIONS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def PCA_dim_red(X, var_desired=0.95):
    """
    Dimensionality reduction using PCA
    X:            matrix (2d np.array)
    var_desired:  desired preserved variance

    Returns X with reduced dimesnions
    """
    # PCA
    pca = PCA(n_components=X.shape[1]-1)
    pca.fit(X)
    var_sum = pca.explained_variance_ratio_.sum()
    var = 0
    for n, v in enumerate(pca.explained_variance_ratio_):
        var += v
        if var / var_sum >= var_desired:
            print("\nReached Variance: {:1.3f}, new dim = {}\n".format(
                var/var_sum, n+1))
            return PCA(n_components=n+1).fit_transform(X)


def split_into_batch(X, y, n, shuffle_data=True):
    """
    Split data into n batch
    X:  feature matrix
    y:  label vector
    n:  number batches
    """
    idx_arr = np.arange(len(y))
    if shuffle_data == True:
        np.random.shuffle(idx_arr)
    idx_arr_split = np.array_split(idx_arr, n)
    X_i_list = []
    y_i_list = []
    for idx_part in idx_arr_split:
        X_i_list.append(X[idx_part])
        y_i_list.append(y[idx_part])
    return X_i_list, y_i_list


def make_binary_weights():
    n_cl = 2
    w_max = .4
    w_min = .0
    # read in weights
    weights_cl_0 = np.load("output_files/stdp_weights{}.npy".format(0))
    weights_cl_1 = np.load("output_files/stdp_weights{}.npy".format(1))
    # mean difference of weights
    delta_weights = weights_cl_0 - weights_cl_1
    mean_delta_w = abs(delta_weights).mean()

    for i, (w_0, w_1) in enumerate(zip(weights_cl_0, weights_cl_1)):
    #    if w_0 > w_1:    #w_0 - w_1 > mean_delta_w
    #        weights_cl_0[i] = w_max
    #        weights_cl_1[i] = w_min
    #    elif w_0 < w_1:   #w_0 - w_1 < -mean_delta_w
    #        weights_cl_1[i] = w_max
    #        weights_cl_0[i] = w_min
    #    else:
    #        weights_cl_1[i] = w_min
    #        weights_cl_0[i] = w_min

        if w_0 - w_1 > mean_delta_w and w_1 > 0.01:
            weights_cl_0[i] = w_max
            weights_cl_1[i] = w_min
        elif w_0 - w_1 < -mean_delta_w and w_0 > 0.01:
            weights_cl_1[i] = w_max
            weights_cl_0[i] = w_min
        else:
            weights_cl_1[i] = w_min
            weights_cl_0[i] = w_min

    np.save("output_files/stdp_weights{}.npy".format(0), weights_cl_0)
    np.save("output_files/stdp_weights{}.npy".format(1), weights_cl_1)




    # Plot weights
    if 1:
        pylab.figure()
        pylab.xlabel('Weight ID')
        pylab.ylabel('Weight Value')
        pylab.title('STDP weights at the end (after binarization)')
        pylab.hold(True)
        pylab.plot(weights_cl_0)
        pylab.plot(weights_cl_1)
        # save fig
        fname = 'plots/binaryweights_1.png'
        while True:
            if os.path.isfile(fname):    # if file already exists
                new_num =  int(fname.split('.')[0].split('_')[1]) + 1
                fname = fname.split('_')[0] + '_' +str(new_num) + '.png'
            else:
                pylab.savefig(fname)
                break




def train_clf(X_train, y_train, batch_n):
    """
    batch_n:    actual batch number that is just trained with
    """
    y_train = y_train.flatten()
    # SNN Parameter
    n_training      = 1    # how often training are samples presented / cycle

    n_feature = 20 * X_train.shape[1]

    if batch_n == 0:
        old_weights = False
    else:
        old_weights = True

    # Train
    eeg_clf_train_snn.train_snn(data            = X_train,
                                cls             = y_train,
                                use_old_weights = old_weights,
                                n_training      = n_training,
                                randomness      = True,
                                tau_pl          = .2,
                                n_feature       = n_feature)



def test_clf(X_test, y_test):
    y_test = y_test.flatten()
    eeg_clf_test_snn.test_snn(data            = X_test,
                              cls             = y_test,
                              randomness      = True)



# - Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Binary weights
binary_weights = True

# Number of splits (n batch)
n_batch = 10

# PCA Dimensionality reduction
var_desired = .85    # min variance preserved with PCA

# Shuffle data befor splitting in batches
shuffle_data = True

t0 = time.time()
# - Load data - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - -
if 1: # B01T Subbands with CTE
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/B01T/'
    #path = '/Users/leonardrychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/Data/eeg/B01T/'
    X_fname  = 'B01T_0_X_sub_cte.npy'
    y_fname  = 'B01T_0_y_sub_cte.npy'
    val_X_fname = 'B01T_0_X_sub_set.npy'
    val_y_fname = 'B01T_0_y_sub_set.npy'
    
X = np.load(path + X_fname)
y = np.load(path + y_fname)
X_val = np.load(path + val_X_fname)
y_val = np.load(path + val_y_fname).flatten()


# - Preprocessing data  - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# Shift to zero mean and normalize y
if set(y) == set([1,2]):
    y = y - 1
if set(y_val) == set([1,2]):
    y_val = y_val - 1
# Normalize X 
X = normalize(X)
# Reduce dimensions
X = PCA_dim_red(X, var_desired)
# Get k data batches
#X = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
#y = np.array([1,2,3,4])
X_i_list, y_i_list = split_into_batch(X, y, n_batch, shuffle_data)


# - Train/Test CLF  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

rates = []    # only used for snn

# Train SNN
for i, (X_train, y_train) in enumerate(zip(X_i_list, y_i_list)):
    train_clf(X_train, y_train, batch_n=i)

# Create binary weights
if binary_weights == True:
    make_binary_weights()

# Test
clf_rate = test_clf(X_val, y_val)
print('Cross-Val. Nr.{}, Test CLF Rate = {}'.format(i, clf_rate))

print('\nCalculation took {} seconds.\n'.format(time.time() - t0))














