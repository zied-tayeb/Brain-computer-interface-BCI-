import os
import sys
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import eeg_clf_test_snn
import eeg_clf_train_snn


def multiply_weights():
    # w: np.array containing the weights
    n_cl = 2
    for i in range(n_cl):
        cl_weights = np.load("output_files/stdp_weights{}.npy".format(i))
        cl_weights *= 0.9
        np.save("output_files/stdp_weights{}.npy".format(i), cl_weights)


def PCA_dim_red(X, var_desired):
    """
    Dimensionality reduction using PCA
    X:            matrix (2d np.array)
    var_desired:  desired preserved variance

    Returns X with reduced dimesnions
    """
    # PCA
    pca = PCA(n_components=X.shape[1]-1)
    pca.fit(X)
    print('pca.explained_variance_ratio_:\n',pca.explained_variance_ratio_)    
    var_sum = pca.explained_variance_ratio_.sum()
    var = 0
    for n, v in enumerate(pca.explained_variance_ratio_):
        var += v
        if var / var_sum >= var_desired:
            X_reduced = PCA(n_components=n+1).fit_transform(X)
            print("Reached Variance: {:1.3f} at {}-Dimensions. New shape: {}"
                  .format(var/var_sum, n+1, X_reduced.shape))
            return X_reduced


# --- INITIALISATION -----------------------------------------------------------

# Check number of cycles
try:
    n_cycles = int(sys.argv[1])
except:
    n_cycles = 1    
print("\n {} Cycle(s)! \n".format(n_cycles))


# --- PARAMETER ---------------------------------------------------------------

# Simulation Parameter          BEST RESULTS => mean: 0.67, std.dev.: 0.058
homeostasis       = False         # True
epochs            = 1             # 1
# Network Parameter         
trial_num         = 90            # 90
randomness        = True          # True
rand_data         = False         # False
reverse_src_del   = False         # False
# STDP Parameter            
tau_pl            = 1#0.02            # 10.


# --- PREPARE DATA -------------------------------------------------------------

# Data Path
data_dir          = "data/X_train_zied.npy"
cls_dir           = "data/y_train_zied.npy"
#path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/B01T/'
#data_dir = path + 'B01T_0_X_sub_cte .npy'
#cls_dir  = path + 'B01T_0_y_sub_cte .npy'

# Data preprocessing
X = np.load(data_dir)
Y = np.load(cls_dir)

X = PCA_dim_red(X, var_desired=0.95)

# --- SIMULATION ---------------------------------------------------------------

### Run Train and Test Scripts n_cycles times
print('Training network ... ')
rates = []

##
##  Normal Training
##
for c in range(n_cycles):
    eeg_clf_train_snn.train_snn(n_training      = 2,  # 4 
                                use_old_weights = False,
                                randomness      = randomness,
                                tau_pl          = tau_pl,
                                rand_data       = rand_data,
                                trial_num       = trial_num,
                                reverse_src_del = reverse_src_del,
                                data_dir        = data_dir,
                                cls_dir         = cls_dir,
                                data            = X, #"load", 
                                cls             = Y) #load")
    for _ in range(epochs - 1):
        if homeostasis == True:
            multiply_weights()
        eeg_clf_train_snn.train_snn(n_training      = 1,
                                    use_old_weights = True,
                                    randomness      = randomness,
                                    tau_pl          = tau_pl,
                                    rand_data       = rand_data,
                                    trial_num       = trial_num,
                                    reverse_src_del = reverse_src_del,
                                    data_dir        = data_dir,
                                    cls_dir         = cls_dir,
                                    data            = "load", 
                                    cls             = "load")
    # Wait 2 sec
    t_sleep = 1    # in seconds
    print('Waiting {} seconds ... '.format(t_sleep))
    time.sleep(t_sleep)
    ### Test network
    print('Testing network ... ')
    rates.append(eeg_clf_test_snn.test_snn(randomness=True))
    print("Cycle {}, CLF Rate = {}".format(c+1, rates[-1]))

    # stop training if clf rate is high enough
    rate_tresh = 0.7
    #if rates[-1] > rate_tresh:
    #   break


# Save clf rates
np.save("output_files/clf_rates.npy", rates)

# Print results
if n_cycles > 1:
    print("Results ({} cylces): Mean CLF-Rate = {}, with std.dev = {}".format(
        c+1, np.mean(rates), np.std(rates)))

