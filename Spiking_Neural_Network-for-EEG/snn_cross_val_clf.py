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


def cross_val_data_split(X, y, k, shuffle_data=True):
    """ 
    Split data corresponding to k-fold cross validation
    """
    idx_arr = np.arange(len(y))
    if shuffle_data == True:
        np.random.shuffle(idx_arr)
    idx_arr_split = np.array_split(idx_arr, k)
    X_i_list = []
    y_i_list = []
    for idx_part in idx_arr_split:
        X_i_list.append(X[idx_part])
        y_i_list.append(y[idx_part])
    return X_i_list, y_i_list


def cross_val_train_set(X_i_list, y_i_list, idx):
    """
    Create training set from X and y lists without batch #idx
    """
    # Init training arrays
    if idx == 0:
        inserted = 1
        X_train = X_i_list[1]
        y_train = y_i_list[1]
    else:
        inserted = 0
        X_train = X_i_list[0]
        y_train = y_i_list[0]
    # Add remaining data to training set
    for i in range(len(X_i_list)):
        if i != idx and i != inserted:
            X_train = np.concatenate((X_train, X_i_list[i]), axis=0)
            y_train = np.concatenate((y_train, y_i_list[i]), axis=0)
    return X_train, y_train


def get_clf_results(y, y_):
    """
    Evaluate prediction results of CLF
    """
    y = y.flatten()
    n_mclf = float(len(np.nonzero(y - y_)[0]))
    clf_rate = 1. - n_mclf / len(y)
    n_total = float(len(y))
    return n_mclf, n_total, clf_rate
    

def test_clf(X_train, y_train, X_test, y_test, clf):
    y_train = y_train.flatten()
    # Random Forest
    if clf == 'rf':
        clf = RandomForestClassifier(criterion='gini', n_estimators=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    # Logistic Regression
    elif clf == 'logr':
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    # Support Vector Machine
    elif clf == 'svm':
        clf = svm.SVC()
        clf.fit(X_train, y_train)  
        y_pred = clf.predict(X_test)

    elif clf == 'snn':
        # SNN Parameter
        n_training      = 2    # how often the training samples are presented
        rand_data       = False
        trial_num       = 10    # numer of trials (samples) presented 
                                # (only if rand_data = True)
        reverse_src_del = False
        n_feature = 20 * X_train.shape[1]
        # Train
        eeg_clf_train_snn.train_snn(data            = X_train,
                                    cls             = y_train,
                                    n_training      = n_training,  # 4 
                                    randomness      = True,
                                    tau_pl          = 10.,
                                    rand_data       = rand_data,
                                    trial_num       = trial_num,
                                    reverse_src_del = reverse_src_del,
                                    n_feature       = n_feature)
        # Test
        clf_rate = eeg_clf_test_snn.test_snn(data       = X_test,
                                             cls        = y_test,
                                             randomness = True)
        y_pred = clf_rate

    # Print data sets
    else:
        print('X_train\n', X_train.shape)
        print('y_train\n', y_train.shape, '\n')
        print('X_test\n', X_test.shape)
        print('y_test\n', y_test.shape, '\n')
        y_pred = 0

    return y_pred


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - Main                                                                      -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
t0 = time.time()

# - Data  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if 0: # B01T subbands (zied/emec)
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/eeg_data_zied_emec/'
    #path = '/Users/leonardrychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/Data/eeg/eeg_data_zied_emec/'
    X_fname = 'X_matrix.npy'
    y_fname = 'y_vector.npy'
if 0: # B01T ERP
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/B01T/'
    X_fname = 'b01t_0_erp_X_set.npy'
    y_fname = 'b01t_0_erp_y_set.npy'
if 1: # B01T Subbands with CTE
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/B01T/'
    X_fname  = 'B01T_0_X_sub_cte.npy'
    y_fname  = 'B01T_0_y_sub_cte.npy'
if 0: # B02T ERP
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/EEG/B02T/'
    X_fname = 'b02t_0_erp_X_set.npy'
    y_fname = 'b02t_0_erp_y_set.npy'
if 0: # iris data set
    path = '/home/rychly/Dropbox/[TUM]/4. WiSe 1617/Masterarbeit/data/iris/'
    X_fname = 'iris_X.npy'
    y_fname = 'iris_y.npy'



# - Parameters - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Select classifier to test         
classifier_type = 'snn'       
                            #       'svm':  Support Vector Machine
                            #       'logr': Logistic Regression
                            #       'rf':   Random Forest
                            #       'snn':  Spiking Neural Network
print('\n> Using {} <\n'.format(classifier_type))

# k for k-fold cross-validation
corss_validation = True
k = 4   # > 1 (using 1/k of total data as test set)

# PCA Dimensionality reduction
var_desired = .93    # min variance preserved with PCA

# Shuffle data befor splitting in batches
shuffle_data = True


# - Load data - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - -

X = np.load(path + X_fname)
y = np.load(path + y_fname)


# - Preprocessing data  - - - - - -  - - - - - - - - - - - - - - - - - - - - - - 

# Shift to zero mean and normalize y
y = y-1
# Normalize X 
X = normalize(X)
#X = StandardScaler().fit_transform(X)    # ! 0.1 worse results than normalize()
# Reduce dimensions
X = PCA_dim_red(X, var_desired)
# Get k data batches
#X = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
#y = np.array([1,2,3,4])
if corss_validation ==  True:
    X_i_list, y_i_list = cross_val_data_split(X, y, k, shuffle_data)


# - Test CLF  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

results = np.zeros((k, 3))    # number cols = number returns of results()-fct
rates = []    # only used for snn
for i, (X_test, y_test) in enumerate(zip(X_i_list, y_i_list)):
    X_train, y_train = cross_val_train_set(X_i_list, y_i_list, i)
    # Train and test CLF
    y_pred = test_clf(X_train, y_train, X_test, y_test, classifier_type)
    if classifier_type == 'snn':
        results[i,0] = y_pred
        print('Cross-Val. Nr.{}, Test CLF Rate = {:.3f}'.format(i, y_pred))
    else:
        results[i] = get_clf_results(y_test, y_pred)
        print('Cross-Val. Nr.{}, Test CLF Rate = {:.3f}'.format(i, results[i,2]))


# Print overall results
# (y_pred = clf_rate for SNN!)
if classifier_type == 'snn':
    print('\nMean CLF Rate = {:.3}'.format(results[:,0].mean()))    #
else:
    print('\nMean CLF Rate = {:.3}\nMisCLF = {} from {} total samples.'.format(
        results[:,2].mean(), results[:,0].sum(), results[:,1].sum()))


print('\nCalculation took {} seconds.\n'.format(time.time() - t0))




