# -*- coding: utf-8 -*-
"""
This code is training code for the conference paper 
Decoding of Motor Imagery Movements from EEG Signals using 
SpiNNaker Neuromorphic Hardware

@author: Emec Ercelik
"""


## Import
import pyNN.spiNNaker as p
import numpy as np
import pandas as pd
import pylab
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def train_snn(### Settings
              data            = "load",
              cls             = "load",
              save            = True,    # True to save all parameters of the network
              randomness      = True,
              reverse_src_del = False,
              use_old_weights = True,
              rand_data       = False,
              ### Parameters
              n_training      = 2,  # How many times the samples will be iterated
              ts              = 1., # Timestep of Spinnaker (ms)
              trial_num       = 10, # Number of features (= 4 features * 20 neurons)
                                #              => 20 neuros: resolution of encoding
              n_feature       = 80,
              # Weights
              wei_src_enc     = .2,    # From Source Array at input to Encoding Layer(Exc)
              wei_enc_filt    = .6,    # From Encoding Layer to Filtering Layer Exc neurons (Exc)
              wei_filt_inh    = 0.03,  # From Filtering Layer Inh neurons to Exc neurons (Inh)
              wei_init_stdp   = .0,    # From Filtering Layer Exc neurons to Output Layer (Exc)
              wei_cls_exc     = 0.9,   # From Output Layer Exc neurons to Inh neurons (Exc)
              wei_cls_inh     = 0.1,#,10   # From Output Layer Inh neurons to Exc neurons (Inh) 
              wei_source_outp = 10.,   # From Source Array at output to Output Layer Exc neurons (Exc)
              wei_noise_poi   = 0.02,
              # Delays
              del_init_stdp   = 1.,
              del_source_outp = 1.,
              del_noise_poi   = 1.,
              # Connection Probabilities
              prob_filt_inh   = .4, # Prob of connectivity inhibitory connections at FilT_Layer
              prob_stdp       = 1., # Prob of STDP connections
              prob_output_inh = .7, # Prob of inhibitory connections at Output Layer
              prob_noise_poi_conn = 0.02,
              ## STDP Parameters
              tau_pl         = 5.,        
              stdp_w_max     = 0.4,           # default 0.4
              stdp_w_min     = 0.0,           # default 0.0
              stdp_A_pl      = 0.02,# 0.01,          # default 0.01 (below 0.01 weights don't change)
                                # => minus in order to get symmetric curve
              # Data Extraction
              scale_data     = 2.): # Scale features into [0-scale_data] range
              
    
    # BUG fix:
    # n_feature is somehow a tuple
    try:
        trial_num = trial_num[0]
    except Exception as e:
        pass

    ############################################################################
    ## Function Definitions
    ############################################################################  
    def gaussian(x, mu, sig):
        return np.float16(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

    def calc_pop_code(feature, rng1, rng2, num):
        interval=np.float(rng2-rng1)/num
        means=np.arange(rng1+interval,rng2+interval,interval)
        pop_code=[gaussian(feature,mu,0.025) for mu in means]
        return pop_code
        
    def PoissonTimes2(t_str=0., t_end=100., rate=10., seed=1.):
        times = [t_str]
        rng = np.random.RandomState(seed=seed)
        cont = True
        while cont == True:
            t_next = np.floor(times[-1] + 1000. * next_spike_times(rng,rate))
            if t_next < t_end - 30:
                times.append(t_next[0])
            else:
                cont=False
                return times[1:]

    def PoissonTimes(t_str=0., t_end=100., rate=10., seed=1.):
        if rate>0:    
            interval = (t_end - t_str + 0.) / rate
            times = np.arange(t_str + 30, t_end - 40, interval)
            return list(times)    
        else:
            return []

    def next_spike_times(rng,rate):
        return -np.log(1.0-rng.rand(1)) / rate   

    def ismember(a, b):
        b=[b]
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        aa=[bind.get(itm, -1) for itm in a]
        return sum(np.array(aa)+1.)

    def get_data(trial_num, test_num=10):
        # trial_num:    number of training samples
        # test_num:     number of test samples
        pass

    def rand_sample_of_train_set(n):
        # n:      number of features
        # Return: np.array containing n samples of the training set
        X = np.load('data/X_iris_train.npy')
        y = np.load('data/y_iris_train.npy')
        idx = np.random.randint(len(X), size=n)
        return X[idx], y[idx]
        
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


        
    ############################################################################
    ## Parameters
    ############################################################################
    # Load training data
    # only load n_rand_data features of training set
    if rand_data == True:
        data, cls = rand_sample_of_train_set(trial_num)
    # load all features of training set
    else:
        # Only read data if not given as argument
        if data == "load" and cls == "load":
            #data = np.load('data/X_iris_train.npy')
            #cls = np.load('data/y_iris_train.npy')
            data = np.load('data_eeg/X_train_zied.npy')
            cls = np.load('data_eeg/y_train_zied.npy')

    if 1:
        data = PCA_dim_red(data, var_desired=0.9)


    # Simulation Parameters
    trial_num = len(cls) # How many samples (trials) from data will be presented 
    #n_training      = 1  # How many times the samples will be iterated
    n_trials        = n_training * trial_num # Total trials
    time_int_trials = 200. # (ms) Time to present each trial data 
    SIM_TIME        = n_trials * time_int_trials # Total simulation time (ms)
    #ts              = 1. # Timestep of Spinnaker (ms)
    min_del         = ts
    max_del         = 144 * ts
    p.setup(timestep=ts, min_delay=min_del, max_delay=max_del)


    ## Neuron Numbers
    #n_feature = 80   # Number of features (= 4 features * 20 neurons)
                     #           => 20 neuros: resolution of encoding
    n_pop     = 4    # Number of neurons in one population
    n_cl      = 2    # Number of classes at the output

    ## Connection Parameters
    # Weights
 #   wei_src_enc     = .2    # From Source Array at input to Encoding Layer(Exc)
 #   wei_enc_filt    = .6    # From Encoding Layer to Filtering Layer Exc neurons (Exc)
 #   wei_filt_inh    = 0.03  # From Filtering Layer Inh neurons to Exc neurons (Inh)
 #   wei_init_stdp   = .0    # From Filtering Layer Exc neurons to Output Layer (Exc)
 #   wei_cls_exc     = 0.9   # From Output Layer Exc neurons to Inh neurons (Exc)
 #   wei_cls_inh     = 10     # 0.1   # From Output Layer Inh neurons to Exc neurons (Inh) 
 #   wei_source_outp = 10.   # From Source Array at output to Output Layer Exc neurons (Exc)
 #   wei_noise_poi   = 0.02

    # Delays
    if randomness == True:    # if True:  calculate "del_src_enc" (randomly) new
                              # if False: load previously saved "del_src_enc"
        if reverse_src_del == True:
            # calc delays erversly proportional to feature value
            del_src_enc = np.zeros(n_feature*n_pop)
        else:
            del_src_enc = [int(np.random.randint(n_pop)+1)
                           for _ in range(n_feature*n_pop)]

        np.save("output_files/del_src_enc.npy", del_src_enc)
    else:
        #del_src_enc = np.load("output_files/del_src_enc.npy")
        del_src_enc = np.ones(n_feature*n_pop).astype(int) #[1 for _ in range(n_feature*n_pop)]
    del_enc_filt    = ts
    del_filt_inh    = ts
#    del_init_stdp   = 1.
    del_cls_exc     = ts
    del_cls_inh     = ts
#    del_source_outp = 1.
#    del_noise_poi   = 1.

    # Firing Rates
    noise_poi_rate  = 10. 
    max_fr_input    = 100.   # maximum firing rate at the input layer
    max_fr_rate_output = 20. # Maximum firing rate at output (supervisory signal)

    ## Connection Probabilities
#    prob_filt_inh   = .4 # Prob of connectivity inhibitory connections at FilT_Layer
#    prob_stdp       = 1. # Prob of STDP connections
#    prob_output_inh = .7 # Prob of inhibitory connections at Output Layer
#    prob_noise_poi_conn = 0.02

    ## STDP Parameters
#    tau_pl      = 0.3           # (0.2 - 0.3 works)
    tau_min     = tau_pl        # default tau_pl
#    stdp_w_max  = 0.4           # default 0.4
#    stdp_w_min  = 0.0           # default 0.0
#    stdp_A_pl   = 0.01          # default 0.01 (below 0.01 weights don't change)
    stdp_A_min  = -stdp_A_pl    # default - stdp_A_pl 
                                # => minus in order to get symmetric curve

    ## Neuron Parameters
    cell_params_lif = {'cm': 0.25,#1.,
                       'i_offset': 0.0,
                       'tau_m': 20.,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50#-65.0
                       }


    ############################################################################
    ## Data Extraction
    ############################################################################

    ## Extract Feature Data
#    scale_data = 2. # Scale features into [0-scale_data] range

    r,c = np.shape(data)

    data_rates = np.reshape(data, (1, r*c))[0]
    # Threshold (to keep spikes in range)
    thr_data_plus = 30
    thr_data_minus = -10
    #dd = [d if d<thr_data_plus else thr_data_plus for d in data_rates]
    #dd = [d if d>thr_data_minus else thr_data_minus for d in dd]

    # Shift and normalize data
    #dd2 = np.array(dd) - min(dd)
    dd2 = np.array(data_rates) - min(data_rates)
    dd2 = dd2 / max(dd2) * 2
    new_data_rates = []
    for r in dd2:
        new_data_rates += calc_pop_code(r, 0., scale_data, n_feature /
                                        (n_pop + 0.0))
    data_rates = list(max_fr_input*np.array(new_data_rates))

    ## Extract Class Data
    # load class vector
    #cls = np.load(path_y)
    cls = np.reshape(cls, (len(cls),1))    # create col vector
    r_cl, c_cl = np.shape(cls)
    #cls = list(np.reshape(cls, (1, r_cl * c_cl))[0] - 1)
    cls = list(np.reshape(cls, (1, r_cl * c_cl))[0])


    ## The class and rate infromation to be used during the simulation
    outputs = n_training * cls[0:trial_num]    # positiv, ints
    poi_rate = n_training  * data_rates[0:trial_num * n_feature]

    ## Save parameters to be used in test

    parameter_dict = {"n_feature":n_feature, "n_pop":n_pop,"n_cl":n_cl,
        "wei_src_enc":wei_src_enc, "wei_enc_filt":wei_enc_filt,
        "wei_filt_inh":wei_filt_inh, "wei_cls_exc":wei_cls_exc,
        "wei_cls_inh":wei_cls_inh, "del_enc_filt":del_enc_filt,
        "del_init_stdp":del_init_stdp, "del_cls_exc":del_cls_exc,
        "del_cls_inh":del_cls_inh, "trial_num":trial_num,
        "time_int_trials":time_int_trials, "scale_data":scale_data,
        "ts":ts,"max_fr_input":max_fr_input, 
        "max_fr_rate_output":max_fr_rate_output,
        "noise_poi_rate":noise_poi_rate, "max_fr_input":max_fr_input,
        "max_fr_rate_output":max_fr_rate_output, "prob_filt_inh":prob_filt_inh,
        "prob_stdp":prob_stdp, "prob_output_inh":prob_output_inh,
        "prob_noise_poi_conn":prob_noise_poi_conn, "tau_pl":tau_pl,
        "stdp_w_max":stdp_w_max, "stdp_w_min":stdp_w_min, "stdp_A_pl":stdp_A_pl,
        "wei_noise_poi":wei_noise_poi, "del_noise_poi":del_noise_poi,
        "thr_data_plus":thr_data_plus, "thr_data_minus":thr_data_minus
        }

    if save == True:
        np.save("output_files/parameters1",parameter_dict)
        np.save("output_files/parameters2",del_src_enc)

    ############################################################################
    ## Create populations for different layers
    ############################################################################
    poi_layer = []
    enc_layer = []
    filt_layer_exc = []
    out_layer_exc = []
    out_layer_inh = []
    out_spike_source = []

    # Calculate spike times at the input using the rate information coming from features
    spike_times = [[] for i in range(n_feature)]
    for i in range(n_trials):
        t_st = i * time_int_trials
        t_end = t_st + time_int_trials
        ind = i * n_feature
        for j in range(n_feature):
            times = PoissonTimes(t_st, t_end, poi_rate[ind+j], 
                                 np.random.randint(100))
            for t in times:
                spike_times[j].append(t)

    if randomness == True:    # if True:  calculate "spike_times" (randomly) new
                              # uf False: load previously saved "spike_times"
        np.save('output_files/spike_times_train.npy', spike_times)
    else:
        spike_times = np.load('output_files/spike_times_train.npy')



    # Calculate spike times at the output (as supervisory signal)
    out_spike_times=[[] for i in range(n_cl)]
    for i in range(n_trials):
        t_st = i * time_int_trials
        t_end = t_st + time_int_trials
        ind = outputs[i]
        times = PoissonTimes(t_st, t_end, max_fr_rate_output, 
                             np.random.randint(100))
        for t in times:
                out_spike_times[int(ind)].append(t)

    if randomness == True:    # if True:  calculate "out_spike_times" (randomly) new
                              # uf False: load previously saved "out_spike_times"
        np.save('output_files/out_spike_times.npy', out_spike_times)
    else:
        out_spike_times = np.load('output_files/out_spike_times.npy')


    # Spike source of input layer
    spike_source=p.Population(n_feature, 
                              p.SpikeSourceArray,
                              {'spike_times':spike_times},
                              label='spike_source')

    # Spike source of output layer (Supervisory signal)
    for i in range(n_cl):
        out_spike_source.append(p.Population(1, p.SpikeSourceArray,
            {'spike_times':[out_spike_times[i]]}, label='out_spike_source'))

    # Encoding layer and Filtering Layer definitions
    enc_layer = p.Population(n_feature * n_pop, 
                             p.IF_curr_exp,
                             cell_params_lif, 
                             label='enc_layer')
    filt_layer = p.Population(n_feature * n_pop, 
                              p.IF_curr_exp, 
                              cell_params_lif, 
                              label='filt_layer')

    # Excitatory and Inhibitory population definitions at the output
    for i in range(n_cl):    
        out_layer_exc.append(p.Population(n_pop, 
                                          p.IF_curr_exp, 
                                          cell_params_lif,
                                          label='out_layer_exc{}'.format(i)))
        out_layer_inh.append(p.Population(n_pop, 
                                          p.IF_curr_exp, 
                                          cell_params_lif, 
                                          label='out_layer_inh{}'.format(i)))
        out_layer_exc[i].record()

    # Noisy poisson population at the input

    poisson_input = p.Population(n_pop * 2, 
                                 p.SpikeSourcePoisson,
                                 {"rate":noise_poi_rate})


    # Record Spikes
    enc_layer.record()
    filt_layer.record()

    #enc_layer.initialize('v',p.RandomDistribution('uniform',[-51.,-69.]))
    #filt_layer.initialize('v',p.RandomDistribution('uniform',[-51.,-69.]))

    ############################################################################
    ## Projections
    ############################################################################

    ## Connection List from Spike Source Array to Encoding Layer
    conn_inp_enc=[]

    for i in range(n_feature):
        ind=i*n_pop
        for j in range(n_pop):
            conn_inp_enc.append([i,ind+j,wei_src_enc,del_src_enc[ind+j]])

    if save == True:
        np.save("output_files/conn_inp_enc",conn_inp_enc)

    ## Connection List for Filtering Layer Inhibitory
    if randomness == True:    # if True:  calculate conn_filt_inh (randomly) new
                              # uf False: load previously saved conn_filt_inh
        conn_filt_inh=[]
        for i in range(n_feature):
            rng1=i*n_pop
            rng2=rng1+n_pop
            inp=range(rng1,rng2)
            outp=range(0,rng1)+range(rng2,n_feature*n_pop)
            for ii in inp:
                for jj in outp:
                    if prob_filt_inh>np.random.rand():
                        conn_filt_inh.append([ii,jj,wei_filt_inh,del_filt_inh])
        if save == True:
            np.save('output_files/conn_filt_inh.npy', conn_filt_inh)
    else:
        conn_filt_inh = np.load('output_files/conn_filt_inh.npy')

    
    ## STDP Connection List 
    if randomness == True:    # if True:  calculate conn_stdp_list (randomly) new
                              # uf False: load previously saved conn_stdp_list
        conn_stdp_list=[[] for i in range(n_cl)]
        for i in range(n_cl): # For each population at output layer
            if use_old_weights == True:
                cl_weights = np.load("output_files/stdp_weights{}.npy".format(i))
                w = 0
            for ii in range(n_pop * n_feature): # For each neuron in filtering layer
                for jj in range(n_pop): # For each neuron in each population of output layer
                    if prob_stdp > np.random.rand(): # If the prob of connection is satiesfied
                        # Make the connection
                        if use_old_weights == True:
                            conn_stdp_list[i].append([ii,
                                                      jj, 
                                                      cl_weights[w], 
                                                      del_init_stdp])
                            w += 1 
                        else:
                            conn_stdp_list[i].append([ii,
                                                      jj, 
                                                      wei_init_stdp, 
                                                      del_init_stdp]) 
        if use_old_weights == False or save == True:
            np.save('output_files/conn_stdp_list.npy', conn_stdp_list)
    else:    
        conn_stdp_list = np.load('output_files/conn_stdp_list.npy')

    
    ## Output Layer Inhibitory Connection List
    if randomness == True:    # if True:  calculate conn_stdp_list (randomly) new
                              # uf False: load previously saved conn_stdp_list
        conn_output_inh = [[] for i in range(n_cl) for j in range(n_cl) if i!=j]
        c = 0
        for i in range(n_cl):
            for j in range(n_cl):
                if i != j:
                    for ii in range(n_pop):
                        for jj in range(n_pop):
                            if prob_output_inh > np.random.rand():
                                conn_output_inh[c].append([ii,
                                                           jj,
                                                           wei_cls_inh,
                                                           del_cls_inh])
                    c += 1
        if save == True:
            np.save("output_files/conn_output_inh.npy",conn_output_inh) 
    else:
        conn_output_inh = np.load("output_files/conn_output_inh.npy")

    

    ## Spike Source to Encoding Layer
    p.Projection(spike_source, enc_layer,
                 p.FromListConnector(conn_inp_enc))
    ## Encoding Layer to Filtering Layer
    p.Projection(enc_layer, filt_layer,
                 p.OneToOneConnector(weights=wei_enc_filt,
                                     delays=del_enc_filt))
    ## Filtering Layer Inhibitory
    p.Projection(filt_layer, filt_layer,
                 p.FromListConnector(conn_filt_inh),
                 target="inhibitory")

    ## STDP Connection between Filtering Layer and Output Layer
    timing_rule = p.SpikePairRule(tau_plus=tau_pl, 
                                  tau_minus=tau_min)
    weight_rule = p.AdditiveWeightDependence(w_max=stdp_w_max, 
                                             w_min=stdp_w_min, 
                                             A_plus=stdp_A_pl, 
                                             A_minus=stdp_A_min)
    stdp_model = p.STDPMechanism(timing_dependence=timing_rule, 
                                 weight_dependence=weight_rule)
    # STDP connection
    stdp_proj = []
    for j in range(n_cl):
        stdp_proj.append(
            p.Projection(filt_layer,out_layer_exc[j], 
                   p.FromListConnector(conn_stdp_list[j]), 
                   synapse_dynamics = p.SynapseDynamics(slow=stdp_model)))

    ## Connection between Output Layer neurons
    c = 0
    for i in range(n_cl):
        p.Projection(out_layer_exc[i], out_layer_inh[i], 
                     p.OneToOneConnector(weights=wei_cls_exc,
                                         delays=del_cls_exc))
        iter_array=[j for j in range(n_cl) if j!=i]
        for j in iter_array:
            p.Projection(out_layer_exc[i], out_layer_exc[j],
                         p.FromListConnector(conn_output_inh[c]),
                                             target="inhibitory")
            c += 1

    ## Spike Source Array to Output
    for i in range(n_cl):
        p.Projection(out_spike_source[i], 
                     out_layer_exc[i], 
                     p.AllToAllConnector(weights=wei_source_outp,
                                         delays=del_source_outp))
        iter_array = [j for j in range(n_cl) if j != i]
        for j in iter_array:
                p.Projection(out_spike_source[i],
                             out_layer_exc[j],
                             p.AllToAllConnector(weights=wei_source_outp,
                                                 delays=del_source_outp),
                                                 target="inhibitory")
    #for i in range(n_cl):
    #    p.Projection(out_spike_source[i], out_layer_exc[i], p.AllToAllConnector\
    #        (weights=wei_source_outp, delays=del_source_outp))
    #    p.Projection(out_spike_source[i], out_layer_exc[1-i], p.AllToAllConnector\
    #        (weights=wei_source_outp, delays=del_source_outp),target="inhibitory")

    ## Noisy poisson connection to encoding layer
    if randomness == True:    # if True:  connect noise to network
                              # if False: don't use noise in network
        p.Projection(poisson_input, enc_layer, 
                     p.FixedProbabilityConnector(p_connect=prob_noise_poi_conn, 
                                                 weights=wei_noise_poi, 
                                                 delays=del_noise_poi))
                
    ############################################################################
    ## Simulation
    ############################################################################
    p.run(SIM_TIME)

    Enc_Spikes = enc_layer.getSpikes()
    Filt_Exc_Spikes = filt_layer.getSpikes()

    Out_Spikes = [[] for i in range(n_cl)]
    for i in range(n_cl):
        Out_Spikes[i] = out_layer_exc[i].getSpikes()

    wei = []
    for i in range(n_cl):
        ww = stdp_proj[i].getWeights()
        if save == True:
            np.save("output_files/stdp_weights{}".format(i), ww)
        wei.append(ww)

    p.end()
    ############################################################################
    ## Plot
    ############################################################################
    ## Plot 1: Encoding Layer Raster Plot
    if 0:
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Encoding Layer Raster Plot')
        pylab.hold(True)
        pylab.plot([i[1] for i in Enc_Spikes], [i[0] for i in Enc_Spikes], ".b")
        pylab.hold(False)
        #pylab.axis([-10,c*SIM_TIME+100,-1,numInp+numOut+numInp+3])
        pylab.show()

    ## Plot 2-1: Filtering Layer Raster Plot
    if 0:
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Filtering Layer Raster Plot')
        pylab.plot([i[1] for i in Filt_Exc_Spikes], 
                   [i[0] for i in Filt_Exc_Spikes], ".b")
        #pylab.axis([-10,c*SIM_TIME+100,-1,numInp+numOut+numInp+3])
        pylab.show()

    ## Plot 2-2: Filtering Layer Layer Raster Plot
    if 0: 
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Filtering Layer Layer Raster Plot')
        pylab.hold(True)
        pylab.plot([i[1] for i in Filt_Exc_Spikes], [i[0] for i in Filt_Exc_Spikes], ".b")
        time_ind=[i*time_int_trials for i in range(len(outputs))]
        for i in range(len(time_ind)):
            pylab.plot([time_ind[i],time_ind[i]],[0,2000],"r")
        pylab.hold(False)
        #pylab.axis([-10,c*SIM_TIME+100,-1,numInp+numOut+numInp+3])
        pylab.show()


    ## Plot 3-1: Output Layer Raster Plot
    if 0:
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron')
        pylab.title('Output Layer Raster Plot')
        pylab.hold(True)
        c=0
        for array in Out_Spikes:
            pylab.plot([i[1] for i in array], [i[0]+c for i in array], ".b")
            c+=0.2
        pylab.hold(False)
        pylab.axis([-10,SIM_TIME+100,-1,n_pop+3])
        pylab.show()

    ## Plot 4: STDP WEIGHTS
    if 1:
        pylab.figure()
        pylab.xlabel('Weight ID')
        pylab.ylabel('Weight Value')
        pylab.title('STDP weights at the end')
        #pylab.title('STDP weights at the end' + ' (trail_num=' + str(trial_num) + ')')
        pylab.hold(True)
        for i in range(n_cl):
            pylab.plot(wei[i])
        pylab.hold(False)
        pylab.axis([-10,n_pop*n_feature*n_pop*0.5+10,-stdp_w_max,2*stdp_w_max])
        str_legend=["To Cl {}".format(i+1) for i in range(n_cl)]
        pylab.legend(str_legend)
        #pylab.show()
        fname = 'plots/weights_1.png'
        while True:
            if os.path.isfile(fname):    # if file already exists
                new_num =  int(fname.split('.')[0].split('_')[1]) + 1
                fname = fname.split('_')[0] + '_' +str(new_num) + '.png'
            else:
                pylab.savefig(fname)
                break
                

                


        #pylab.figure()
        #pylab.xlabel('Weight ID')
        #pylab.ylabel('Weight Value')
        #pylab.title('STDP weights at the end')
        #pylab.hold(True)
        #pylab.plot(wei[0], "b")
        #pylab.plot(wei[1], "g")
        #pylab.hold(False)
        #pylab.axis([-10, n_pop * n_feature * n_pop * 0.5 + 10, 
        #            -stdp_w_max, 2 * stdp_w_max])
        #pylab.legend(['To Cl 1','To Cl 2'])
        #pylab.show()

    ## Plot 5: Spike Source Spiking Times
    if 0:
        pylab.figure()
        pylab.hold(True)
        pylab.plot(out_spike_times[0],
                   [1 for i in range(len(out_spike_times[0]))],"x")
        pylab.plot(out_spike_times[1],
                   [1.05 for i in range(len(out_spike_times[1]))],"x")
        pylab.hold(False)
        pylab.title("Spike Source Spiking Times")
        pylab.axis([-100,SIM_TIME+100,-2,3])
        pylab.show()

        

    ## Calculate spiking activity of each neuron to each class inputs
    sum_filt=[[0 for i in range(n_feature*n_pop)] for j in range(n_cl)]
    sum_filt=np.array(sum_filt)

    for i in range(n_trials):
        t_st = i * time_int_trials
        t_end = t_st + time_int_trials
        cl = outputs[i]
        for n,t in Filt_Exc_Spikes:
            if t >= t_st and t < t_end:
                sum_filt[int(cl),int(n)] = sum_filt[int(cl), int(n)] + 1


    a4=sum_filt[0]
    b4=sum_filt[1]

    thr=20

    diff_vec=np.abs(a4 - b4)
    diff_thr=[i if i>thr else 0. for i in diff_vec]
    diff_ind=[i for i in range(len(diff_thr)) if diff_thr[i]!=0]
    if save == True:
        np.save("output_files/diff_ind_filt",diff_ind)

    diff2 = a4 - b4
    diff_thr2=[i if i > thr or i <- thr else 0. for i in diff2]
    diff_ind2=[i for i in range(len(diff_thr2)) if diff_thr2[i] != 0]
    if save == True:
        np.save("output_files/diff_ind_filt2",diff_ind2)
        np.save("output_files/diff_thr2",diff_thr2)

    ## Plot 6: Total Spiking Activity of Neurons at Decomposition Layer for Each Class
    if 0:
        a4=sum_filt[0]
        b4=sum_filt[1]
        pylab.figure()
        pylab.hold(True)
        pylab.plot(a4,"b")
        pylab.plot(b4,"r")
        pylab.xlabel('Neuron ID')
        pylab.ylabel('Total Firing Rates Through Trials')
        pylab.title("Total Spiking Activity of Neurons at Decomposition Layer for Each Class")
        pylab.hold(False)
        pylab.legend(["Activity to AN1","Activity to AN2"])
        pylab.show()

if __name__ == '__main__':
    train_snn()