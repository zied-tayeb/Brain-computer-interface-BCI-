# -*- coding: utf-8 -*-
"""
This code is test code for the conference paper 
Decoding of Motor Imagery Movements from EEG Signals using 
SpiNNaker Neuromorphic Hardware

@author: Emec Ercelik
"""

import pyNN.spiNNaker as p
import numpy as np
import pylab
import time
import scipy.io as sio


def test_snn(randomness      = False,
             data_dir        = "data/X_test_zied.npy",
             cls_dir         = "data/y_test_zied.npy",
             data            = "load",  # pass data as argument
             cls             = "load"): # pass labels as argument
    ###############################################################################
    ## Function Definitions
    ###############################################################################  
    def gaussian(x, mu, sig):
        return np.float16(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

    def calc_pop_code(feature, rng1, rng2, num):
        interval = np.float(rng2 - rng1) / num
        means = np.arange(rng1 + interval,rng2 + interval, interval)
        pop_code = [gaussian(feature, mu, 0.025) for mu in means]
        return pop_code
        
    def PoissonTimes2(t_str=0., t_end=100., rate=10., seed=1.):
        times = [t_str]
        rng = np.random.RandomState(seed=seed)
        cont = True
        while cont == True:
            t_next = np.floor(times[-1] + 1000. * next_spike_times(rng, rate))
            if t_next < t_end - 30:
                times.append(t_next[0])
            else:
                cont = False
                return times[1:]

    def PoissonTimes(t_str=0., t_end=100., rate=10., seed=1.):
        if rate > 0:    
            interval = (t_end - t_str+0.) / rate
            times = np.arange(t_str + 30, t_end - 40, interval)
            return list(times)    
        else:
            return []

    def next_spike_times(rng,rate):
        return -np.log(1.0 - rng.rand(1)) / rate   

    def ismember(a, b):
        b = [b]
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        aa=[bind.get(itm, -1) for itm in a]
        return sum(np.array(aa) + 1.)


    ###############################################################################
    ## Parameters
    ###############################################################################
    # Load Parameter
    parameters = np.load("output_files/parameters1.npy")
    parameters = parameters.item()
    # Load test data
    if data == "load" and cls == "load":
        data = np.load(data_dir)
        cls = np.load(cls_dir)
    # Simulation Parameters

    trial_num       = parameters["trial_num"] # How many samples (trials) from data will be presented 
    n_trials        = len(cls)#10#20 #int(trial_num) # Total trials
    time_int_trials = parameters["time_int_trials"] # (ms) Time to present each trial data 
    SIM_TIME        = n_trials * time_int_trials # Total simulation time (ms)
    ts              = parameters["ts"] # Timestep of Spinnaker (ms)
    min_del         = ts
    max_del         = 144 * ts
    p.setup(timestep=ts, min_delay=min_del, max_delay=max_del)


    ## Neuron Numbers

    n_feature = parameters["n_feature"] # Number of features
    n_pop     = parameters["n_pop"] # Number of neurons in one population
    n_cl      = parameters["n_cl"] # Number of classes at the output

    ## Connection Parameters
    # Weights
    wei_src_enc   = parameters["wei_src_enc"] # From Source Array at input to Encoding Layer(Exc)
    wei_enc_filt  = parameters["wei_enc_filt"] # From Encoding Layer to Filtering Layer Exc neurons (Exc)
    wei_filt_inh  = parameters["wei_filt_inh"]  # From Filtering Layer Inh neurons to Exc neurons (Inh)
    wei_cls_exc   = parameters["wei_cls_exc"] # From Output Layer Exc neurons to Inh neurons (Exc)
    wei_cls_inh   = parameters["wei_cls_inh"] # From Output Layer Inh neurons to Exc neurons (Inh) 
    wei_noise_poi = parameters["wei_noise_poi"]

    # Delays
    del_src_enc   = np.load("output_files/parameters2.npy")
    del_enc_filt  = parameters["del_enc_filt"]
    del_init_stdp = parameters["del_init_stdp"]
    del_cls_exc   = parameters["del_cls_exc"]
    del_cls_inh   = parameters["del_cls_inh"]
    del_noise_poi = parameters["del_noise_poi"]

    # Firing Rates
    noise_poi_rate     = parameters["noise_poi_rate"] 
    max_fr_input       = parameters["max_fr_input"] # maximum firing rate at the input layer
    max_fr_rate_output = parameters["max_fr_rate_output"] # Maximum firing rate at output (supervisory signal)

    ## Connection Probabilities
    prob_filt_inh       = parameters["prob_filt_inh"] # Prob of connectivity inhi-connections at Filtering Layer
    prob_stdp           = parameters["prob_stdp"] # Probability of STDP connections
    prob_output_inh     = parameters["prob_output_inh"] # Prob of inhi-connections at Output Layer
    prob_noise_poi_conn = parameters["prob_noise_poi_conn"]

    ## STDP Parameters
    tau_pl     = parameters["tau_pl"] #5
    tau_min    = tau_pl
    stdp_w_max = parameters["stdp_w_max"]
    stdp_w_min = parameters["stdp_w_min"]
    stdp_A_pl  = parameters["stdp_A_pl"]
    stdp_A_min = -stdp_A_pl # minus in order to get symmetric curve

    ## Neuron Parameters
    cell_params_lif = {'cm': 1.,
                       'i_offset': 0.0,
                       'tau_m': 20.,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -65.0
                       }



    ###############################################################################
    ## Data Extraction
    ###############################################################################

    ## Extract Feature Data
    scale_data = parameters["scale_data"] # Scale features into [0-scale_data] range

    #data = np.load("features_without_artifact.npy")
    #data = np.load('X_test.npy')
    r, c = np.shape(data)

    # Threshold (to keep spikes amplitudes in range)
    thr_data_plus = parameters["thr_data_plus"]
    thr_data_minus = parameters["thr_data_minus"]
    data_rates = np.reshape(data, (1, r * c))[0]

    # Shift an normalize the data
    #dd = [d if d<thr_data_plus else thr_data_plus for d in data_rates]
    #dd = [d if d>thr_data_minus else thr_data_minus for d in dd]
    #dd2 = np.array(dd) - min(dd)
    #dd2 = dd2 / max(dd2) * 2
    dd2 = np.array(data_rates) - min(data_rates)
    dd2 = dd2 / max(dd2) * 2
    new_data_rates = []
    for r in dd2:
        new_data_rates += calc_pop_code(r, 0., scale_data, 
                                        n_feature / (n_pop + 0.0))
    data_rates = list(max_fr_input * np.array(new_data_rates))

    ## Extract Class Data
    #cls = np.load("classes_without_artifact.npy")
    #cls = np.load("y_test.npy")
    cls = cls.reshape(len(cls), 1)
    r_cl, c_cl = np.shape(cls)
    cls = list(np.reshape(cls, (1, r_cl * c_cl))[0])

    outputs = cls[:n_trials]
    poi_rate = data_rates[:n_feature * n_trials]
    t1 = 0#70
    t2 = int(t1 + n_trials)
    outputs = cls[t1:t2]
    poi_rate = data_rates[t1 * n_feature:n_feature * t2]



    ###############################################################################
    ## Create populations for different layers
    ###############################################################################
    poi_layer = []
    enc_layer = []
    filt_layer_exc = []
    out_layer_exc = []
    out_layer_inh = []

    # Calculate poisson spike times for features
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
                              # if False: load previously saved "spike_times"
        np.save('output_files/spike_times_test.npy', spike_times)
    else:
        spike_times = np.load('output_files/spike_times_test.npy')



    # Spike source of input layer
    spike_source = p.Population(n_feature, 
                                p.SpikeSourceArray,
                                {'spike_times':spike_times},
                                label='spike_source')


    enc_layer = p.Population(n_feature * n_pop,
                             p.IF_curr_exp,
                             cell_params_lif,
                             label='enc_layer')
    
    filt_layer = p.Population(n_feature * n_pop,
                              p.IF_curr_exp, 
                              cell_params_lif, 
                              label='filt_layer')
    #filt_layer_inh=p.Population(n_feature*n_pop, p.IF_curr_exp, cell_params_lif, label='filt_layer_inh')


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

    poisson_input = p.Population(n_pop * 2,
                                 p.SpikeSourcePoisson,
                                 {"rate":noise_poi_rate})

    enc_layer.record()
    filt_layer.record()


    ###############################################################################
    ## Projections
    ###############################################################################


    ## Connection List from Spike Source Array to Encoding Layer
    conn_inp_enc = np.load("output_files/conn_inp_enc.npy")

    #Connection List for Filtering Layer Inhibitory
    conn_filt_inh = np.load("output_files/conn_filt_inh.npy")

    ## STDP Connection List
    conn_stdp_list = np.load("output_files/conn_stdp_list.npy")
    diff_ind = np.load("output_files/diff_ind_filt.npy")
    diff_ind2 = np.load("output_files/diff_ind_filt2.npy")
    diff_thr2 = np.load("output_files/diff_thr2.npy")
    c1 = 0
    for cls_list in conn_stdp_list:
        c2 = 0
        cls_wei = np.load("output_files/stdp_weights{}.npy".format(c1))
        mx = max(cls_wei)
        for conn in cls_list:
    #        if ismember(diff_ind,conn[0]):
            if (ismember(diff_ind2,conn[0]) and 
                    np.sign(c1-0.5) * np.sign(diff_thr2[int(conn[0])]) == -1.):
    #            conn[2]=0.08*cls_wei[c2]/mx
               conn[2] = 0.08#*diff_thr2[conn[0]]/36.
    #        conn[2]=2.*cls_wei[c2]
            c2 += 1
        c1 += 1
    conn_stdp_list = list(conn_stdp_list)


    ## Output Layer Inhibitory Connection List

    conn_output_inh = np.load("output_files/conn_output_inh.npy")


    ## Spike Source to Encoding Layer
    p.Projection(spike_source,enc_layer,p.FromListConnector(conn_inp_enc))
    ## Encoding Layer to Filtering Layer
    p.Projection(enc_layer, filt_layer,
                 p.OneToOneConnector(weights=wei_enc_filt, 
                                     delays=del_enc_filt))
    ## Filtering Layer Inhibitory
    p.Projection(filt_layer,filt_layer,
                 p.FromListConnector(conn_filt_inh),
                target="inhibitory")

    ## STDP Connection between Filtering Layer and Output Layer
    stdp_proj = []
    for j in range(n_cl):
        stdp_proj.append(p.Projection(filt_layer, out_layer_exc[j], 
                                      p.FromListConnector(conn_stdp_list[j])))

    ## Connection between Output Layer neurons
    c = 0
    for i in range(n_cl):
        p.Projection(out_layer_exc[i], out_layer_inh[i],
                     p.OneToOneConnector(weights=wei_cls_exc, 
                                         delays=del_cls_exc))
        iter_array = [j for j in range(n_cl) if j != i]
        for j in iter_array:
            p.Projection(out_layer_inh[i], out_layer_exc[j],
                         p.FromListConnector(conn_output_inh[c]),
                                             target="inhibitory")
            c+=1

    ## Noisy poisson connection to encoding layer
    if randomness == True:    # if True:  connect noise to network
                              # if False: don't use noise in network
        p.Projection(poisson_input,
                     enc_layer, 
                     p.FixedProbabilityConnector(p_connect=prob_noise_poi_conn,
                                                 weights=wei_noise_poi, 
                                                 delays = del_noise_poi))
    

    ###############################################################################
    ## Simulation
    ###############################################################################
    p.run(SIM_TIME)

    Enc_Spikes = enc_layer.getSpikes()
    Filt_Exc_Spikes = filt_layer.getSpikes()
    #Filt_Inh_Spikes = filt_layer_inh.getSpikes()

    Out_Spikes = [[] for i in range(n_cl)]
    for i in range(n_cl):
        Out_Spikes[i] = out_layer_exc[i].getSpikes()

    p.end()

    ###############################################################################
    ## Plot
    ###############################################################################
    ## Plot 1
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

    ## Plot 2-1
    if 0:
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Filtering Layer Raster Plot')
        pylab.plot([i[1] for i in Filt_Exc_Spikes], [i[0] for i in Filt_Exc_Spikes], ".b")
        #pylab.axis([-10,c*SIM_TIME+100,-1,numInp+numOut+numInp+3])
        pylab.show()

    ## Plot 2-2
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Filtering Layer Raster Plot')
        pylab.hold(True)
        pylab.plot([i[1] for i in Filt_Exc_Spikes], [i[0] for i in Filt_Exc_Spikes], ".b")
        time_ind=[i*time_int_trials for i in range(len(outputs))]
        for i in range(len(time_ind)):
            pylab.plot([time_ind[i],time_ind[i]],[0,2000],"r")
        pylab.hold(False)
        #pylab.axis([-10,c*SIM_TIME+100,-1,numInp+numOut+numInp+3])
        pylab.show()

    ## Plot 3-1
    if 0:
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title('Association Layer Raster Plot\nTest for Trial Numbers {}-{}'.format(t1,t2))
        pylab.hold(True)
        c=0
        for array in Out_Spikes:
            pylab.plot([i[1] for i in array], [i[0]+c for i in array], ".b")
            c+=0.2
        time_cls=[j*time_int_trials+i for j in range(len(outputs)) for i in range(int(time_int_trials))]
        cls_lb=[outputs[j]+0.4 for j in range(len(outputs)) for i in range(int(time_int_trials))]
        time_ind=[i*time_int_trials for i in range(len(outputs))]
        for i in range(len(time_ind)):
            pylab.plot([time_ind[i],time_ind[i]],[0,10],"r")
        #pylab.plot(time_cls,cls_lb,".")
        pylab.hold(False)
        pylab.axis([-10,SIM_TIME+100,-1,n_pop+2])
        pylab.show()


    ## Plot 3-2
        pylab.figure()
        pylab.xlabel('Time (ms)')
        pylab.ylabel('Neuron ID')
        pylab.title(('Association Layer Raster Plot\n',
                     'Test for Samples {}-{}').format(t1,t2))
        pylab.hold(True)

        pylab.plot([i[1] for i in Out_Spikes[0]], 
                   [i[0] for i in Out_Spikes[0]], 
                   ".b")
        pylab.plot([i[1] for i in Out_Spikes[1]], 
                   [i[0] + 0.2 for i in Out_Spikes[1]], 
                   ".r")

        time_ind = [i * time_int_trials for i in range(len(outputs))]
        for i in range(len(time_ind)):
            pylab.plot([time_ind[i], time_ind[i]], [0,n_pop], "k")
        #pylab.plot(time_cls,cls_lb,".")
        pylab.hold(False)
        pylab.axis([-10, SIM_TIME+100, -1, n_pop + 2])
        pylab.legend(["AN1","AN2" ])
        pylab.show()



    sum_output = [[] for i in range(n_cl)]

    for i in range(n_trials):
        t_st = i * time_int_trials
        t_end = t_st + time_int_trials
        for j in range(n_cl):
            sum_output[j].append(np.sum(
                [1 for n, t in Out_Spikes[j] if t >= t_st and t < t_end])
            )

    ## Plot 4
    if 0:
    #    pylab.figure()
    #    pylab.hold(True)
    #    pylab.plot(sum_output[0], "b.")
    #    pylab.plot(sum_output[1], "r.")
    #    out_cl0 = [i for i in range(len(outputs)) if outputs[i] == 0]
    #    out_cl1 = [i for i in range(len(outputs)) if outputs[i] == 1]
    #    pylab.plot(out_cl0,[-2 for i in range(len(out_cl0))], "xb")
    #    pylab.plot(out_cl1,[-2 for i in range(len(out_cl1))], "xr")
    #    pylab.hold(False)
    #    pylab.title("Total spikes at each AN population for each trial")
    #    pylab.xlabel("Trials")
    #    pylab.ylabel("Firing Rate")
    #    pylab.legend(["Cl0","Cl1","Winning Cl 0", "Winning Cl 1"])
    #    pylab.axis([-2, n_trials + 2, -4, max(max(sum_output)) + 30])
    #    pylab.show()
        pylab.figure()
        pylab.hold(True)
        pylab.plot(sum_output[0], "b^")
        pylab.plot(sum_output[1], "r^")
        #pylab.plot(sum_output[0],"b")
        #pylab.plot(sum_output[1],"r")
        ppp0 = np.array(sum_output[0])
        ppp1 = np.array(sum_output[1])
        out_cl0 = [i for i in range(len(outputs)) if outputs[i] == 0]
        out_cl1 = [i for i in range(len(outputs)) if outputs[i] == 1]
        pylab.plot(out_cl0, ppp0[out_cl0], "bs")
        pylab.plot(out_cl1, ppp1[out_cl1], "rs")
        pylab.hold(False)
        pylab.title("Total spikes at each AN population for each trial")
        pylab.xlabel("Trials")
        pylab.ylabel("Spike Count for Each Trial")
        pylab.legend(["Cls 0", "Cls 1", "Actual Winner Cls 0", 
                      "Actual Winner Cls 1"])
        pylab.axis([-2, n_trials + 2, -4, max(max(sum_output)) + 30])
        pylab.show()


    ## Check Classification rate
    s = np.array(sum_output)
    cl = np.floor((np.sign(s[1] - s[0]) + 1) / 2)
    r_cl = np.array(outputs)
    wrong = np.sum(np.abs(cl - r_cl))
    rate = (n_trials - wrong) / n_trials
    print("success rate: {}%".format(abs(rate)*100.))

    print("cl:\n", cl)
    print("r_cl:\n", r_cl)

    ## Plot 5
    if 0:
        pylab.figure()
        cf = 0.1
        pylab.hold(True)
        cls_wei0 = np.load("output_files/stdp_weights{}.npy".format(0))
        mx = max(cls_wei0)
        cls_wei0 = cf * cls_wei0 / mx
        cls_wei1 = np.load("output_files/stdp_weights{}.npy".format(1))
        mx = max(cls_wei1)
        cls_wei1 = cf * cls_wei1/ mx
        l = min(len(cls_wei0), len(cls_wei1))
        new_array0 = [cls_wei0[i] for i in range(l) if cls_wei0[i] > cls_wei1[i]]
        x0 = [i for i in range(l) if cls_wei0[i] > cls_wei1[i]]
        new_array1 = [cls_wei1[i] for i in range(l) if cls_wei1[i] > cls_wei0[i]]
        x1 = [i for i in range(l) if cls_wei1[i] > cls_wei0[i]]

        pylab.plot(x0, new_array0, "gx")
        pylab.plot(x1, new_array1, "bx")
        #for i in range(2):
        #    cls_wei=np.load("stdp_weights{}.npy".format(i))
        #    mx=max(cls_wei)
        #    cls_wei=0.05*cls_wei/mx
        #    pylab.plot(cls_wei,"x")
        pylab.axis([-10, 2000, -0.1, 0.15])
        pylab.hold(False)
        pylab.show()
     
    ## Plot 7
    if 0:
        sum_filt = [[0 for i in range(n_feature * n_pop)] for j in range(n_cl)]
        sum_filt = np.array(sum_filt)

        for i in range(n_trials):
            t_st = i * time_int_trials
            t_end = t_st + time_int_trials
            cl = outputs[i]
            for n, t in Filt_Exc_Spikes:
                if t >= t_st and t < t_end:
                    sum_filt[int(cl),int(n)] = sum_filt[(cl),int(n)] + 1

        a4=sum_filt[0]
        b4=sum_filt[1]
        pylab.figure()
        pylab.hold(True)
        pylab.plot(a4,"b.")
        pylab.plot(b4,"r.")
        pylab.xlabel('Neuron ID')
        pylab.ylabel('Total Firing Rates Through Trials')
        pylab.title("Total Spiking Activity of Neurons at Decomposition Layer for Each Class")
        pylab.hold(False)
        pylab.legend(["Activity to AN1","Activity to AN2"])
        pylab.show()   

    return rate

if __name__ == '__main__':
    test_snn()