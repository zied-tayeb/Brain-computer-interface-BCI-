import os
import sys
import time
import numpy as np
import iris_clf_test_snn
import iris_clf_train_snn


def multiply_weights():
	# w: np.array containing the weights
	n_cl = 2
	for i in range(n_cl):
		cl_weights = np.load("output_files/stdp_weights{}.npy".format(i))
		print("\n\n Weights: {}\n\n".format(cl_weights[:10]))
		cl_weights *= 0.5
		print("\n\n Weights: {}\n\n".format(cl_weights[:10]))
		np.save("output_files/stdp_weights{}.npy".format(i), cl_weights)


# ------------------------------------------------------------------------------
# Check number of cycles
try:
	n_cycles = int(sys.argv[1])
except:
	n_cycles = 1	
print("\n {} Cycle(s)! \n".format(n_cycles))


# --- SIMULATION ---------------------------------------------------------------

# Simulation Parameter
homeostasis     = False
binary_weights  = False
epochs          = 1
# Network Parameter
randomness 		= True
rand_data       = False
trial_num       = 90
reverse_src_del = False
# STDP Parameter
tau_pl          = 10.

### Run Train and Test Scripts n_cycles times
rates = []
for c in range(n_cycles):

	print('Training network ... ')
	iris_clf_train_snn.train_snn(n_training      = 4,   #2
								 use_old_weights = False,
								 randomness      = randomness,
							 	 tau_pl          = tau_pl,
							 	 rand_data       = rand_data,
								 trial_num       = trial_num,
								 reverse_src_del = reverse_src_del) 
	for _ in range(epochs - 1):
		if homeostasis == True:
			multiply_weights()
		if binary_weights == True:
			make_binary_weights()
		iris_clf_train_snn.train_snn(n_training      = 1,
									 use_old_weights = True,
									 randomness      = randomness,
									 tau_pl          = tau_pl,
									 rand_data       = rand_data,
									 trial_num       = trial_num,
									 reverse_src_del = reverse_src_del)

	# Wait 2 sec
	t_sleep = 1    # in seconds
	print('Waiting {} seconds ... '.format(t_sleep))
	time.sleep(t_sleep)

	### Test network
	print('Training network ... ')
	rates.append(iris_clf_test_snn.test_snn(randomness = True))
	print("Cycle {}, CLF Rate = {}".format(c+1, rates[-1]))


# Save clf rates
np.save("output_files/clf_rates.npy", rates)

# Print results
if n_cycles > 1:
	print("Results ({} cylces): Mean CLF-Rate = {}, with std.dev = {}".format(
		n_cycles, np.mean(rates), np.std(rates)))

