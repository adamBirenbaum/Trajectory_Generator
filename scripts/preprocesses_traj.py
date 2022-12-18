
import numpy as np
import os
from outputs_lstm import Output
import matplotlib.pyplot as plt
from scipy import interpolate


import sys



def interp_features(data, dt, i):

	

	time = data[:,0]
	time_new = np.arange(time[0],time[-1],dt)

	interp_data = np.array([interpolate.interp1d(time, data[:,i])(time_new) for i in range(1,10)]).T
	
	
	return interp_data


def add_padding(data, max_len):

	num_steps, num_cols = data.shape

	if num_steps < max_len:

		num_to_add = max_len - num_steps
		
		data = np.concatenate((data, np.zeros((num_to_add, num_cols))), axis=0)
	else:
		data = data[:max_len, :]

	return data
	

def extract_6dof(data):

	# time x y z vx vy vz e0 e1 e2 e3 w1 w2 w3

	e0 = data[:,7]  
	e1 = data[:,8] 
	e2 = data[:,9] 
	e3 = data[:,10] 

	
	# spin euler angle
	phi = np.arctan2(e3, e0) - np.arctan2(-e2, -e1)
	# nutation euler angle
	theta =  np.arcsin(-((e1 ** 2 + e2 ** 2) ** 0.5))
	# precession euler angle
	psi =  np.arctan2(e3, e0) + np.arctan2(-e2, -e1)
	#pitch = np.arcsin(2*(e0*e2-e3*e1))
	#roll = np.arctan2(2*(e0*e3+e1*e2), 1-2*(np.power(e2,2)+np.power(e3,2)))


	data[:,7] = phi
	data[:,8] = theta
	data[:,9] = psi
	
	
	return data[:,:10]

def process_data(dir_name, dt, max_len):

	runs_per_file = 10000

	proc_dir = os.path.join('..','outputs','Processed',dir_name)
	raw_dir = os.path.join('..','outputs','Raw',dir_name)

	os.makedirs(proc_dir, exist_ok=True)

	n = int(len(os.listdir(raw_dir))/3)

	# ToDo: This Output class isn't needed. only using nums attribute
	output = Output(raw_dir)

	out_inds = output.nums

	nfiles = n // runs_per_file
	remaining = n % runs_per_file
	

	# num runs total
	total_runs = 0
	# num runs in file
	current_file = 0
	# file number with 10000 runs
	ifile = 0
	# empty_array
	array_to_write = []
	n_bad = 0
	

	for file_ind in out_inds:

		amtDone = current_file / len(out_inds)

		sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

		#file_ind = out_inds[total_runs]
		data = np.loadtxt(os.path.join(raw_dir, 'Outputs_run_{:07d}.txt'.format(file_ind)))

		data = extract_6dof(data)
		
		# Specific case here where some runs are going in the opposite direction. weeding those out
		if np.any((data[:,1] < 0) & (data[:,0] > 10)):
			n_bad += 1
			continue 
		
		data = interp_features(data, dt, total_runs)
		
		data = add_padding(data,max_len)
		

		array_to_write.append(data)


		

		total_runs += 1
		current_file += 1


	
	filename = os.path.join(proc_dir, 'processed_features.txt')
		
	np.savetxt(filename, np.concatenate(array_to_write))


	print('\n\nSaved: {}\nN Bad: {}'.format(total_runs, n_bad))
	


if __name__ == '__main__':

	
	dt = 0.25

	dir_name = 'Test'

	max_len = 500

	
	
	process_data(dir_name, dt, max_len)