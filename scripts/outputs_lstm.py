

import glob
from make_model_function import make_model
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from tensorflow.keras.metrics import mse
physical_devices = tf.config.experimental.list_physical_devices('GPU')

config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Output:
	def __init__(self, dir_path):
		self.dir = dir_path

		self.get_nums()

		#self.get_inputs()



	def get_nums(self):

		input_files = glob.glob(os.path.join(self.dir, 'Inputs_*.txt'))

		nums = [0] * len(input_files)

		for i, input_file, in enumerate(input_files):

			
			nums[i] = int(re.search('Inputs_run_(.*).txt', os.path.basename(input_file)).group(1))

		nums.sort()
		self.nums = nums


	def get_lstm_inputs(self, n_samples, n_seq, fun, nfeat):
		
		
		inputs = np.zeros((n_samples, n_seq, nfeat))


		total_size = 0
		_iter = 0
		while total_size < n_samples:
			amtDone = total_size / n_samples
			sys.stdout.write("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(amtDone * 50), amtDone * 100))

			output_file = os.path.join(self.dir, 'Outputs_run_{:07d}.txt'.format(_iter))
			if os.path.exists(output_file):
				outputs = np.loadtxt(output_file)

				
				new_inputs = fun(outputs,n_inputs)

				if len(new_inputs) == n_seq:
					inputs[total_size] = new_inputs

					total_size += 1

			_iter += 1
		
		# max_val = np.max(inputs)
		# normed = (inputs - 0) / (max_val - 0)
	
		self.inputs = np.reshape(inputs,(n_samples, n_seq, nfeat))


	def get_features(self, filename):

		feat = np.loadtxt(filename)

		nruns = re.search('.*_([0-9]+)_[0-9]+_13.txt', filename).group(1)
		nsteps = re.search('.*_[0-9]+_([0-9]+)_13.txt', filename).group(1)
		
		feat = feat.reshape((nruns, nsteps, 13))

		self.inputs = feat

	def get_inputs(self):

		inputs = [None] * len(self.nums)
		normed_inputs = [None] * len(self.nums)

		for i, ilabel in enumerate(self.nums):

			input_file = os.path.join(self.dir, 'Inputs_run_{:07d}.txt'.format(ilabel))
			normed_file = os.path.join(self.dir, 'NormInputs_run_{:07d}.txt'.format(ilabel))

			inputs[i] = np.loadtxt(input_file)

			normed_inputs[i] = np.loadtxt(normed_file)

		inputs = np.vstack(inputs)
		normed_inputs = np.vstack(normed_inputs)

		self.inputs = inputs
		self.normed_inputs = normed_inputs


	def get_labels(self, fun, *params):

		labels = [None] * len(self.nums)

		for i, ilabel in enumerate(self.nums):

			output_file = os.path.join(self.dir, 'Outputs_run_{:07d}.txt'.format(ilabel))
			outputs = np.loadtxt(output_file)

			labels[i] = fun(outputs, *params)
			
		labels = np.squeeze(np.vstack(labels))
		self.labels=labels

	def train(self, train_size, model_path, epochs):
	
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.normed_inputs, self.labels, train_size=train_size)

		model = make_model(model_path)


		batch_size = 256


		tf.random.set_seed(100)

		model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=batch_size, epochs=epochs)

		self.model = model
		#model_basename = os.path.splitext(os.path.basename(model_path))[0]
		#model.save('../models/{}.h5'.format(model_basename))

	def predict(self):

		self.predict_test = self.model.predict(self.X_test)
		breakpoint()

	def plot_inputs(self):
		#major_keys = ['Motor','Rocket','Nose','Fin','Tail','Flight']
		minor_keys = ['Thrust', 'Burnout', 'NGrains', 'GrainSep', 'GrainDen', 'NozzleRad', 'ThroatRad', 'Radius', 'Mass', 'InertiaI', 'InertiaZ', 'DistNozzle', 'DistProp', 'Length', 'DistCM', 'NFins', 'Span', 'RootChord', 'TipChord', 'DistCM', 'TopRad', 'BotRad', 'Length', 'DistCM',	'Incline', 'Heading']


		fig, ax = plt.subplots(5,6, figsize=(15,15))
		ax = ax.flatten()
		for i in range(self.inputs.shape[1]):

			ax[i].hist(self.inputs[:,i],bins=50)
			ax[i].set_title(minor_keys[i])

		fig2, ax2 = plt.subplots(5,6, figsize=(15,15))
		ax2 = ax2.flatten()
		for i in range(self.inputs.shape[1]):

			ax2[i].hist(self.normed_inputs[:,i],bins=50)
			ax2[i].set_title(minor_keys[i])


		plt.show()
	



def is_impact_within_circle(data, ymin, ymax):

	return int(data[-1,2] >= ymin and data[-1,2] <= ymax)


def get_data_around_apogee(data, n):

	
	ind = np.argmax(data[:,3])

	
	ind_range = range(ind-n,ind+n+1)


	try:
		norm_data = data[ind_range, 3:4]
		max_height = np.max(norm_data[:,0])
	except IndexError:
		return [None]
	
	min_height = np.min(norm_data[:,0])
	norm_data[:,0] = (norm_data[:,0] - min_height) / (max_height - min_height)
	
	return norm_data

def normalize(vec, _min=None,_max=None):

	if _min is None:
		_min = np.min(vec)
		_max = np.max(vec)
	
	if _min == 0 and _max == 0:
		return 0.0

	return (vec - _min) / (_max - _min)



def get_data_around_apogee_all(data, n):

	max_height_ind = np.argmax(data[:,3])

	ind_range = range(max_height_ind - n, max_height_ind + n + 1)

	try:
		norm_data = data[ind_range, 1:]
	except IndexError:
		return [None]

	#x
	norm_data[:,0] = normalize(norm_data[:,0])
	#y
	norm_data[:,1] = normalize(norm_data[:,1])
	#z
	norm_data[:,2] = normalize(norm_data[:,2])
	#vx
	norm_data[:,3] = normalize(norm_data[:,3])
	#vy
	norm_data[:,4] = normalize(norm_data[:,4])
	#vz
	norm_data[:,5] = normalize(norm_data[:,5])
	#e0
	norm_data[:,6] = normalize(norm_data[:,6],_min=-1,_max=1)
	#e1
	norm_data[:,7] = normalize(norm_data[:,7],_min=-1,_max=1)
	#e2
	norm_data[:,8] = normalize(norm_data[:,8],_min=-1,_max=1)
	#e3
	norm_data[:,9] = normalize(norm_data[:,9],_min=-1,_max=1)
	#w0
	norm_data[:,10] = normalize(norm_data[:,10])
	#w1
	norm_data[:,11] = normalize(norm_data[:,11])
	#w2
	norm_data[:,12] = normalize(norm_data[:,12])



	# fig, ax = plt.subplots(4,4, sharey=True)
	# ax = ax.flatten()

	# for i in range(norm_data.shape[1]):
	# 	ax[i].plot(norm_data[:,i])
	# 	ax[i].set_title('plot {}'.format(i))
	# 	ax[i].set_ylim(0,1)

	# plt.show()
	# breakpoint()

	return norm_data

def get_data_around_apogee2(data, n):

	
	ind = np.argmax(data[:,3])

	
	ind_range = range(ind-n,ind+n+1)

	
	try:
		norm_data = data[ind_range, 2:15]
		max_height = np.max(norm_data[:,1])
		max_y = np.max(norm_data[:,0])
	except IndexError:
		return [None]
	
	min_height = np.min(norm_data[:,1])
	min_y = np.min(norm_data[:,0])
	norm_data[:,0] = (norm_data[:,0] - min_y) / (max_y - min_y)
	norm_data[:,1] = (norm_data[:,1] - min_height) / (max_height - min_height)
	
	return norm_data

def make_good_model(nfeat):
	# define model
	model = Sequential()
	# encoder
	#model.add(LSTM(100, activation='relu', input_shape=(n_seq,1)))
	model.add(LSTM(100, input_shape=(n_seq,nfeat),return_sequences=True,activation='tanh'))
	model.add(Dropout(rate=0.2))
	model.add(LSTM(50,return_sequences=False,activation='tanh'))
	# decoder
	model.add(RepeatVector(n_seq))
	#model.add(LSTM(100, activation='relu', return_sequences=True))

	model.add(LSTM(50, return_sequences=True,activation='tanh'))
	model.add(LSTM(100,return_sequences=True,activation='tanh'))
	model.add(Dropout(rate=0.2))
	model.add(TimeDistributed(Dense(nfeat)))
	model.compile(optimizer='adam', loss='mse',metrics='mse')
	return model

def make_simple_model(nfeat):
	# define model
	model = Sequential()
	# encoder
	model.add(LSTM(100, activation='relu', input_shape=(n_seq,nfeat), return_sequences=True))
	#model.add(CuDNNLSTM(100, input_shape=(n_seq,nfeat)))
	model.add(Dropout(rate=0.2))
	# decoder
	model.add(RepeatVector(n_seq))
	model.add(LSTM(100, activation='relu', return_sequences=True))

	#model.add(CuDNNLSTM(100,return_sequences=True))
	model.add(Dropout(rate=0.2))
	model.add(TimeDistributed(Dense(nfeat)))
	model.compile(optimizer='adam', loss='mse',metrics='mse')
	return model

if __name__ == '__main__':


	# aa = np.loadtxt('/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Output2/Outputs_run_0000275.txt')

	# fig, ax = plt.subplots(4,4)
	# ax = ax.flatten()

	# for i in range(aa.shape[1]):
	# 	ax[i].plot(aa[:,i])
	# 	ax[i].set_title('plot {}'.format(i))

	# plt.show()

	# breakpoint()


	dir_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/outputs/Output2'
	model_path = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/models/neural_network1.yaml'
	plot_dir = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/plots'

	plot_iter_name = 'test'

	n_inputs = 10
	n_samples = 275000
	epochs=100
	train_size = 0.8

	batch_size = 256

	n_seq = n_inputs * 2 + 1
	output = Output(dir_path)
	nfeat = 13

	if nfeat == 1:
		fun = get_data_around_apogee
	elif nfeat < 4:
		fun = get_data_around_apogee2
	else:
		fun = get_data_around_apogee_all

	output.get_lstm_inputs(n_samples, n_seq, fun,nfeat)


	X_train, X_test, y_train, y_test = train_test_split(output.inputs, output.inputs, train_size=train_size)

	
	model = make_good_model(nfeat)
	print(model.summary())

	history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,  validation_data=(X_test, X_test))

	yhat = model.predict(X_test)

	
	this_plot_dir = os.path.join(plot_dir, plot_iter_name)

	os.makedirs(this_plot_dir, exist_ok=True)

	fig1, ax1 = plt.subplots(figsize=(10,10))

	ax1.plot(history.history['val_loss'], label='validation')
	ax1.plot(history.history['loss'], label='train')
	ax1.grid()
	ax1.legend()
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')
	ax1.set_title('Training Loss: {:4.5f}\nValidation Loss: {:4.5f}'.format(history.history['loss'][-1], history.history['val_loss'][-1]))
	fig1.savefig(os.path.join(this_plot_dir, 'learning_curve__{}.png'.format(n_samples)))

	feature_labels=['x','y','z','vx','vy','vz','e0','e1','e2','e3','w0','w1','w2']

	for j in range(nfeat-1):

		n_plots = 16

		fig, ax = plt.subplots(4,4, figsize=(16,16))
		ax = ax.flatten()

		for i in range(n_plots):

			actual = output.inputs[i,:,j]
			pred = yhat[i,:,j]
			ax[i].plot(actual, '-r', label='truth')
			ax[i].plot(pred, '--b', label='prediction')
			
			ax[i].grid()
			ax[i].set_title('MSE: {:6.7f}'.format(mse(actual, pred).numpy()))

			ax[i].legend()

		fig.suptitle('Feature: {}'.format(feature_labels[j]))
		fig.savefig(os.path.join(this_plot_dir, 'feature_{}__{}.png'.format(j, n_samples)))

	plt.show()
	
