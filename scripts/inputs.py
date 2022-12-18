
import copy
import numpy as np
import yaml


class Inputs:

	def __init__(self, mc_yaml):
		self.mc_yaml = yaml.load(open(mc_yaml, 'r'), Loader=yaml.BaseLoader)
		self.input_dict = None
		self.normalized_input_dict = None


	def draw(self, n=1, seed=1):


		np.random.seed(seed=10)

		# new_yaml_dict = copy.deepcopy(self.mc_yaml)


		# for key, mc_dict in self.mc_yaml.items():

		# 	for mc_name, mc_params in mc_dict.items():

		# 		params = mc_params['params']
		# 		params = [float(_val) for _val in params]

		# 		if mc_params['type'] == 'normal':
		# 			val = np.random.normal(params[0], params[1], size=n)
		# 		elif mc_params['type'] == 'lognormal':
		# 			val = np.random.lognormal(params[0], params[1], size=n)
		# 		elif mc_params['type'] == 'uniform':
		# 			val = np.random.uniform(params[0], params[1], size=n)
		# 		else:
		# 			print('undefined')
		# 			print(mc_name)

		# 		new_yaml_dict[key][mc_name]['value'] = val
		# 		new_yaml_dict[key][mc_name]['params'] = params
		
		# self.input_dict = new_yaml_dict


		major_keys = ['Motor','Rocket','Nose','Fin','Tail','Flight']
		minor_keys = [['Thrust', 'Burnout', 'NGrains', 'GrainSep', 'GrainDen', 'NozzleRad', 'ThroatRad'],
		['Radius', 'Mass', 'InertiaI', 'InertiaZ', 'DistNozzle', 'DistProp'],
		['Length', 'DistCM'],
		['NFins', 'Span', 'RootChord', 'TipChord', 'DistCM'],
		['TopRad', 'BotRad', 'Length', 'DistCM'],
		['Incline', 'Heading']]

		data_vec = []
		normed_data_vec = []

		for major, minor_list in zip(major_keys, minor_keys):
			for minor in minor_list:

				params = self.mc_yaml[major][minor]['params']
				params = [float(_val) for _val in params]

				_type = self.mc_yaml[major][minor]['type']

				if _type == 'normal':
					val = np.random.normal(params[0], params[1], size=n)
					normed_val = (val - params[0]) / params[1]
				elif _type == 'lognormal':
					val = np.random.lognormal(params[0], params[1], size=n)
					normed_val = (np.log(val) - params[0]) / params[1]
				elif _type == 'uniform':
					val = np.random.uniform(params[0], params[1], size=n)
					mean = (params[1] + params[0]) / 2
					std = (params[1] - params[0]) / np.sqrt(12)
					normed_val = (val - mean) / std

				else:
					print('undefined')
					print(mc_name)

				data_vec.append(val)
				normed_data_vec.append(normed_val)

		data_vec = np.array(data_vec).T
		normed_data_vec = np.array(normed_data_vec).T
	
		self.input_data = data_vec.tolist()
		self.normalized_data = normed_data_vec


	def normalize(self):
		

		normalized_input_dict = copy.deepcopy(self.input_dict)
	
		for key, mc_dict in self.input_dict.items():
			for mc_name, mc_params in mc_dict.items():
				
				params = mc_params['params']
				#params = [float(_val) for _val in params]

				if mc_params['type'] == 'normal':
					normalized_input_dict[key][mc_name]['value'] = (normalized_input_dict[key][mc_name]['value'] - params[0]) / params[1]
				elif mc_params['type'] == 'lognormal':
					val = np.log(normalized_input_dict[key][mc_name]['value'])
					normalized_input_dict[key][mc_name]['value'] = (val - params[0]) / params[1]
				elif mc_params['type'] == 'uniform':
					val = normalized_input_dict[key][mc_name]['value']
					mean = params[1]-params[0]
					std = (params[1] - params[0]) / np.sqrt(12)
					normalized_input_dict[key][mc_name]['value'] = (val - mean) / std


		self.normalized_input_dict = normalized_input_dict

	def __iter__(self):
		self._i = 0
		self._max = len(self.input_data)

		return self

	def __next__(self):
		if self._i < self._max:
			
			inputs = self.input_data[self._i]
			normed_inputs = self.normalized_data[self._i,:]
			self._i += 1

			return inputs, normed_inputs
		else:
			raise StopIteration



if __name__ == '__main__':

	mc_yaml = '/home/adambirenbaum/Documents/SEG/Trajectory_ML/inputs/mc_params.yaml'

	inputs = Inputs(mc_yaml)
	inputs.draw(n=5)
	inputs.normalize()
	print(inputs.normalized_input_dict)
	breakpoint()

