

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import os





if __name__ == '__main__':

	out_dir = os.path.join('..','outputs','Raw','Test')
	N = 1000


	fig = plt.figure(figsize = (10,10))
	ax = plt.axes(projection='3d')

	for i in range(N):
		try:
			data = np.loadtxt(os.path.join(out_dir, 'Outputs_run_{:07d}.txt'.format(i)))

			ax.plot3D(data[:,1], data[:,2], data[:,3],color='k',alpha=0.3)
			ax.set_xlabel('x', labelpad=20)
			ax.set_ylabel('y', labelpad=20)
			ax.set_zlabel('z', labelpad=20)
		except:
			continue

	plt.show()