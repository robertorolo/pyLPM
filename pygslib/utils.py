import math
import numpy as np

def autogrid(x, y, z, sx, sy, sz, buffer=0):

	if z is None:
		z = np.zeros(len(x))

	ox = min(x) - buffer
	oy = min(y) - buffer
	oz = min(z) - buffer
	max_x = max(x) + buffer
	max_y = max(y) + buffer
	max_z = max(z) + buffer
	nx = math.ceil((max_x - ox)/sx) #+ 1
	ny = math.ceil((max_y - oy)/sy) #+ 1
	nz = math.ceil((max_z - oz)/sz) #+ 1

	return {'ox':ox,'oy':oy,'oz':oz,'sx':sx,'sy':sy,'sz':sz,'nx':nx,'ny':ny,'nz':nz}