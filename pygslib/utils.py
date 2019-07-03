import math

def autogrid(x, y, z, sx, sy, sz, buffer=0):

	if z is None:
		nz = 1
		oz = 0
		max_z = 0
	else:
		oz = min(z) - buffer
		max_z = max(z) + buffer
		nz = math.ceil((max_z - oz)/sz) #+ 1

	ox = min(x) - buffer
	oy = min(y) - buffer
	max_x = max(x) + buffer
	max_y = max(y) + buffer
	nx = math.ceil((max_x - ox)/sx) #+ 1
	ny = math.ceil((max_y - oy)/sy) #+ 1
	

	return {'ox':ox,'oy':oy,'oz':oz,'sx':sx,'sy':sy,'sz':sz,'nx':nx,'ny':ny,'nz':nz}