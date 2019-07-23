import math

def autogrid(x, y, z, sx, sy, sz, buffer=0):

	if z is None:
		nz = 1
		oz = 0
		max_z = 0
	else:
		oz = min(z) - buffer + sz/2
		max_z = max(z) + buffer
		nz = math.ceil((max_z - oz)/sz)

	ox = min(x) - buffer + sx/2
	oy = min(y) - buffer + sy/2
	max_x = max(x) + buffer
	max_y = max(y) + buffer
	nx = math.ceil((max_x - ox)/(sx))
	ny = math.ceil((max_y - oy)/(sy))
	

	return {'ox':ox,'oy':oy,'oz':oz,'sx':sx,'sy':sy,'sz':sz,'nx':nx,'ny':ny,'nz':nz}