import math
import numpy as np
from itertools import product

def autogrid(x, y, z, sx, sy, sz, buffer=0):

	if z is None:
		nz = 1
		oz = 0
		max_z = 0
	else:
		oz = min(z) - buffer #+ sz/2
		max_z = max(z) + buffer + sz/2
		nz = math.ceil((max_z - oz)/sz)

	ox = min(x) - buffer #+ sx/2
	oy = min(y) - buffer #+ sy/2
	max_x = max(x) + buffer + sx/2
	max_y = max(y) + buffer + sy/2
	nx = math.ceil((max_x - ox)/(sx))
	ny = math.ceil((max_y - oy)/(sy))
	

	return {'ox':ox,'oy':oy,'oz':oz,'sx':sx,'sy':sy,'sz':sz,'nx':nx,'ny':ny,'nz':nz}

def add_coord(grid):
    x_coord = np.array([(grid['ox']+x*grid['sx']) for x in range(grid['nx'])])
    y_coord = np.array([(grid['oy']+y*grid['sy']) for y in range(grid['ny'])])
    z_coord = np.array([(grid['oz']+z*grid['sz']) for z in range(grid['nz'])])

    coords_array = []
    for x,y,z in product(x_coord, y_coord, z_coord):
        coords_array.append(np.array([x,y,z]))

    return np.array(coords_array)