from scipy.interpolate import NearestNDInterpolator
from pyLPM import utils
import numpy as np

def NN(x,y,z,var,grid):
    mask = np.isfinite(var)
    if z is None:
        z = np.zeros(len(x))
    points_array = np.array([x[mask],y[mask],z[mask]]) 
    knn = NearestNDInterpolator(points_array.T, var)

    grids_points_array = utils.add_coord(grid)

    results = knn(grids_points_array)

    return results