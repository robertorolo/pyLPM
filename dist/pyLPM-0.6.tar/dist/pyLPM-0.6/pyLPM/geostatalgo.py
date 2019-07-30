from scipy.interpolate import NearestNDInterpolator
from pyLPM import utils
import numpy as np

def nn(x,y,z,var,grid):
    """Neares neighbor estimator
    
    Args:
        x (array): x coordinates data array
        y (array): y coordinates data array
        z (array): z coordinates data array
        var (array): variable data array
        grid (dict): grid definitions dictionary
    
    Returns:
        array: NN results array
    """
    mask = np.isfinite(var)
    if z is None:
        z = np.zeros(len(x))
    points_array = np.array([x[mask],y[mask],z[mask]]) 
    knn = NearestNDInterpolator(points_array.T, var)

    grids_points_array = utils.add_coord(grid)

    results = knn(grids_points_array)

    return results