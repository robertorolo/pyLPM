from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import GridspecLayout
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline as pyo
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors,ticker,cm 
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import fitpack
from scipy.interpolate import griddata
from scipy import interpolate
import numpy as np 
import pandas as pd 
import time
import math 
import numba
import warnings
import math
import os

from pyLPM import plots

# INITIALIZE OFFLINE NOTEBOOK MODE
pyo.init_notebook_mode()

##############################################################################################################################################

# global variables 

global return_exp_var # return of experimental variograms
return_exp_var = []

global return_model_var #return of model variograms 
return_model_var = {}

##############################################################################################################################################

# Calculate the distance coordinates 

@numba.jit(fastmath=False)
def _hdist(distancex, distancey, distancez):
	
	"""euclidian distance between samples 
	
	Args:
	    distancex (np.array): distances in x coordinates 
	    distancey (np.array): distances in y coordinates 
	    distancez (np.array): distances in z coordinantes
	
	Returns:
	    np.array: list of euclidian distances between samples 
	"""
	dist =np.zeros(distancex.shape[0])
	for i in range(distancex.shape[0]):
		dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2 + distancez[i]**2) + 0.0000000001
	return dist

##############################################################################################################################################

# Calculate the distances in planar coordinates

@numba.jit(fastmath=False)
def _xydist(distancex, distancey):
	
	"""Planar distances between samples 
	
	Args:
	    distancex (np.array): distances in x coordinates 
	    distancey (np.array): distances in y coordinates 
	
	Returns:
	    np.array: List of planar distances 
	"""
	dist =np.zeros(distancex.shape[0])
	for i in range(distancex.shape[0]):
		dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2 ) + 0.0000000001
	return dist

##############################################################################################################################################

# Calculate the x coordinates distances

@numba.jit(fastmath=False)
def _xdist(pairs):
	"""Summary
	
	Args:
	    pairs (lst): Sample points 
	
	Returns:
	    np.array: x coordinate distances 
	"""
	dist =np.zeros(pairs.shape[0])

	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][0] - pairs[i][1][0])
	return dist

##############################################################################################################################################

# Calculate the y coordinates distances

@numba.jit(fastmath=False)
def _ydist(pairs):
	"""Summary
	
	Args:
	    pairs (lst): Sample points 
	
	Returns:
	    np.array: y coordinate distances 
	"""
	dist =np.zeros(pairs.shape[0])
	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][1] - pairs[i][1][1])
	return dist

##############################################################################################################################################

# Calculate the z coordinates distances

@numba.jit(fastmath=False)
def _zdist(pairs):
	"""Summary
	
	Args:
	    pairs (lst):  Sample points  
	
	Returns:
	    np.array: z coordinate distances 
	"""
	dist =np.zeros(pairs.shape[0])
	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][2] - pairs[i][1][2])
	return dist

##############################################################################################################################################

# Calculate pairs combination where vertical and horizontal distances are less than the maximum distance

@numba.jit(fastmath=False)
def _combin(points,n, max_dist):
	"""Summary
	
	Args:
	    points (lst): sample points 
	    n (integer): number of samples 
	    max_dist (float): maximum permissible distance
	
	Returns:
	    lst: List of tuples containing the permissible pairs 
	"""
	dist =[]
	p = 0
	for i in range(0,n):
		for j in range((i+1),n):
			if (points[i][0] - points[j][0]) < max_dist:
				if (points[i][1] - points[j][1]) < max_dist:
					if (points[i][2] - points[j][2]) < max_dist:
						dist.append((points[i] , points[j]))
			p += 1
	return np.array(dist)

##############################################################################################################################################

# Rotate data according azimuth and dip directions 

@numba.jit(fastmath=False)
def _rotate_data(xh, yh, zh, azimute, dip):
	"""Summary
	
	Args:
	    xh (lst): x coordinate distances
	    yh (lst): y coordinate distances 
	    zh (lst): z coordinate distances
	    azimute (lst): azimuth directions 
	    dip (lst): dip directions 
	
	Returns:
	    lst: Rotate coordiantes Xrot, Yrot and Zrot
	"""
	xhrot = []
	yhrot = []
	zhrot = []

	for i in range(0, len(xh)):

		    # ROTACIONE PRIMEIRAMENTE NO AZIMUTE

		    xrot = math.cos(math.radians(azimute))*xh[i] - math.sin(math.radians(azimute))*yh[i]
		    yrot = math.sin(math.radians(azimute))*xh[i] + math.cos(math.radians(azimute))*yh[i]
		    yrot2 = math.cos(math.radians(dip))*yrot - math.sin(math.radians(dip))*zh[i]
		    zrot = math.sin(math.radians(dip))*yrot + math.cos(math.radians(dip))*zh[i]

		    xhrot.append(xrot)
		    yhrot.append(yrot2)
		    zhrot.append(zrot)

	return xhrot, yhrot, zhrot

##############################################################################################################################################
			 
# Calculate all distances in dataset 

def _distances(dataset, nlags, x_label, y_label, z_label, 
	lagdistance, head_property, tail_property, choice):

	'''distances

	Args:
	    dataset (pd.DataFrame): pandas dataset containing all spatial data 
	    nlags (int): Number of lags
	    x_label (string): Label for X direction 
	    y_label (string): Label for Y direction 
	    z_label (string): Label for Z direction 
	    lagdistance (float): Lag size 
	    head_property (string): Label for head variable 
	    tail_property (string): Label for the tail variable 
	    choice (float): Percentual of datasest sampling 0.0-1.0
	
	Returns:
		pd.DataFrame: Distances of permissible pairs 


	'''

	# Variables definition 
	max_dist = (nlags + 1)*lagdistance # Define the maximum distance search 
	X = dataset[x_label].values 
	Y = dataset[y_label].values
	Z = dataset[z_label].values
	HEAD = dataset[head_property].values
	TAIL = dataset[tail_property].values

	# If random option is selected, select a number of data according some proportion 

	if (choice != 1.0):
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
		points = np.array([points[np.random.randint(0,len(points))] for i in range(int(points.shape[0]*choice))])
	else:
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))

	# Define the permissible combination of samples according the maximum distance

	pairs = _combin(points,points.shape[0], max_dist)

	# Define distance variables and dataset 

	distancex = _xdist(pairs) 
	distancey =  _ydist(pairs) 
	distancez =  _zdist(pairs)
	distanceh = _hdist(distancex, distancey, distancez)
	distancexy = _xydist(distancex, distancey)
	head_1 = np.array([pair[0][3] for pair in pairs])
	head_2 = np.array([pair[0][4] for pair in pairs])
	tail_1 = np.array([pair[1][3] for pair in pairs])
	tail_2 = np.array([pair[1][4] for pair in pairs])

	# Return the distances dataframe 

	distance_dataframe =  np.array([distancex, 
					distancey, 
					distancez, 
					distancexy, 
					distanceh, 
					head_1,
					head_2,
					tail_1,
					tail_2,]).T

	return distance_dataframe[distanceh[:,] < max_dist ] # Only values less than the maximum distance

##############################################################################################################################################

def _distances_varmap(dataset, X,Y,Z, nlags, lagdistance, head_property, tail_property,
					choice):

	'''distances
	
	Args:
	    dataset (pd.DataFrame): DataFrame containing all spatial data
	    X (string): Label for the X coordinates 
	    Y (string): Label for the Y coordinates 
	    Z (string): Label for the Z coordinates 
	    nlags (int): Number of lags to calculate distannces
	    lagdistance (float): Size of the lag 
	    head_property (string): Label for the head property 
	    tail_property (string): Label for the tail property
	    choice (float): Perncentual of data sample 
	Return:
		np.array: All permissible dataset distances 
	'''

	# Define coordinates 

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	# Define the maximum distance 

	max_dist = (nlags + 1)*lagdistance

	# Define tail and head properties 

	HEAD = dataset[head_property].values
	TAIL = dataset[tail_property].values

	# If random option is selected, select a number of data according some proportion 

	if (choice != 1.0):
		np.random.seed(0)
		np.random.seed(138276)
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
		points = np.array([points[np.random.randint(0,len(points))] for i in range(int(points.shape[0]*choice))])
	else:
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))

	# Define the permissible combination of samples according the maximum distance

	pairs = _combin(points,points.shape[0], max_dist)

	# Define distance variables and dataset 


	distancex = _xdist(pairs) 
	distancey =  _ydist(pairs) 
	distancez =  _zdist(pairs)
	distanceh = _hdist(distancex, distancey, distancez)
	distancexy = _xydist(distancex, distancey)
	head_1 = np.array([pair[0][3] for pair in pairs])
	head_2 = np.array([pair[0][4] for pair in pairs])
	tail_1 = np.array([pair[1][3] for pair in pairs])
	tail_2 = np.array([pair[1][4] for pair in pairs])

	# Return the distances dataframe 


	distance_dataframe =  np.array([distancex, 
					distancey, 
					distancez, 
					distancexy, 
					distanceh, 
					head_1,
					head_2,
					tail_1,
					tail_2,]).T
	return distance_dataframe[distanceh[:,] < max_dist ] # Only values less than the maximum distance

##############################################################################################################################################

def __permissible_pairs (lag_multiply, lagdistance, lineartolerance, check_azimuth,
						check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,
						dist):

	'''permissible_pairs
	
	Args:
	    lag_multiply (double): Mutliplemaxi of lag distance
	    lagdistance (TYPE): Description
	    lineartolerance (TYPE): Description
	    check_azimuth (TYPE): Description
	    check_dip (TYPE): Description
	    check_bandh (TYPE): Description
	    check_bandv (TYPE): Description
	    htol (TYPE): Description
	    vtol (TYPE): Description
	    hband (TYPE): Description
	    vband (TYPE): Description
	    omni (TYPE): Description
	    dist (TYPE): Description
	
	Returns:
	    distances (numpy array): Returns the permissible sample pairs for omnidirecional functions
	'''
	# Define the minimum range and the maximum rannge 

	minimum_range = lag_multiply*lagdistance - lineartolerance
	maximum_range = lag_multiply*lagdistance + lineartolerance

	# Filter samples acoording distances tolerances 

	if omni == False:
		filter_dist = dist[(dist[:,4] >= minimum_range) & 
						  (dist[:,4] <= maximum_range) & 
						  (check_azimuth >= htol) &
						  (check_dip >= vtol) &
						  (check_bandh < hband)&
						  (check_bandv < vband)]
	else:
		filter_dist = dist[(dist[:,4] >= minimum_range) & 
						  (dist[:,4] <= maximum_range)]
	return filter_dist

##############################################################################################################################################

def __calculate_experimental(dist, lag_multiply, lagdistance, lineartolerance, check_azimuth,
					check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,type_var):

	'''calculate_experimental
	
	Args:
	    dist (pd.DataFrame): Pandas Dataframe containing all spatial data
	    lag_multiply (int): Number of lag mutliplication
	    lagdistance (float): Lag distance size 
	    lineartolerance (float): Linear tolerance length 
	    check_azimuth (float): check if azimuth direction is permissible 
	    check_dip (float): check if dip direction is permissible 
	    check_bandh (float): check if horizontal band is permissible 
	    check_bandv (float): check if vertical band is permissible 
	    htol (float): Horizontal angular tolerance 
	    vtol (float): Vertical angular tolerance
	    hband (float): Horizontal band width 
	    vband (float): Vertical band width 
	    omni (bool): Boolean for omnidirecional variogram (True= use, False =don`t use)
	    type_var (string): Experimental Continuity function 

	    type_var:
	    	'Variogram'
	    	'Covariogram'
	    	'Correlogram'
	    	'PairWise'
	    	'Relative_Variogram'
	
	Returns:
	    value (double): Experimental continuity function value 
	    number_of_pairs (int): number of pairs used to calculate the experimental function 
	    average_distace (double): average distance of experimental continuity function value  
	
	Raises:
	    Exception: Experimental continuity function not in permissible functions 
	'''

	# Calculate the permissible pairs for experimental continuity functions 

	points = __permissible_pairs(lag_multiply, lagdistance, lineartolerance, check_azimuth,
						check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,
						dist)

	# Show error if experimental variogram is not valid 
	
	if type_var not in['Variogram','Covariogram','Correlogram','PairWise','Relative_Variogram']:
		raise Exception("Experimental continuity function not in admissible functions")

	# Calculate experimental variogram according the specific experimental function 

	if points.size  != 0:
		number_of_pairs = float(points.shape[0])
		average_distance = points[:,4].mean()
		value_exp = 0
		if type_var == 'Variogram': 
			value_exp = ((points[:,5] - points[:,7])*(points[:,6] - points[:,8]))/(2*number_of_pairs)
			value_exp = value_exp.sum()
		if type_var == 'Covariogram': 
			value_exp = ((points[:,5] - points[:,5].mean())*(points[:,8]-points[:,8].mean()))/number_of_pairs
			value_exp = value_exp.sum()
		if type_var == 'Correlogram':
			value_exp = ((points[:,5] - points[:,5].mean())*(points[:,8]-points[:,8].mean()))/(number_of_pairs*points[:,5].var()*points[:,8].var())
			value_exp = value_exp.sum()
		if type_var == 'PairWise':
			value_exp = 2*((points[:,5] - points[:,7])/(points[:,6] + points[:,8]))**2/number_of_pairs
			value_exp = value_exp.sum()
		if type_var == 'Relative_Variogram':
			average_tail = (points[:,7] +  points[:,8])/2
			average_head = (points[:,5] +  points[:,6])/2
			value_exp = 4*((points[:,5] - points[:,7])*(points[:,6] - points[:,8]))/(number_of_pairs*(average_head + average_tail)**2)
			value_exp = value_exp.sum()
		return [value_exp, number_of_pairs, average_distance]
	return [np.nan , np.nan, np.nan]

##############################################################################################################################################

def _calculate_experimental_function(dataset, x_label, y_label, z_label, 
					type_var, lagdistance, lineartolerance, htol, vtol, hband, vband, azimuth, dip, nlags, 
					head_property, tail_property, choice, omni):

	'''calculate_experimental_function
	
	Args:
	    dataset (pd.DataFrame): Pandas dataframe containing all values
	    x_label (string): Label for x coordinates
	    y_label (string): Label for y coordinates
	    z_label (string): Label for z coordinates 
	    type_var (string): Experimental continuity function 
		
		type_var:
	    	'Variogram'
	    	'Covariogram'
	    	'Correlogram'
	    	'PairWise'
	    	'Relative_Variogram'

	    lagdistance (float): Size of lag size
	    lineartolerance (float): Linear tolerance
	    htol (float): Horizontal angular tolerance 
	    vtol (float): Vertical angular tolerance
	    hband (float): Horizontal band width 
	    vband (float): Vertical band width 
	    azimuth (float): Azimuth direction in degrees
	    dip (float): Float direction in degrees
	    nlags (int): Number of lags to calculate experimental functions
	    head_property (string): String with name of  Head Property
	    tail_property (string): String with anme of Tail Property
	    choice (float): Percentual of random sampling 0-1
	    omni (bool): Omnidirecional option True =Use omnidirecional , False = Don't use omnidirecional
	
	Returns:
	    df (pandas.DataFrame): Pandas Dataframe containing the experimental continuity functions of all lags
	'''
	# Calculate all permissible distances

	dist = _distances(dataset, nlags, x_label, y_label, z_label, lagdistance, head_property, tail_property, choice)

	# Define the angles and tolerances 

	cos_Azimuth = np.cos(np.radians(90-azimuth))
	sin_Azimuth = np.sin(np.radians(90-azimuth))
	cos_Dip     = np.cos(np.radians(90-dip))
	sin_Dip     = np.sin(np.radians(90-dip))
	htol = np.abs(np.cos(np.radians(htol)))
	vtol= np.abs(np.cos(np.radians(vtol)))
	check_azimuth = np.abs((dist[:,0]*cos_Azimuth + dist[:,1]*sin_Azimuth)/dist[:,3])
	check_dip     = np.abs((dist[:,3]*sin_Dip + dist[:,2]*cos_Dip)/dist[:,4])
	check_bandh   = np.abs(cos_Azimuth*dist[:,1]- sin_Azimuth*dist[:,0])
	check_bandv	  = np.abs(sin_Dip*dist[:,2] - cos_Dip*dist[:,3])

	# Create experimental variogram dataset 

	number_of_int = range(1, (nlags +1))
	value_exp = np.array([__calculate_experimental(dist, i, lagdistance, lineartolerance, check_azimuth,
					check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,type_var) for i in number_of_int])
	df = pd.DataFrame(value_exp, 
					  columns = ['Spatial continuity', 'Number of pairs', 'Average distance'])
	df = df.dropna()

	return df

##############################################################################################################################################

def _calculate_experimental_function_varmap(type_var,  lineartolerance, htol, vtol, hband, vband, 
					azimuth, dip,dataset, nlags, 
					X_rot, Y_rot,Z_rot,
					lagdistance, head_property, tail_property, choice , 
					omni = False, plot_graph=False, show_pairs=False):

	'''calculate_experimental_function
	
	Args:
	    type_var (string): Type of experimental variogram function 

		type_var:
	    	'Variogram'
	    	'Covariogram'
	    	'Correlogram'
	    	'PairWise'
	    	'Relative_Variogram'

	    lineartolerance (TYPE): Description
	    htol (TYPE): Description
	    vtol (TYPE): Description
	    hband (TYPE): Description
	    vband (TYPE): Description
	    azimuth (TYPE): Description
	    dip (TYPE): Description
	    dataset (TYPE): Description
	    nlags (TYPE): Description
	    X_rot (TYPE): Description
	    Y_rot (TYPE): Description
	    Z_rot (TYPE): Description
	    lagdistance (TYPE): Description
	    head_property (TYPE): Description
	    tail_property (TYPE): Description
	    choice (TYPE): Description
	    omni (bool, optional): Description
	    plot_graph (bool, optional): Description
	    show_pairs (bool, optional): Description
	
	Returns:
	    df (pandas.DataFrame): Pandas Dataframe containing the experimental continuity functions of all lags
	'''
	# Calculate all permissible distances

	dist = _distances_varmap(dataset, X_rot, Y_rot,Z_rot, nlags, lagdistance, head_property, tail_property,choice)

	# Define the angles and tolerances 

	cos_Azimuth = np.cos(np.radians(90-azimuth))
	sin_Azimuth = np.sin(np.radians(90-azimuth))
	cos_Dip     = np.cos(np.radians(90-dip))
	sin_Dip     = np.sin(np.radians(90-dip))
	htol = np.abs(np.cos(np.radians(htol)))
	vtol= np.abs(np.cos(np.radians(vtol)))
	check_azimuth = np.abs((dist[:,0]*cos_Azimuth + dist[:,1]*sin_Azimuth)/dist[:,3])
	check_dip     = np.abs((dist[:,3]*sin_Dip + dist[:,2]*cos_Dip)/dist[:,4])
	check_bandh   = np.abs(cos_Azimuth*dist[:,1]- sin_Azimuth*dist[:,0])
	check_bandv	  = np.abs(sin_Dip*dist[:,2] - cos_Dip*dist[:,3])

	# Create experimental variogram dataset 

	number_of_int = range(1, (nlags +1))
	value_exp = np.array([__calculate_experimental(dist, i, lagdistance, lineartolerance, check_azimuth,
					check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,type_var) for i in  number_of_int])
	df = pd.DataFrame(value_exp, 
					  columns = ['Spatial continuity', 'Number of pairs', 'Average distance'])
	df = df.dropna()
	if plot_graph == True:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(df['Average distance'].values, df['Spatial continuity'].values)
		ax.set_xlabel('Lag distance (h)')
		ax.set_ylabel(type_c)
		ax.set_title('Experimental continuity function : {} , azimuth {}  and dip {} '.format(str(type_c), str(azimuth), str(dip)))
		if show_pairs == True:
			x, y = df['Average distance'].values, df['Spatial continuity'].values
			for i, j  in enumerate(df['Number of pairs'].values):
				ax.annotate(str(j), xy =(x[i], y[i]), xytext =(x[i], (y[i]+0.05*y[i])))
			ax.set_ylim((min(y),1.10*max(y)))
		plt.grid()
		plt.show()
	return df  

##############################################################################################################################################

def _modelling(experimental_dataframe,azimuths, dips, rotation_reference,
 model_func, ranges, contribution, nugget, inverted= False, plot_graph = True ):

	'''plot_experimental_function)_omni
	
	Args:
	    experimental_dataframe (pd.DataFrame): pandas Data Frame containing the experimental continuity functions 
	    azimuths (lst): List of azimuth directions  
	    dips (lst): List of dip directions 
	    rotation_reference (lst): List containing the azimuth and dips for all experimental variograms [azm, dip]
	    model_func (str): String containing the experimental function model 

		model_func:
		'Exponential'
		'Gaussian'
		'Spherical'

	    ranges (lst): List containing the ranges for each structure for experimental continuity functionns 
	    contribution (lst): List containing the variogram model contribution for each structure  
	    nugget (float): The nugget effect
	    inverted (bool, optional): Invert variogram modelling to a covariogram modelling 
	    plot_graph (bool, optional): Option to plot graph 
	
	Returns:
	    plot (matplotlib.pyplot): Plot of omnidirecional experimental continuity function and the spatial continuity model for one direction 
	
	Raises:
	    ValueError: Number of principal directions must be less or equal 3 or Variogram structures not have the same size
	'''

	# Give errors for modelling inconsistences 

	if len(ranges[0]) != 3:
		raise ValueError("Number of principal directions must be less or equal 3")
	if len(ranges) == len(contribution) and len(ranges) == len(model_func):
		pass
	else:
		raise ValueError("Variogram structures must be have the same size")

	# Calculate cartesian coordiantes according polar approach 


	y = math.cos(math.radians(dips))*math.cos(math.radians(azimuths))
	x = math.cos(math.radians(dips))*math.sin(math.radians(azimuths))  
	z = math.sin(math.radians(dips))

	# Define reference plane direction s

	angle_azimuth = math.radians(rotation_reference[0])
	angle_dip = math.radians(rotation_reference[1])
	angle_rake = math.radians(rotation_reference[2])

	# Define rotation matrix


	rotation1 = np.array([[math.cos(angle_azimuth), -math.sin(angle_azimuth), 0],
				 [math.sin(angle_azimuth), math.cos(angle_azimuth), 0],
				 [0,0,1]])

	rotation2 = np.array([[1, 0, 0],
				 [0, math.cos(angle_dip), math.sin(angle_dip)],
				 [0,-math.sin(angle_dip),math.cos(angle_dip)]])

	rotation3 = np.array([[math.cos(angle_rake), 0, -math.sin(angle_rake)],
				 [0, 1, 0],
				 [math.sin(angle_rake),0,math.cos(angle_rake)]])

	rot1 = np.dot(rotation1.T, np.array([x,y,z]))
	rot2 = np.dot(rotation2.T, rot1)
	rot3= np.dot(rotation3.T,rot2)

	# Rotate samples 

	rotated_range =[]

	for i in ranges:
		rangex = float(i[1])
		rangey = float(i[0])
		rangez = float(i[2])


		rotated = (np.multiply(rot3, np.array([rangex, rangey, rangez]).T))
		rotated_range.append(math.sqrt(rotated[0]**2+rotated[1]**2+rotated[2]**2))

	# Calculate model function 

	distancemax = experimental_dataframe['Average distance'].max()
	distances = np.linspace(0, distancemax, 200)

	model = []

	if inverted == False:
		for i in distances:
			soma = 0
			for j, o, l  in zip(contribution, model_func, rotated_range):
				if o == 'Exponential':
					soma += j*(1-math.exp(-3*i/float(l)))
				if o == 'Gaussian':
					soma += j*(1-math.exp(-3*(i/float(l))**2))
				if o == 'Spherical':
					if i <= l:
						soma += j*(1.5*i/float(l)-0.5*(i/float(l))**3)
					else:
						soma += j
			soma += nugget
			model.append(soma)
	else:
		for i in distances:
			soma = 0
			for j, o, l  in zip(contribution, model_func, rotated_range):
				if o == 'Exponential':
					soma += (j+nugget)*(math.exp(-3*i/float(l)))
				if o == 'Gaussian':
					soma += (j+nugget)*(math.exp(-3*(i/float(l)**2)))
				if o == 'Spherical':
					if i <= l:
						soma += (j+nugget)*(1 - (1.5*i/float(l)-0.5*(i/float(l))**3))
					else:
						soma += 0
			model.append(soma)

	df = pd.DataFrame(np.array([distances, model]).T, columns= ['distances', 'model']) 

	return df 
	
		
		

##############################################################################################################################################
	
def _varmap(azimuth, dip, lineartolerance, htol, vtol, hband, vband, type_var, dataset, x_label, y_label, z_label, nlags, 
					lagdistance, head_property, tail_property, ndirections, wait_please, choice = 1.00, interactive = False):
	"""Summary
	
	Args:
	    azimuth (float): Azimuth of reference plane
	    dip (dip): Dip of the reference plane 
	    lineartolerance (TYPE): Linear tolerance 
	    htol (float): Horizontal tolerance
	    vtol (float): Vertical tolerance 
	    hband (float): Horizontal bandwidth 
	    vband (float): Vertical bandwidth 
	    type_var (float): Experimental continuity function 

		type_var:
	    	'Variogram'
	    	'Covariogram'
	    	'Correlogram'
	    	'PairWise'
	    	'Relative_Variogram'

	    dataset (pd.DataFrame): pandas DataFrame containing the data
	    x_label (str): Label for x coordinates
	    y_label (str): Label for y coordinates
	    z_label (str): Label for z coordinates 
	    nlags (int): Number of lags to calculate experimental variogram
	    lagdistance (float): Lag size 
	    head_property (str): String for the head property
	    tail_property (str): String for the tail property
	    ndirections (int): Number of directions for calculate experimental function
	    wait_please (obj): Object to update progress bar
	    choice (float, optional): Values to random select experimental variograms
	    interactive (bool, optional): Enable interactive varmaps
	"""
	# Define X, Y, Z coordinates

	X = dataset[x_label].values
	Y = dataset[y_label].values
	Z = dataset[z_label].values

	# Set progress bar to zero 

	if interactive == True:
			wait_please.value = 0

	# rotate dataset acording the reference plane

	Xrot, Yrot, Zrot = _rotate_data(X, Y, Z, azimuth, dip)

	# Calculate experimental functions 

	lag_adm = []
	azimute_adm = []
	continuidade = []
	for i in np.arange(0,360,ndirections):

		if interactive == True:
			wait_please.value += 1
		df_exp = _calculate_experimental_function_varmap(type_var,  lineartolerance, htol, vtol, hband, vband, 
					i, 0,dataset, nlags, Xrot, Yrot,Zrot,lagdistance, head_property, tail_property, choice)
		var_val = df_exp['Spatial continuity'].values
		continuidade = np.append(continuidade,var_val)
		values_1 = df_exp['Average distance'].values
		lag_adm = np.append(lag_adm,values_1)
		values_2 = np.array([np.radians(i) for j in range(df_exp.shape[0])])
		azimute_adm = np.append(azimute_adm, values_2)

	# Define the interplation parameters 

	gdiscrete = 40
	ncontour = 40

	# define the x and y coordinates in cartesian way 

	x = np.array(lag_adm)*np.sin(azimute_adm)
	y = np.array(lag_adm)*np.cos(azimute_adm)

	# Define the maximum grid size 

	maximo = max([max(x), max(y)])
	
	# Filter values less than a circle distance

	ray = maximo 
	truzao = []
	for j, i in enumerate(x):
		if math.sqrt(x[j]**2 + y[j]**2) > ray:
			truzao.append(False)
		else: 
			truzao.append(True) 

	x= x[truzao]
	y =y[truzao]
	continuidade = continuidade[truzao]

	x= x.ravel()
	y= y.ravel()

	# Create grid and interpolation 


	Xi = np.linspace(-maximo,maximo,gdiscrete) 
	Yi = np.linspace(-maximo,maximo,gdiscrete)


	f = plt.figure(figsize=(15,10))
	left, bottom, width, height= [0,0.1, 0.7, 0.7]
	ax  = plt.axes([left, bottom, width, height])
	pax = plt.axes([left, bottom, width, height],
			projection='polar',
			facecolor='none')

	pax.set_theta_zero_location("N")
	pax.set_theta_direction(-1)
	pax.set_xticks(np.pi/180. * np.linspace(0,  360, 36, endpoint=False))


	ax.set_aspect(1)
	ax.axis('Off')


	# grid the data.

	Vi = griddata((x, y), np.array(continuidade), (Xi[None,:], Yi[:,None]), method='linear')	
	cf = ax.contourf(Xi,Yi,Vi, ncontour, cmap=plt.cm.jet)

	# Define gradient 

	gradient = np.linspace(1,0, 256)
	gradient = np.vstack((gradient, gradient))

	# Create matplotlib graph 


	cax = plt.axes([0.72,0.1, 0.05, 0.7])
	cax.xaxis.set_major_locator(plt.NullLocator())
	cax.yaxis.tick_right()
	cax.imshow(gradient.T, aspect='auto', cmap=plt.cm.jet)
	cax.set_yticks(np.linspace(0,256,12))
	cax.set_yticklabels(list(map(str, cf.get_array()))[::-1])

	plt.show()

##############################################################################################################################################

# EM FASE DE TESTE 

def _covariogram_map_3d(property_value, dataset, x_label, y_label, z_label, neighbors, division = 20, alpha= 0.7,  
	cutx =[-np.inf, np.inf],cuty =[-np.inf,np.inf],cutz =[-np.inf,np.inf], size =20 ):

	'''covariogram_map_3d
	
	Args:
	    property_value (TYPE): Description
	    dataset (TYPE): Description
	    x_label (TYPE): Description
	    y_label (TYPE): Description
	    z_label (TYPE): Description
	    neighbors (int): Number of neighbors using in KNearest neighbors 
	    division (int, optional): Description
	    alpha (float, optional): Description
	    cutx (list, optional): list containing the minimum cutsize and the maximum cutsize for x coordinates
	    cuty (list, optional): list containing the minimum cutsize and the maximum cutsize for y coordinates
	    cutz (list, optional): list containing the minimum cutsize and the maximum cutsize for z coordinates
	    size (int, optional): Description
	'''


	X = dataset[x_label].values
	Y = dataset[y_label].values
	Z = dataset[z_label].values
	R = dataset[property_value].values 

	max_x, min_x = max(X), min(X)
	max_y, min_y = max(Y), min(Y)
	max_z, min_z = max(Z), min(Z)

	cordinatesx = np.linspace(min_x,max_x,division)
	cordinatesy = np.linspace(min_y,max_y, division)
	cordinatesz = np.linspace(min_z,max_z,division)

	cordinates = np.array([np.array([i,j,k]).T for i in cordinatesx for j in cordinatesy for k in cordinatesz])
	estimates = np.zeros(len(cordinates))

	nb = KNeighborsRegressor(n_neighbors=neighbors).fit(np.array([X,Y,Z]).T, R)

	for i, j in zip(range(len(estimates)), cordinates): 
		estimates[i] = nb.predict(j.reshape(1, -1))[0]
		
	estimates = estimates.reshape((division,division,division))
	
	fft_u  = np.conjugate(np.fft.fftn(estimates,axes=(0,1,2), norm ='ortho'))
	fft_u_roll = np.fft.fftn(estimates,axes=(0,1,2), norm ='ortho')
	product = np.multiply(fft_u_roll,fft_u)
	Covariance = np.fft.fftshift(np.fft.ifftn(product,axes=(0,1,2), norm ='ortho'))
	Covariance = np.real(Covariance)	
	Covariance = Covariance.reshape(-1,1)[:,0]/(division*division*division)

	filter_cord = (cordinates[:,0]>cutx[0]) & (cordinates[:,0]<cutx[1]) & (cordinates[:,1]>cuty[0]) & (cordinates[:,1]<cuty[1]) &  (cordinates[:,2]>cutz[0]) & (cordinates[:,2]<cutz[1])

	cx = cordinates[:,0][filter_cord]
	cy = cordinates[:,1][filter_cord]
	cz = cordinates[:,2][filter_cord]

	Covariance = Covariance[filter_cord]

	df_varmap = pd.DataFrame(np.array([cx, cy, cz, Covariance]).T, columns=['cx','cy','cz','Covariance'])

	fig = px.scatter_3d(df_varmap, x='cx', y='cy', z='cz',
	              color='Covariance')
	fig.show()

##############################################################################################################################################

def _modelling_to_interact(**kargs):
	"""Summary
	
	Args:
	    **kargs: Args to modelling variogram interactively 
	
	Returns:
	    TYPE: Variogram model 
	"""

	# Extract model parameters 

	rotation_reference = [kargs.get('rotation_max') ,kargs.get('rotation_med'),kargs.get('rotation_min') ]
	ranges =[]
	for i in range(kargs.get('nstructures')):
		mx_rg = 'rangemax_{}'.format(str(i))
		md_rg = 'rangemed_{}'.format(str(i))
		mi_rg = 'rangemin_{}'.format(str(i))

		ranges.append([kargs.get(mx_rg), kargs.get(md_rg), kargs.get(mi_rg)])


	model_func = []
	contribution = []
	for i in range(kargs.get('nstructures')):
		mld = 'model_{}'.format(str(i))
		contr = 'contribution_{}'.format(str(i))
		model_func.append(kargs.get(mld))
		contribution.append(kargs.get(contr))
	
	nugget = kargs.get('nugget')
	inverted = kargs.get('inverted')
	azimuths = kargs.get('azimuths')
	dips = kargs.get('dips')

	# Calculate models for each experimental directions 

	dfs = []
	for j, i in enumerate(kargs.get('experimental_dataframe')):
		dfs.append(_modelling(i, azimuths[j], dips[j], rotation_reference,model_func ,ranges,contribution,nugget, inverted))

	
	# Plot graph 

	size_row = 1 if len(dfs) < 4 else int(math.ceil(len(dfs)/4))
	size_cols = 4 if len(dfs) >= 4 else int(len(dfs))

	titles = ["Azimuth {} - Dip {}".format(azimuths[j], dips[j]) for j in range(len(dfs))]
	fig = make_subplots(rows=size_row, cols=size_cols, subplot_titles=titles)

	count_row = 1
	count_cols = 1

	for j, i in enumerate(dfs):
		fig.add_trace(go.Scatter(x=kargs.get('experimental_dataframe')[j]['Average distance'], y=kargs.get('experimental_dataframe')[j]['Spatial continuity'],
                mode='markers',
                name='Experimental',
				marker= dict(color =kargs.get('experimental_dataframe')[j]['Number of pairs'] ),
				text =kargs.get('experimental_dataframe')[j]['Number of pairs'].values if kargs.get('show_pairs')== True else None) , row=count_row, col=count_cols)
		fig.add_trace(go.Scatter(x=i['distances'], y=i['model'],
                mode='lines',
                name='Model'), row=count_row, col=count_cols )
		fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols, automargin = True)
		fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols, automargin=True)
		fig.update_layout(autosize=True)

		count_cols += 1
		if count_cols > 4:
			count_cols = 1
			count_row += 1	

	fig.show()

	# Store variogram model 

	global return_model_var
	return_model_var = {'number_of_structures' : kargs.get('nstructures'),
						'rotation_reference': rotation_reference,
						'models': model_func, 
						'nugget': nugget, 
						'contribution': contribution, 
						'ranges' : ranges}

	return return_model_var

##############################################################################################################################################

def interactive_modelling(experimental_dataframe: callable, number_of_structures: int , show_pairs: bool = False):
	"""Opens the interactive modeling controlls. Store the results in a global variable `gammapy.return_model_var`.
	
	Args:
	    experimental_dataframe (DataFrame): Experimental variogram DataFrame. Assessed by `gammapy.return_exp_var`
	    number_of_structures (int): number of structures
	    show_pairs (bool, optional): show number of pairs flag. Defaults to False.

	Returns:
	    dict: Variogram model 
	"""

	warnings.filterwarnings('ignore')

	azimuths = experimental_dataframe['Directions'][0]
	dips = experimental_dataframe['Directions'][1] 


	rotation_max = widgets.FloatSlider(description='Rotação Azim', min= 0, max= 360, step=1,continuous_update=False)
	rotation_med = widgets.FloatSlider(description='Rotação Dip',min= 0, max= 360, step=1,continuous_update=False)
	rotation_min =  widgets.FloatSlider(description='Rotação Rake',min= 0, max= 360, step=1,continuous_update=False)
	nugget = widgets.FloatSlider(description='Pepita',min= 0, max= 2*max(experimental_dataframe['Values'][0]['Spatial continuity']), step=10*max(experimental_dataframe['Values'][0]['Spatial continuity'])/1000,continuous_update=False)
	inverted = widgets.Checkbox(description='Inverter a modelagem')


	u1 = widgets.HBox([rotation_max, rotation_med, rotation_min])


	grid1 = GridspecLayout(number_of_structures, 1)
	values1 = []
	for i in range(number_of_structures):
		values1.append([widgets.FloatSlider(description='Alcance máximo', min= 0, max= 2*max(experimental_dataframe['Values'][0]['Average distance']), step=max(experimental_dataframe['Values'][0]['Average distance'])/1000.0,continuous_update=False),
		 widgets.FloatSlider(description='Alcance médio', min= 0, max= 2*max(experimental_dataframe['Values'][0]['Average distance']), step=max(experimental_dataframe['Values'][0]['Average distance'])/1000.0,continuous_update=False), 
		 widgets.FloatSlider(description='Alcance mínimo',min= 0, max= 2*max(experimental_dataframe['Values'][0]['Average distance']), step=max(experimental_dataframe['Values'][0]['Average distance'])/1000.0,continuous_update=False)])
		grid1[i,0] = widgets.HBox(values1[i])

	u4 = widgets.HBox([nugget, inverted])

	values2 = []
	grid2 = GridspecLayout((number_of_structures+1), 1)
	for i in range(number_of_structures):
		values2.append([widgets.Dropdown(options= ['Spherical','Exponential','Gaussian']), 
			widgets.FloatSlider(description='Contribution',min= 0, max= 2*max(experimental_dataframe['Values'][0]['Spatial continuity']), step=max(experimental_dataframe['Values'][0]['Spatial continuity'])/1000,continuous_update=False)])
		grid2[i,0] = widgets.HBox(values2[i])
	grid2[number_of_structures,0] = u4

	children = [u1,grid1,grid2]
	accordion = widgets.Accordion(children=children)

	accordion.set_title(0, 'Rotações dos eixos principais')
	accordion.set_title(1, 'Alcance dos eixos principais ')
	accordion.set_title(2, 'Função, contribuição e efeito pepita')

	outputs = {'experimental_dataframe': fixed(experimental_dataframe['Values']),
			   'azimuths' : fixed(azimuths),
			   'dips' : fixed(dips),
			   'nstructures': fixed(number_of_structures), 
			   'rotation_max' : rotation_max,
			   'rotation_med' :rotation_med,
			   'rotation_min' : rotation_min,
			   'nugget': nugget,
			   'inverted' : inverted}
	
	for i in range(number_of_structures):
		outputs['rangemax_{}'.format(str(i))] = values1[i][0]
		outputs['rangemed_{}'.format(str(i))] = values1[i][1]
		outputs['rangemin_{}'.format(str(i))] = values1[i][2] 
		outputs['model_{}'.format(str(i))] = values2[i][0]
		outputs['contribution_{}'.format(str(i))] = values2[i][1] 
	outputs['show_pairs'] = fixed(show_pairs)
		

	out = widgets.interactive_output(_modelling_to_interact, outputs)
	display(accordion,  out)


##############################################################################################################################################	

##############################################################################################################################################

def modelling(experimental_dict: callable, rotation_max: float, rotation_med: float, rotation_min: float,
	nugget: int ,inverted: bool, rangemax: list, rangemed: list, rangemin: list, model: list, contribution:list, 
	show_pairs: bool = False):

	"""Variogram modeling function without interactive controlls.
	
	Args:
	    experimental_dict (dict): experimental variogram 
	    rotation_max (float): azimuth
	    rotation_med (float): dip
	    rotation_min (float): rake
	    nugget (float): nugget contribution
	    inverted (bool): inverted model flag
	    rangemax (lst): list of ranges in maximum continuity direction for each structure
	    rangemed (lst): list of ranges in intermediate continuity direction for each structure
	    rangemin (lst): list of ranges in minumim continuity direction for each structure
	    model (lst of str): lst of models for each structure. `Spherical`, `Exponential` or `Gaussian`.
	    contribution (lst of floats): list of contributions for each structure
	    show_pairs (bool, optional): Show pairs flag. Defaults to False.
	
	Returns:
	    dict: variogram model DataFrame
	"""

	warnings.filterwarnings('ignore')

	azimuths = experimental_dict['Directions'][0]
	dips = experimental_dict['Directions'][1]
	number_of_structures = len(rangemax)
	outputs = {'experimental_dataframe': experimental_dict['Values'],
			   'azimuths' : azimuths,
			   'dips' : dips,
			   'nstructures': number_of_structures, 
			   'rotation_max' : rotation_max,
			   'rotation_med' :rotation_med,
			   'rotation_min' : rotation_min,
			   'nugget': nugget,
			   'inverted' : inverted}
	
	for i in range(number_of_structures):
		outputs['rangemax_{}'.format(str(i))] = rangemax[i]
		outputs['rangemed_{}'.format(str(i))] = rangemed[i]
		outputs['rangemin_{}'.format(str(i))] = rangemin[i] 
		outputs['model_{}'.format(str(i))] = model[i]
		outputs['contribution_{}'.format(str(i))] = contribution[i]
	outputs['show_pairs'] = show_pairs
		

	return _modelling_to_interact(**outputs)
	


##############################################################################################################################################	

def interactive_varmap(dataset:callable, X:str, Y:str, head:str, tail:str, Z:str =None, choice:float =1.0):
	"""Opens interactive variogram map controls.
	
	Args:
	    dataset (DataFrame): Data points DataFrame
	    X (str): x column coordinates name
	    Y (str): y column cordinates name
	    head (str): head property name
	    tail (str): tail property name
	    Z (str, optional): z coordinates column name. Defaults to None.
	    choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.
	"""

	# set same seed 


	# Define the Ipython Widgets 

	warnings.filterwarnings('ignore')

	azimuth = widgets.BoundedFloatText(value=0,min = 0, max = 360, description='Azimuth:',disabled=False)
	dip = widgets.BoundedFloatText(value=0,min= 0 , max=360, description='Dip:',disabled=False)
	nlags = widgets.BoundedIntText(value=5,min= 1, max= 1000000, description='Nlags:',disabled=False)
	lagdistance = widgets.BoundedFloatText(value=7.5, min= 0.0000001, max=1000000,  description='Lag size:',disabled=False)
	ndirections = widgets.Dropdown(options=[9, 18, 36],value=18,description='Ndirections:',disabled=False)
	type_var = widgets.Dropdown(options=['Variogram', 'Covariogram', 'Correlogram', 'PairWise', 'Relative_Variogram'],
		value='Variogram',description='Number:',disabled=False)

	execute = widgets.Button(description='Execute',icon='check')
	output = widgets.Output()

	# Defining progress bar 

	max_count = 1000
	wait_please = widgets.IntProgress(min=0, max=ndirections.value) # instantiate the bar

	# Define tolerance properties according Ipython values 

	lineartolerance = lagdistance.value/2.0
	hband = lagdistance.value/2.0
	vband= lagdistance.value/2.0
	htol = 360/(ndirections.value)
	vtol = 360/(ndirections.value)

	# Creating Hbox layout 

	u1 = widgets.HBox([azimuth, dip])
	u2 = widgets.HBox([nlags, lagdistance])
	u3 = widgets.HBox([type_var, ndirections])
	u4 = widgets.HBox([execute, wait_please])

	# Creating grid layout 

	grid = GridspecLayout(4, 1)
	grid[0,0] = u1
	grid[1,0] = u2
	grid[2,0] = u3
	 

	# Creating accordion layout 

	children = [grid]
	accordion = widgets.Accordion(children=children)
	accordion.set_title(0, 'Varmap parameters')


	display(accordion, u4) # display values 

	# Calling varmap according 2D or 3D dataset 

	if Z == None:

		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		def on_varmap(change):	
			"""Summary
			
			Args:
			    change (TYPE): Description
			"""
			_varmap(azimuth.value,0, lineartolerance, htol, vtol, hband, vband, type_var.value, dataset, X, Y, 'Z', nlags.value, 
					lagdistance.value, head, tail, ndirections.value, wait_please,  choice, True)
	else: 
		def on_varmap(change):			
			"""Summary
			
			Args:
			    change (TYPE): Description
			"""
			_varmap(azimuth.value, dip.value, lineartolerance, htol, vtol, hband, vband, type_var.value, dataset, X, Y, Z, nlags.value, 
					lagdistance.value, head, tail, ndirections.value, wait_please,  choice, True)

	execute.on_click(on_varmap)


##############################################################################################################################################

def varmap(dataset: callable, X: str, Y: str, head: str, tail: str, azimuth: float,
 dip: float, nlags: int, lagdistance: float, ndirections: int, type_var: str,  Z:str =None, choice =1.0):
	"""Plots a variogram map.
	
	Args:
	    dataset (DataFrame): Point set DataFrame
	    X (str): x coordinates column name
	    Y (str): y coordinates column name
	    head (str): head variable name
	    tail (str): tail variable name
	    azimuth (float): azimuth
	    dip (float): dip
	    nlags (int): number of lags
	    lagdistance (float): lag distance
	    ndirections (int): number of directions
	    type_var (str): covariance funcrion type. `Variogram`, `Correlogram` ...
	    Z (str, optional): z coordinates variable name. Defaults to None.
	    choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.
	"""

	# set same seed 


	# Define tolerance properties according Ipython values 

	warnings.filterwarnings('ignore')

	lineartolerance = lagdistance/2.0
	hband = lagdistance/2.0
	vband= lagdistance/2.0
	htol = 360/(ndirections)
	vtol = 360/(ndirections)

	wait_please = 0 

	# Calling varmap according 2D or 3D dataset 

	if Z == None:
		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		_varmap(azimuth,0, lineartolerance, htol, vtol, hband, vband, type_var, dataset, X, Y, 'Z', nlags, 
					lagdistance, head, tail, ndirections, wait_please,  choice)
	else:
		_varmap(azimuth, dip, lineartolerance, htol, vtol, hband, vband, type_var, dataset, X, Y, Z, nlags, 
					lagdistance, head, tail, ndirections, wait_please,  choice)




##############################################################################################################################################

def interactive_experimental(dataset:callable, X:str, Y:str, head:str, tail:str, ndirections:int, show_pairs =False,  Z =None, choice =1.0):
	"""Calculates experimental variogram. Store the results in a global variable `gammapy.return_exp_var`.
	
	Args:
	    dataset (DataFrame): data points DataFrame
	    X (str): x coordinates column name
	    Y (str): y coordinates column name
	    head (str): head variable
	    tail (str): tail variable
	    ndirections (int): number of directions
	    show_pairs (bool, optional): show number of pairs flag. Defaults to False.
	    Z (str, optional): z coordinates column name. Defaults to None.
	    choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.

	Returns:
	    dict: experimental variogram DataFrame
	"""

	warnings.filterwarnings('ignore') # Ignore numpy erros
	
	# If dataset 2D, fill Z values with zeros 

	if Z == None:
		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		Z = 'Z'


	# Creating Ipython Widgets			

	widgets_l = []
	vboxes = []
	for i in range(ndirections):
		widgets_l.append([widgets.Dropdown(options=['Variogram', 'Covariogram', 'Correlogram','PairWise', 'RelativeVariogram' ],value='Variogram',description='Type:',disabled=False),
			widgets.BoundedFloatText(value=0,min = 0, max = 360, description='Azimuth:',disabled=False),
			widgets.BoundedFloatText(value=0,min = 0, max = 360, description='Dip:',disabled=False), 
			widgets.BoundedIntText(value=10,min = 1, max = 10000, description='nlags:',disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 1000000000, description='lagsize:',disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 1000000000, description='lineartol:',disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 90, description='htol:', disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 90, description='vtol:', disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 90, description='hband:', disabled=False),
			widgets.BoundedFloatText(value=10,min = 0.000000001, max = 1000000000, description='vband:', disabled=False),
			widgets.Dropdown(options=[True, False ],value=False,description='Omni:',disabled=False)])
		vboxes.append(widgets.VBox(widgets_l[i], layout=widgets.Layout(width='370px')))



	execute = widgets.Button(description='Execute',icon='check')
	output = widgets.Output()

	# Defining progress bar 

	max_count = 1000
	wait_please = widgets.IntProgress(min=0, max=ndirections) # instantiate the bar

	# More boxes and layouts 

	grid = widgets.HBox(vboxes, layout=widgets.Layout(width='5000px'))
	box2 = widgets.HBox([execute, wait_please], display='flex',align_items='stretch',width='50%')
	box3 = widgets.VBox([grid, box2], display='flex',align_items='stretch',width='50%')

	children = [box3]
	accordion = widgets.Accordion(children=children)
	accordion.set_title(0, 'Experimental parameters')


	display(accordion)

	# Define variograms with change 

	returning = []

	def on_variogram(change):

		# fill parameters

		type_c, nlags, lagsize, lineartol,htol, vtol, hband, vband, azimuth, dip, omni = [], [], [], [], [], [] , [] ,[] ,[] ,[], []
		for i in range(ndirections):
			type_c.append(widgets_l[i][0].value)
			azimuth.append(widgets_l[i][1].value)
			dip.append(widgets_l[i][2].value)
			nlags.append(widgets_l[i][3].value)
			lagsize.append(widgets_l[i][4].value)
			lineartol.append(widgets_l[i][5].value)
			htol.append(widgets_l[i][6].value)
			vtol.append(widgets_l[i][7].value)
			hband.append(widgets_l[i][8].value)
			vband.append(widgets_l[i][9].value)
			omni.append(widgets_l[i][10].value)

		# Calculate experimental variograms 

		dfs = []
		for i in range(ndirections):
			wait_please.value += 1
			dfs.append(_calculate_experimental_function(dataset, X, Y, Z, 
						type_c[i], lagsize[i], lineartol[i], htol[i], vtol[i], hband[i], vband[i], azimuth[i], dip[i], 
						nlags[i], head, tail, choice, omni[i]))

		# Store experimental variograms 

		global return_exp_var
		returning = {'Directions': [azimuth, dip],
					 'Values' : dfs}
		return_exp_var = returning

		# Plot variograms 
	
		size_row = 1 if len(dfs) < 4 else int(math.ceil(len(dfs)/4))
		size_cols = 4 if len(dfs) >= 4 else int(len(dfs))

		titles = ["Azimuth {} - Dip {}".format(azimuth[j], dip[j]) if omni[j]==False else 'Omni' for j in range(len(dfs))]
		fig = make_subplots(rows=size_row, cols=size_cols, subplot_titles=titles)

		count_row = 1
		count_cols = 1


		for j, i in enumerate(dfs):

			fig.add_trace(go.Scatter(x=dfs[j]['Average distance'], y=dfs[j]['Spatial continuity'],
	                mode='markers',
	                name='Experimental' ,
					marker= dict(color =dfs[j]['Number of pairs']),
					text =dfs[j]['Number of pairs'].values if show_pairs== True else None,
					textposition='bottom center') , row=count_row, col=count_cols)
			fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols, automargin = True)
			fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols, automargin=True)
			fig.update_layout(autosize=True)

			count_cols += 1
			if count_cols > 4:
				count_cols = 1
				count_row += 1
		fig.show()
	execute.on_click(on_variogram)

##############################################################################################################################################	

##############################################################################################################################################

def experimental(dataset:callable, X:str, Y:str, head:str, tail:str,type_c:str,
 nlags:list , lagsize:list, lineartol:list ,htol: list, vtol: list, hband: list, vband:list, azimuth:list, dip:list, omni:bool, 
 show_pairs:bool =False,  Z:str =None, choice:float =1.0):
	"""Calculates experimental variogram. Store the results in a global variable `gammapy.return_exp_var`.
	
	Args:
	    dataset (DataFrame): point set DataFrame
	    X (str): x coordinates column name
	    Y (str): y coordinates column name
	    head (str): head variable name
	    tail (str): tail variable name
	    type_c (lst of str): covariance funcrion type list. `Variogram`, `Correlogram` ...
	    nlags (lst): list of lag number for each direction
	    lagsize (lst): list of lag size for each direction
	    lineartol (lst): list of linear tolerance for each direction
	    htol (lst): list of horizontal tolerance for each direction
	    vtol (lst): liist of vertical tolerance for each direction
	    hband (lst): list of horizontal band for each direction
	    vband (lst): list of vertical band for each direction
	    azimuth (lst): azimuth
	    dip (lst): dip
	    omni (bool): omnidirectionl flag
	    show_pairs (bool, optional): show pairs flag. Defaults to False.
	    Z (str, optional): z coordinates column name. Defaults to None.
	    choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.
	
	Returns:
	    dict: experimental variogram DataFrame
	"""


	# Ignore numpy erros 

	warnings.filterwarnings('ignore')

	# define the number of directions in experimental variogram 

	ndirections = len(nlags)

	# Fill dataset with 0 z coordinates if 2D case 

	if Z == None:
		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		Z = 'Z'	

	# Return the dictionary containing all information of experimental variograms 		

	dfs = []
	returning = {'Directions': [azimuth, dip],
				 'Values' : dfs}

	# Calculate experimental variograms for each directions 

	for i in range(ndirections):
		dfs.append(_calculate_experimental_function(dataset, X, Y, Z, 
					type_c[i], lagsize[i], lineartol[i], htol[i], vtol[i], hband[i], vband[i], azimuth[i], dip[i], 
					nlags[i], head, tail, choice, omni[i]))

	# Plot experimental variograms 

	size_row = 1 if len(dfs) < 4 else int(math.ceil(len(dfs)/4))
	size_cols = 4 if len(dfs) >= 4 else int(len(dfs))

	titles = ["Azimuth {} - Dip {}".format(azimuth[j], dip[j]) if omni[j]==False else 'Omni' for j in range(len(dfs))]
	fig = make_subplots(rows=size_row, cols=size_cols, subplot_titles=titles)

	count_row = 1
	count_cols = 1

	for j, i in enumerate(dfs):

		fig.add_trace(go.Scatter(x=dfs[j]['Average distance'], y=dfs[j]['Spatial continuity'],
                mode='markers',
                name='Experimental' ,
				marker= dict(color =dfs[j]['Number of pairs']),
				text =dfs[j]['Number of pairs'].values if show_pairs== True else None,
				textposition='bottom center') , row=count_row, col=count_cols)
		fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols, automargin = True)
		fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols, automargin=True)
		fig.update_layout(autosize=True)

		count_cols += 1
		if count_cols > 4:
			count_cols = 1
			count_row += 1
	fig.show()

	return returning


##############################################################################################################################################	
	

def hscatterplots (dataset, X:str, Y:str, head:str, tail:str, lagsize: float, nlags: int, 
					azimuth: float, dip: float, lineartol: float,htol: float, vtol: float, hband: float, vband: float , Z:str= None, choice:float=1.0, figsize=(700,700)):
	"""Plots H-scatterplot for n lags
	
	Args:
		dataset (pd.DataFrame): Pandas DataFrame
		X (str): x coordinates column name
		Y (str): y coordinates column name
		head (str): head variable name
		tail (str): tail variable name
		lagsize (float): lag size
		nlags (int): number of lags
		azimuth (float): azimuth 
		dip (float): dip
		lineartol (float): linear tolerance
		htol (float): angular horizontal tolerance in degrees
		vtol (float): angular vertical tolerance in degrees
		hband (float): horinztal bandwidth
		vband (float): vertical bandwidth
		Z (str, optional): z coordinates column name. Defaults to None.
		choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.	"""

	# If dataset 2D, fill Z values with zeros 

	warnings.filterwarnings('ignore')

	if Z == None:
		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		Z = 'Z'

	# Calculate all permissible distances

	dist = _distances(dataset, nlags, X, Y, Z, lagsize, head, tail, choice)

	# Define the angles and tolerances 

	cos_Azimuth = np.cos(np.radians(90-azimuth))
	sin_Azimuth = np.sin(np.radians(90-azimuth))
	cos_Dip     = np.cos(np.radians(90-dip))
	sin_Dip     = np.sin(np.radians(90-dip))
	htol = np.abs(np.cos(np.radians(htol)))
	vtol= np.abs(np.cos(np.radians(vtol)))
	check_azimuth = np.abs((dist[:,0]*cos_Azimuth + dist[:,1]*sin_Azimuth)/dist[:,3])
	check_dip     = np.abs((dist[:,3]*sin_Dip + dist[:,2]*cos_Dip)/dist[:,4])
	check_bandh   = np.abs(cos_Azimuth*dist[:,1]- sin_Azimuth*dist[:,0])
	check_bandv	  = np.abs(sin_Dip*dist[:,2] - cos_Dip*dist[:,3])
	
	store = []
	lagmultiply = []
	for i in range(1,(nlags+1)):
		lagmultiply.append(i)
		points = __permissible_pairs (i, lagsize, lineartol, check_azimuth,
						check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, False,
						dist)
		average_tail = (points[:,7] +  points[:,8])/2
		average_head = (points[:,5] +  points[:,6])/2
		store.append([average_head, average_tail])

	plots.plot_hscat(store, lagsize, lagmultiply, figsize)