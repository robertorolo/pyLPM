
# IMPORT PACKAGES # 


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

# INITIALIZE OFFLINE NOTEBOOK MODE
pyo.init_notebook_mode()

##############################################################################################################################################

# global variables  
global return_exp_var
return_exp_var = []

global return_model_var
return_model_var = {}

##############################################################################################################################################

@numba.jit(fastmath=True)
def _hdist(distancex, distancey, distancez):
	dist =np.zeros(distancex.shape[0])
	for i in range(distancex.shape[0]):
		dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2 + distancez[i]**2) + 0.0000000001
	return dist

##############################################################################################################################################

@numba.jit(fastmath=True)
def _xydist(distancex, distancey):
	dist =np.zeros(distancex.shape[0])
	for i in range(distancex.shape[0]):
		dist[i] = np.sqrt(distancex[i]**2 + distancey[i]**2 ) + 0.0000000001
	return dist

##############################################################################################################################################

@numba.jit(fastmath=True)
def _xdist(pairs):
	dist =np.zeros(pairs.shape[0])

	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][0] - pairs[i][1][0])
	return dist

##############################################################################################################################################

@numba.jit(fastmath=True)
def _ydist(pairs):
	dist =np.zeros(pairs.shape[0])
	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][1] - pairs[i][1][1])
	return dist

##############################################################################################################################################

@numba.jit(fastmath=True)
def _zdist(pairs):
	dist =np.zeros(pairs.shape[0])
	for i in range(pairs.shape[0]):
		dist[i] = (pairs[i][0][2] - pairs[i][1][2])
	return dist

##############################################################################################################################################

@numba.jit(fastmath=True)
def _combin(points,n, max_dist):
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

@numba.jit(fastmath=True)
def _rotate_data(xh, yh, zh, azimute, dip):

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
			 

def _distances(dataset, nlags, x_label, y_label, z_label, 
	lagdistance, head_property, tail_property, choice):

	'''distances
	Returns:	
	 distance_dataframe (pandas.DataFrame): Pandas Dataframe containing all the distance metrics
	 DX (pandas.DataFrame.Series) : Difference of x cartesian coordinates 
	 DY (pandas.DataFrame.Series) : = diference of y cartesian values from the head and tails of the vector  
	 DZ (pandas.DataFrame.Series) : = diference of z cartesian values from the head and tails of the vector 
	 XY (pandas.DataFrame.Series) : = Distance projection on XY plane of the vector  
	 H  (pandas.DataFrame.Series) : = Distance value from head and tail of vector  
	 Var 1 (head) (pandas.DataFrame.Series) : Value from variable 1 on the head of vector 
	 Var 2 (head) (pandas.DataFrame.Series) : Value from variable 2 on the head of vector  
	 Var 1 (tail) (pandas.DataFrame.Series) : Value from variable 1 on the tail of vector 
	 Var 2 (tail) (pandas.DataFrame.Series) : Value form variable 2 on the tail of vector 
	 INDEX HEAD   (pandas.DataFrame.Series) : Index of propertie 1 sample 
	 INDEX TAIL   (pandas.DataFrame.Series) : Index of propertie 2 sample
	'''
	max_dist = (nlags + 1)*lagdistance
	X = dataset[x_label].values
	Y = dataset[y_label].values
	Z = dataset[z_label].values
	HEAD = dataset[head_property].values
	TAIL = dataset[tail_property].values
	if (choice != 1.0):
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
		points = np.array([points[np.random.randint(0,len(points))] for i in range(int(points.shape[0]*choice))])
	else:
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
	pairs = _combin(points,points.shape[0], max_dist)
	distancex = _xdist(pairs) 
	distancey =  _ydist(pairs) 
	distancez =  _zdist(pairs)
	distanceh = _hdist(distancex, distancey, distancez)
	distancexy = _xydist(distancex, distancey)
	head_1 = np.array([pair[0][3] for pair in pairs])
	head_2 = np.array([pair[0][4] for pair in pairs])
	tail_1 = np.array([pair[1][3] for pair in pairs])
	tail_2 = np.array([pair[1][4] for pair in pairs])
	distance_dataframe =  np.array([distancex, 
					distancey, 
					distancez, 
					distancexy, 
					distanceh, 
					head_1,
					head_2,
					tail_1,
					tail_2,]).T
	return distance_dataframe[distanceh[:,] < max_dist ]

##############################################################################################################################################

def _distances_varmap(dataset, X,Y,Z, nlags, lagdistance, head_property, tail_property,
					choice):

	'''distances
	Returns:	
	 distance_dataframe (pandas.DataFrame): Pandas Dataframe containing all the distance metrics
	 DX (pandas.DataFrame.Series) : Difference of x cartesian coordinates 
	 DY (pandas.DataFrame.Series) : = diference of y cartesian values from the head and tails of the vector  
	 DZ (pandas.DataFrame.Series) : = diference of z cartesian values from the head and tails of the vector 
	 XY (pandas.DataFrame.Series) : = Distance projection on XY plane of the vector  
	 H  (pandas.DataFrame.Series) : = Distance value from head and tail of vector  
	 Var 1 (head) (pandas.DataFrame.Series) : Value from variable 1 on the head of vector 
	 Var 2 (head) (pandas.DataFrame.Series) : Value from variable 2 on the head of vector  
	 Var 1 (tail) (pandas.DataFrame.Series) : Value from variable 1 on the tail of vector 
	 Var 2 (tail) (pandas.DataFrame.Series) : Value form variable 2 on the tail of vector 
	 INDEX HEAD   (pandas.DataFrame.Series) : Index of propertie 1 sample 
	 INDEX TAIL   (pandas.DataFrame.Series) : Index of propertie 2 sample
	'''

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	max_dist = (nlags + 1)*lagdistance
	HEAD = dataset[head_property].values
	TAIL = dataset[tail_property].values
	if (choice != 1.0):
		np.random.seed(0)
		np.random.seed(138276)
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
		points = np.array([points[np.random.randint(0,len(points))] for i in range(int(points.shape[0]*choice))])
	else:
		points = np.array(list(zip(X,Y,Z,HEAD,TAIL)))
	pairs = _combin(points,points.shape[0], max_dist)
	distancex = _xdist(pairs) 
	distancey =  _ydist(pairs) 
	distancez =  _zdist(pairs)
	distanceh = _hdist(distancex, distancey, distancez)
	distancexy = _xydist(distancex, distancey)
	head_1 = np.array([pair[0][3] for pair in pairs])
	head_2 = np.array([pair[0][4] for pair in pairs])
	tail_1 = np.array([pair[1][3] for pair in pairs])
	tail_2 = np.array([pair[1][4] for pair in pairs])
	distance_dataframe =  np.array([distancex, 
					distancey, 
					distancez, 
					distancexy, 
					distanceh, 
					head_1,
					head_2,
					tail_1,
					tail_2,]).T
	return distance_dataframe[distanceh[:,] < max_dist ]

##############################################################################################################################################

def __permissible_pairs (lag_multiply, lagdistance, lineartolerance, check_azimuth,
						check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,
						dist):

	'''permissible_pairs
	Args:
	 lag_multiply (double): Mutliplemaxi of lag distance
	Returns:	
	 distances (numpy array): Returns the permissible sample pairs for omnidirecional functions
	'''
	minimum_range = lag_multiply*lagdistance - lineartolerance
	maximum_range = lag_multiply*lagdistance + lineartolerance
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
	 lag_multiply (double): Mutliple of lag distance	
	 type_var (string): String containing the type of spatial continuity function to calculate
	 					5 admissible functions are possible:
						"Variogram"
		 				"Covariogram"
		 				"Correlogram"
		 				"PairWise"
		 				"RelativeVariogram"
	Returns:
	 value (double): Experimental continuity function value 
	 number_of_pairs (int) : number of pairs used to calculate the experimental function 
	 average_distace (double): average distance of experimental continuity function value  
	'''

	points = __permissible_pairs(lag_multiply, lagdistance, lineartolerance, check_azimuth,
						check_dip, check_bandh, check_bandv, htol, vtol, hband, vband, omni,
						dist)
	
	if type_var not in['Variogram','Covariogram','Correlogram','PairWise','Relative_Variogram']:
		raise Exception("Experimental continuity function not in admissible functions")

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
	 plot_graph (bool) = Boolean for selecting plotting experimental values 
	 show_pairs (bool) = Boolean for selecting plotting experimental number of pairs 
	 type_var (string): String containing the type of spatial continuity function to calculate
	 					5 admissible functions are possible:
						"Variogram"
		 				"Covariogram"
		 				"Correlogram"
		 				"PairWise"
		 				"RelativeVariogram" 
	Returns:
	 df (pandas.DataFrame): Pandas Dataframe containing the experimental continuity functions of all lags
	'''

	dist = _distances(dataset, nlags, x_label, y_label, z_label, lagdistance, head_property, tail_property, choice)
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
	 plot_graph (bool) = Boolean for selecting plotting experimental values 
	 show_pairs (bool) = Boolean for selecting plotting experimental number of pairs 
	 type_var (string): String containing the type of spatial continuity function to calculate
	 					5 admissible functions are possible:
						"Variogram"
		 				"Covariogram"
		 				"Correlogram"
		 				"PairWise"
		 				"RelativeVariogram" 
	Returns:
	 df (pandas.DataFrame): Pandas Dataframe containing the experimental continuity functions of all lags
	'''

	dist = _distances_varmap(dataset, X_rot, Y_rot,Z_rot, nlags, lagdistance, head_property, tail_property,choice)
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
	 experimental dataframe (pandas.DataFrame): Pandas DataFrame containing experimental continuity functions 	
	 rotation reference (list(azimuth, dip, rake)): List containing the reference of principal directions angles in degrees 
	 model_func(list(string)) : List containing the models for all structures. size of the list must be the same of the number of structures
	 							3 admissible functions are possible:
	 							"Spherical"
								"Gaussian"
								"Exponential"
	 ranges(list(list(maximum range, medium range, minimum range))) : list of lists containing the maximum, medium and minimum range for each number of structures
	 contribution (list): list of contributions for each strucutre 
	 nugget (double): Nugget effect value 
	 inverted (bool): If true plot model according covariogram form, otherwise plot model according the variogram form
	 plot_graph (bool): If true plot the experimental variogram and the spatial continuity model
			
	Returns:
	 plot (matplotlib.pyplot): Plot of omnidirecional experimental continuity function and the spatial continuity model for one direction 
	'''

	if len(ranges[0]) != 3:
		raise ValueError("Variogram ranges must range 3 principal directions")
	if len(ranges) == len(contribution) and len(ranges) == len(model_func):
		pass
	else:
		raise ValueError("Variogram structures must be the same size")


	y = math.cos(math.radians(dips))*math.cos(math.radians(azimuths))
	x = math.cos(math.radians(dips))*math.sin(math.radians(azimuths))  
	z = math.sin(math.radians(dips))

	angle_azimuth = math.radians(rotation_reference[0])
	angle_dip = math.radians(rotation_reference[1])
	angle_rake = math.radians(rotation_reference[2])


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

	rotated_range =[]

	for i in ranges:
		rangex = float(i[1])
		rangey = float(i[0])
		rangez = float(i[2])


		rotated = (np.multiply(rot3, np.array([rangex, rangey, rangez]).T))
		rotated_range.append(math.sqrt(rotated[0]**2+rotated[1]**2+rotated[2]**2))

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
					lagdistance, head_property, tail_property, ndirections, wait_please, choice = 5000):

	X = dataset[x_label].values
	Y = dataset[y_label].values
	Z = dataset[z_label].values

	wait_please.value = 0

	Xrot, Yrot, Zrot = _rotate_data(X, Y, Z, azimuth, dip)

	lag_adm = []
	azimute_adm = []
	continuidade = []
	for i in np.arange(0,360,ndirections):

		wait_please.value += 1
		df_exp = _calculate_experimental_function_varmap(type_var,  lineartolerance, htol, vtol, hband, vband, 
					i, 0,dataset, nlags, Xrot, Yrot,Zrot,lagdistance, head_property, tail_property, choice)
		var_val = df_exp['Spatial continuity'].values
		continuidade = np.append(continuidade,var_val)
		values_1 = df_exp['Average distance'].values
		lag_adm = np.append(lag_adm,values_1)
		values_2 = np.array([np.radians(i) for j in range(df_exp.shape[0])])
		azimute_adm = np.append(azimute_adm, values_2)


	gdiscrete = 40
	ncontour = 40

	x = np.array(lag_adm)*np.sin(azimute_adm)
	y = np.array(lag_adm)*np.cos(azimute_adm)

	maximo = max([max(x), max(y)])
	

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

	gradient = np.linspace(1,0, 256)
	gradient = np.vstack((gradient, gradient))


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
	property_value(string): String containing the property to create the covariogram map
	neighbors (int) : Number of neighbors using in KNearest neighbors 
	division(int, optional): discretize number of covariogram map
	size(int, optional): size of bullet
	alpha(float, optional): the level of transparency (0- transparent, 1-solid)
	cutx (list, optional): list containing the minimum cutsize and the maximum cutsize for x coordinates
	cuty (list, optional): list containing the minimum cutsize and the maximum cutsize for y coordinates
	cutz (list, optional): list containing the minimum cutsize and the maximum cutsize for z coordinates
					
	Returns:
	plot (matplotlib.pyplot): Plot of Covariance map in three dimensional scale 
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


	dfs = []
	for j, i in enumerate(kargs.get('experimental_dataframe')):
		dfs.append(_modelling(i, azimuths[j], dips[j], rotation_reference,model_func ,ranges,contribution,nugget, inverted))

	
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
		fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols)
		fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols)
		fig.update_layout(margin=dict(l=5, r=40, t=20, b=20))

		count_cols += 1
		if count_cols > 4:
			count_cols = 1
			count_row += 1	

	fig.show()

	global return_model_var
	return_model_var = {'number_of_structures' : kargs.get('nstructures'),
						'rotation_reference': rotation_reference,
						'models': model_func, 
						'nugget': nugget, 
						'contribution': contribution, 
						'ranges' : ranges}

##############################################################################################################################################

def interactive_modelling(experimental_dataframe, directions, number_of_structures, show_pairs = False):
	"""Opens the interactive modeling controlls. Store the results in a global variable `gammapy.return_model_var`.
	
	Args:
		experimental_dataframe (DataFrame): Experimental variogram DataFrame. Assessed by `gammapy.return_exp_var`
		directions (lst): directions to model variogram
		number_of_structures (int): number of structures
		show_pairs (bool, optional): show number of pairs flag. Defaults to False.
	"""



	azimuths = directions[0]
	dips = directions[1] 


	rotation_max = widgets.FloatSlider(description='Rotação Azim', min= 0, max= 360, step=1,continuous_update=False)
	rotation_med = widgets.FloatSlider(description='Rotação Dip',min= 0, max= 360, step=1,continuous_update=False)
	rotation_min =  widgets.FloatSlider(description='Rotação Rake',min= 0, max= 360, step=1,continuous_update=False)
	nugget = widgets.FloatSlider(description='Pepita',min= 0, max= 2*max(experimental_dataframe[0]['Spatial continuity']), step=10*max(experimental_dataframe[0]['Spatial continuity'])/1000,continuous_update=False)
	inverted = widgets.Checkbox(description='Inverter a modelagem')


	u1 = widgets.HBox([rotation_max, rotation_med, rotation_min])


	grid1 = GridspecLayout(number_of_structures, 1)
	values1 = []
	for i in range(number_of_structures):
		values1.append([widgets.FloatSlider(description='Alcance máximo', min= 0, max= 2*max(experimental_dataframe[0]['Average distance']), step=max(experimental_dataframe[0]['Average distance'])/1000.0,continuous_update=False),
		 widgets.FloatSlider(description='Alcance médio', min= 0, max= 2*max(experimental_dataframe[0]['Average distance']), step=max(experimental_dataframe[0]['Average distance'])/1000.0,continuous_update=False), 
		 widgets.FloatSlider(description='Alcance mínimo',min= 0, max= 2*max(experimental_dataframe[0]['Average distance']), step=max(experimental_dataframe[0]['Average distance'])/1000.0,continuous_update=False)])
		grid1[i,0] = widgets.HBox(values1[i])

	u4 = widgets.HBox([nugget, inverted])

	values2 = []
	grid2 = GridspecLayout((number_of_structures+1), 1)
	for i in range(number_of_structures):
		values2.append([widgets.Dropdown(options= ['Spherical','Exponential','Gaussian']), 
			widgets.FloatSlider(description='Contribution',min= 0, max= 2*max(experimental_dataframe[0]['Spatial continuity']), step=max(experimental_dataframe[0]['Spatial continuity'])/1000,continuous_update=False)])
		grid2[i,0] = widgets.HBox(values2[i])
	grid2[number_of_structures,0] = u4

	children = [u1,grid1,grid2]
	accordion = widgets.Accordion(children=children)

	accordion.set_title(0, 'Rotações dos eixos principais')
	accordion.set_title(1, 'Alcance dos eixos principais ')
	accordion.set_title(2, 'Função, contribuição e efeito pepita')

	outputs = {'experimental_dataframe': fixed(experimental_dataframe),
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

def interactive_varmap(dataset, X, Y, head, tail, Z =None, choice =1.0):
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
			_varmap(azimuth.value,0, lineartolerance, htol, vtol, hband, vband, type_var.value, dataset, X, Y, 'Z', nlags.value, 
					lagdistance.value, head, tail, ndirections.value, wait_please,  choice)
	else: 
		def on_varmap(change):			
			_varmap(azimuth.value, dip.value, lineartolerance, htol, vtol, hband, vband, type_var.value, dataset, X, Y, Z, nlags.value, 
					lagdistance.value, head, tail, ndirections.value, wait_please,  choice)

	execute.on_click(on_varmap)


##############################################################################################################################################

def interactive_experimental(dataset, X, Y, head, tail, ndirections, show_pairs =False,  Z =None, choice =1.0):
	"""Calculates experimental variogram. Store the results in a global variable `gammapy.return_exp_var`.
	
	Args:
		dataset (DataFrame): data points DataFrame
		X (str): x coorcinates column name
		Y (str): y coordinates column name
		head (str): head variable
		tail (str): tail variable
		ndirections (int): number of directions
		show_pairs (bool, optional): show number of pairs flag. Defaults to False.
		Z (str, optional): z coordinates column name. Defaults to None.
		choice (float, optional): pool a random number of data to calculate the variogram. Defaults to 1.0.
	"""

	global return_values_from_exp_var 

	if Z == None:
		dataset['Z'] = np.zeros(dataset[X].values.shape[0])
		Z = 'Z'			

	widgets_l = []
	hboxes = []
	for i in range(ndirections):
		widgets_l.append([widgets.Dropdown(options=['Variogram', 'Covariogram', 'Correlogram','PairWise', 'RelativeVariogram' ],value='Variogram',description='Type:',disabled=False),
			widgets.BoundedFloatText(value=0,min = 0, max = 360, description='Azimuth:',disabled=False),
			widgets.BoundedFloatText(value=0,min = 0, max = 360, description='Dip:',disabled=False), 
			widgets.BoundedIntText(value=1,min = 1, max = 10000, description='nlags:',disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 1000000000, description='lagsize:',disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 1000000000, description='lineartol:',disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 90, description='htol:', disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 90, description='vtol:', disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 90, description='hband:', disabled=False),
			widgets.BoundedFloatText(value=0.000000001,min = 0.000000001, max = 1000000000, description='vband:', disabled=False),
			widgets.Dropdown(options=[True, False ],value=False,description='Omni:',disabled=False)])
		hboxes.append(widgets.HBox(widgets_l[i]))



	execute = widgets.Button(description='Execute',icon='check')
	output = widgets.Output()

	# Defining progress bar 

	max_count = 1000
	wait_please = widgets.IntProgress(min=0, max=ndirections) # instantiate the bar



	grid = GridspecLayout(ndirections+1, 1)
	for i in range(ndirections):
		grid[i,0] = hboxes[i]
	grid[ndirections,0] = widgets.HBox([execute, wait_please])

	children = [grid]
	accordion = widgets.Accordion(children=children)
	accordion.set_title(0, 'Experimental parameters')


	display(accordion)

	returning = []

	def on_variogram(change):


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


		dfs = []

		for i in range(ndirections):
			wait_please.value += 1
			dfs.append(_calculate_experimental_function(dataset, X, Y, Z, 
						type_c[i], lagsize[i], lineartol[i], htol[i], vtol[i], hband[i], vband[i], azimuth[i], dip[i], 
						nlags[i], head, tail, choice, omni[i]))

		global return_exp_var
		return_exp_var = dfs

	
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
			fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols)
			fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols)
			fig.update_layout(margin=dict(l=5, r=40, t=20, b=20))

			count_cols += 1
			if count_cols > 4:
				count_cols = 1
				count_row += 1
		fig.show()
	execute.on_click(on_variogram)

##############################################################################################################################################	
	



