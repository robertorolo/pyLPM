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

import subprocess
import numpy as np
import pandas as pd
from pyLPM import plots
import pkg_resources
import os
import re
import math

#############################################################################################################

#creating a global tem variable
global temp_dir_str
temp_dir_str = os.getcwd() + '\\pyLPM_data\\'
temp_dir_str = temp_dir_str.replace('\\','/')

#creating dir
if not os.path.exists(temp_dir_str): 
		os.mkdir(temp_dir_str)

#defining gslib90 folder inside package folder
global DATA_PATH
DATA_PATH = pkg_resources.resource_filename('pyLPM', 'gslib90/')

def call_program(program, parfile, usewine=False):
		"""Run a GSLib program
		
		Args:
		    program (str): gslib program path
		    parfile (str): parameter file file path
		    usewine (bool, optional): use wine flag. Defaults to False.
		"""
		if usewine == True:
				p = subprocess.Popen(['wine', program, parfile], stdout=subprocess.PIPE)
		else:
				 p = subprocess.Popen([program, parfile], stdout=subprocess.PIPE)

		for line in p.stdout:    
				print(line.decode('utf-8'), end='')

def write_GeoEAS(df,dh,x,y,z,vars=[]):
		"""Write GeoEAS file from a DataFrame
		
		Args:
		    df (DataFrame): data DataFrame
		    dh (str): dh column name
		    x (str): x column name
		    y (str): y column name
		    z (str): z column name
		    vars (list, optional): list of string with variables names. Defaults to [].
		"""

		df.replace(float('nan'),-999,inplace=True)
		columns = []
		if dh != None:
				columns.append(dh)
		columns.append(x)
		columns.append(y)
		if z != None:
				columns.append(z)
		for var in vars:
				columns.append(var)
		data = ""
		data= data +'tmp_pygslib_file\n'
		data= data + str(len(columns))+'\n'
		for col in columns:
				data=data + col+'\n'
		values = df[columns].to_string(header=False, index=False)
		data= data + values+'\n'
		f = open(temp_dir_str+'tmp.dat', 'w')
		f.write(data)
		f.close()

def read_GeoEAS(file):
		"""Read GeoEAS file into a DataFrame
		
		Args:
		    file (str): file path
		
		Returns:
		    DataFrame: DataFrame with data
		"""
		f = open(file, 'r')
		col_names = []
		results = []
		
		for index, line in enumerate(f):
			if index == 0:
				continue
			elif index == 1:
				n_cols = int(line.split()[0])
			elif index <= n_cols+1:
				col_names.append(line[:-1])
			else:
				values = [float(i) for i in line.split()]
				results.append(values)

		f.close()

		df = pd.DataFrame(results, columns = col_names)
		df.replace(-999, np.nan, inplace=True)

		return df
		#return pd.DataFrame(results, columns = col_names)

def col_number(file, col):
		"""Returns the column number from ist name on a GeoEAS file
		
		Args:
		    file (str): file path
		    col (str): column name
		
		Returns:
		    int: column number
		"""
		f = open(file, 'r')
		col_names = []
		
		for index, line in enumerate(f):
			if index == 0:
				continue
			elif index == 1:
				n_cols = int(line)
			elif index <= n_cols+1:
				col_names.append(line[:-1])

		return col_names.index(col) + 1 if col is not None else 0

def write_varg_str(varg):
		"""Summary
		
		Args:
		    varg (TYPE): Description
		
		Returns:
		    TYPE: Description
		"""
		varg_str = '{} {} \n'.format(varg['number_of_structures'], varg['nugget'])
		
		for struct in range(varg['number_of_structures']):
				
				if varg['models'][struct] is 'Spherical':
								it = 1
				elif varg['models'][struct] is 'Exponetial':
								it = 2
				elif varg['models'][struct] is 'Gaussian':
								it = 3
				
				new_line = '{} {} {} {} {} \n {} {} {} \n'.format(it, varg['contribution'][struct], varg['rotation_reference'][0], varg['rotation_reference'][1], varg['rotation_reference'][2], varg['ranges'][struct][0], varg['ranges'][struct][1], varg['ranges'][struct][2])

				varg_str = varg_str + new_line

		return varg_str

#############################################################################################################

def declus(df, x, y, z, var, tmin=-1.0e21, tmax=1.0e21, x_anis=1, z_anis=1, n_cell=10, min_size=1, max_size=20, keep_min = True, number_offsets=4, usewine=False):
		"""cell declustering algortihm. This function shows the declustering reults summary and writes weights to the DataFrame.
		
		Args:
		    df (DataFrame): points data DataFrame
		    x (str): x coordinates column name
		    y (str): y coordinates column name
		    z (str): z coordinates column name
		    var (str): variable column name
		    tmin (float, optional): minimum triming limit. Defaults to -1.0e21.
		    tmax (float, optional): maximum triming limit. Defaults to 1.0e21.
		    x_anis (float, optional): the anisotropy factors to consider rectangular cells. The cell size in the x direction is multiplied by these factors to get the cell size in the y and z directions, e.g., if a cell size of 10 is being considered and anisy2 and anisz3 then the cell size in the y direction is 20 and the cell size in the z direction is 30.. Defaults to 1.
		    z_anis (float, optional): anisotropy factor. Defaults to 1.
		    n_cell (int, optional): number of cells. Defaults to 10.
		    min_size (float, optional): minimum size. Defaults to 1.
		    max_size (float, optional): maximum size. Defaults to 20.
		    keep_min (bool, optional): an boolean flag that specifies whether a minimum mean value (True) or maximum mean value (False) is being looked for. Defaults to True.
		    number_offsets (int, optional): the number of origin offsets. Each of the ncell cell sizes are considered with noff different original starting points. This avoids erratic results caused by extreme values falling into specific cells. A good number is 4 in 2-D and 8 in 3-D. A short description of the program. Defaults to 4.
		    usewine (bool, optional): use wine flag. Defaults to False.
		"""

		write_GeoEAS(df=df,dh=None,x=x,y=y,z=z,vars=[var])
		
		decluspar = '''
										Parameters for CELLDECLUS
																	Parameters for DECLUS
									*********************

START OF PARAMETERS:
{datafile}         -file with data
{x}   {y}   {z}   {var}               -  columns for X, Y, Z, and variable
{tmin}     {tmax}          -  trimming limits
{sum}                  -file for summary output
{out}                  -file for output with data & weights
{xanis}   {zanis}                   -Y and Z cell anisotropy (Ysize=size*Yanis)
{kmin}                           -0=look for minimum declustered mean (1=max)
{ncell}  {min}  {max}               -number of cell sizes, min size, max size
{noff}                           -number of origin offsets
'''
		map_dict = {
				'datafile':temp_dir_str+'tmp.dat',
				'x':col_number(temp_dir_str+'tmp.dat', x),
				'y':col_number(temp_dir_str+'tmp.dat', y),
				'z':col_number(temp_dir_str+'tmp.dat', z),
				'var':col_number(temp_dir_str+'tmp.dat', var),
				'tmin':str(tmin),
				'tmax':str(tmax),
				'sum':temp_dir_str+'tmpsum.dat',
				'out':temp_dir_str+'tmpfile.dat',
				'xanis':str(x_anis),
				'zanis':str(z_anis),
				'ncell':str(n_cell),
				'min':str(min_size),
				'max':str(max_size),
				'kmin': -1 if keep_min == True else 0,
				'noff':str(number_offsets)
		}

		formatted_str = decluspar.format(**map_dict)
		parfile = temp_dir_str+'partmp.par'
		f = open(parfile, 'w')
		f.write(formatted_str)
		f.close()
		program = DATA_PATH+"declus.exe"

		call_program(program, parfile, usewine)

		df1 = read_GeoEAS(temp_dir_str+'tmpsum.dat')
		plots.cell_declus_sum(df1['Cell Size'],df1['Declustered Mean'])
		df2 = read_GeoEAS(temp_dir_str+'tmpfile.dat')
		df['Declustering Weight'] = df2['Declustering Weight']

def kt3d(df, dh, x, y, z, var, grid, variogram, min_samples, max_samples, max_oct, search_radius, search_ang = [0,0,0], discretization = [5,5,1], krig_type='OK', sk_mean = 0, tmin=-1.0e21, tmax=1.0e21, option='grid', debug_level=0, usewine=False):
    """Kriging algorithm. This function will show cross validation results if ``option = 'cross'`` or ``option = 'jackknife'`` or it will return estimated values and variace arrays if ``option = 'grid'``. 
    
    Args:
        df (DataFrame): points data DataFrame
        x (str): x coordinates column name
        y (str): y coordinates column name
        z (str): z coordinates column name
        var (str): variable column name
        grid (dict): grid definitions dictionary
        variogram (dict): variogram dictionary
        min_samples (int): minimum samples
        max_samples (int): maximum number of samples
        max_oct (int): maximum number of samples per octant
        search_radius (list): range1, ramge2, range3 values list
        search_ang (list, optional): azimuth, dip, rake values list. Defaults to [0,0,0].
        discretization (list, optional): block discretization. Defaults to [5,5,1].
        krig_type (str, optional): 'SK' or 'OK' flag. Defaults to 'OK'.
        sk_mean (float, optional): simple kriging mean. Defaults to 0.
        tmin (float, optional): minimum trimming limit. Defaults to -1.0e21.
        tmax (float, optional): maximum trimming limit. Defaults to 1.0e21.
        option (str, optional): cross validation 'cross', jackknife 'jackknife' or estimation 'grid' flag . Defaults to 'grid'.
        debug_level (int, optional): debug level. Defaults to 0. If 2 plots the negative weights histogram.
        usewine (bool, optional): use wine flag. Defaults to False.
    """

    write_GeoEAS(df=df,dh=dh,x=x,y=y,z=z,vars=[var])



    kt3dpar = '''
											Parameters for KT3D
									*******************

START OF PARAMETERS:
{datafile}              -file with data
{dh}  {x}  {y}  {z}  {var}  0                 -   columns for DH,X,Y,Z,var,sec var
{tmin}   {tmax}                 -   trimming limits
{option}                                -option: 0=grid, 1=cross, 2=jackknife
{datafile}                          -file with jackknife data
{x}   {y}   {z}    {var}    0              -   columns for X,Y,Z,vr and sec var
{debug}                                -debugging level: 0,1,2,3
{debugout}                         -file for debugging output
{kt3dout}                         -file for kriged output
{nx}   {ox}    {sx}                  -nx,xmn,xsiz
{ny}   {oy}    {sy}                  -ny,ymn,ysiz
{nz}    {oz}    {sz}                  -nz,zmn,zsiz
{dx}    {dy}      {dz}                    -x,y and z block discretization
{min}    {max}                           -min, max data for kriging
{max_oct}                                -max per octant (0-> not used)
{r1}  {r2}  {r3}                 -maximum search radii
 {a1}   {a2}   {a3}                 -angles for search ellipsoid
{krig_type}     {mean}                      -0=SK,1=OK,2=non-st SK,3=exdrift
0 0 0 0 0 0 0 0 0                -drift: x,y,z,xx,yy,zz,xy,xz,zy
0                                -0, variable; 1, estimate trend
extdrift.dat                     -gridded file with drift/mean
4                                -  column number in gridded file
{varg}'''

    map_dict = {
				'datafile':temp_dir_str+'tmp.dat',
				'dh': col_number(temp_dir_str+'tmp.dat', dh),
				'x': col_number(temp_dir_str+'tmp.dat', x),
				'y': col_number(temp_dir_str+'tmp.dat', y),
				'z':col_number(temp_dir_str+'tmp.dat', z),
				'var':col_number(temp_dir_str+'tmp.dat', var),
				'tmin':str(tmin),
				'tmax':str(tmax),
				'option': 0 if option is 'grid' else 1 if option is 'cross' else 2,
				'debug':debug_level,
				'debugout':temp_dir_str+'debug.out',
				'kt3dout':temp_dir_str+'output.out',
				'nx':grid['nx'],
				'ny':grid['ny'],
				'nz':grid['nz'],
				'ox':grid['ox'],
				'oy':grid['oy'],
				'oz':grid['oz'],
				'sx':grid['sx'],
				'sy':grid['sy'],
				'sz':grid['sz'],
				'dx':discretization[0],
				'dy':discretization[1],
				'dz':discretization[2],
				'min':min_samples,
				'max':max_samples,
				'max_oct':max_oct,
				'r1':search_radius[0],
				'r2':search_radius[1],
				'r3':search_radius[2],
				'a1':search_ang[0],
				'a2':search_ang[1],
				'a3':search_ang[2],
				'krig_type':0 if krig_type is 'SK' else 0,
				'mean':sk_mean,
				'varg':write_varg_str(variogram)
		}

    formatted_str = kt3dpar.format(**map_dict)
    parfile = temp_dir_str+'partmp.par'
    f = open(parfile, 'w')
    f.write(formatted_str)
    f.close()
    program = DATA_PATH+"kt3d.exe"

    call_program(program, parfile, usewine)

    if debug_level == 2:
    
        f = open(temp_dir_str+'debug.out', 'r')
        lines=f.readlines()

        begin = []
        end = []
        for idx, line in enumerate(lines):
            if 'BLOCK EST: x,y,z,vr,wt' in line:
                idxi = idx + 1
                begin.append(idxi)
                #print(idxi, lines[idxi])
            if 'estimate' in line:
                idxf = idx -1
                end.append(idxf)
                #print(idxf, lines[idxf])

        wts = []
        for b,e in zip(begin, end):
            slicing = lines[b:e+1]
            for i in slicing:
                wts.append(float(i.split()[4]))

        wts = np.array(wts)
        mask_negative = wts < 0
        neg_wts = wts[mask_negative]
        total = len(wts)

        plots.histogram(data=neg_wts,title='Negative weights of total {}'.format(total), n_bins=8)

    if option is 'grid':

        df1 = read_GeoEAS(temp_dir_str+'output.out')
        return df1['Estimate'], df1['EstimationVariance']

    if option is 'cross' or option is 'jackknife':

        df1 = read_GeoEAS(temp_dir_str+'output.out')
        real, estimate, error = df1['True'], df1['Estimate'], df1['Error: est-true']
        mask = np.isfinite(estimate)
        plots.xval(real[mask], estimate[mask], error[mask], pointsize=8, figsize=(500,900))


##########################################################################################################################################################3


def gamv(df,  x='x', y='y', z='z', var_h ='V', var_tail = 'V',  nlags:int =10, lagdist=10, lagtol= 5, 
	ndirections=3, azm = [0,0,0], atol = [22,22,22], bandh =[5,5,5],
	dip = [0,0,0], dtol =[22,22,22], bandv = [5,5,5], standardize = 0, 
	variogram_type = 1, tmin =  -999, tmax = 999, usewine = False):
	
	"""Calculate experimental functions using gamv 
	
	Args:
	    df (pandas.DataFrame): DataFrame containing all spatial information
	    x (str): Label for the X coordinates 
	    y (str, optional): Label for the Y coordinates 
	    z (str, optional): Label for the Z coordinates 
	    var_h (str, optional): Label for the variable at the head of the distance vector
	    var_tail (str, optional): Label for the variable at the tail of the distance vector
	    nlags (int, optional): Number of lags for the experimental variogram
	    lagdist (int, optional): Size of the lag for the experimental variogram
	    lagtol (int, optional): Linear tolerance for the experimental variogram 
	    ndirections (int, optional): Number of directions to calculate
	    azm (list, optional): List of azimuth angles in degrees
	    atol (list, optional): List of horizontal angular tolerances in degrees
	    bandh (list, optional): List of horizontal band width 
	    dip (list, optional): List of dip angles in degrees 
	    dtol (list, optional): List of vertical angular tolerances in degrees
	    bandv (list, optional): List of vertical band width 
	    standardize (int, optional): Standardize variables 0- No, 1- Yes 
	    variogram_type (int, optional): Variogram experimental function. 1 = traditional semivariogram, 2 = traditional cross semivariogram, 3 = covariance,  4 = correlogram, 5 = general relative semivariogram,   6 = pairwise relative semivariogram, 7 = semivariogram of logarithms, 8 = semimadogram, 9 = indicator semivariogram - continuous, 10= indicator semivariogram - categorical
	    tmin (TYPE, optional): Minimum trimming limits 
	    tmax (int, optional): Maximum trimming limits 
	    usewine (bool, optional): Option to use wine for Linux users
	"""

	# Write gslib file on directory 

	write_GeoEAS(df=df,dh=None,x=x,y=y,z=z,vars=[var_h, var_tail])

	# Define the variogram directions strings for the file parameter

	directions = ""
	for i in range(ndirections):
		if i == ndirections - 1:
			directions += "{} {} {} {} {} {} -azm,atol,bandh,dip,dtol,bandv".format(str(azm[i]), str(atol[i]), str(bandh[i]), str(dip[i]), str(dtol[i]), str(bandv[i]))
		else:
			directions += "{} {} {} {} {} {} -azm,atol,bandh,dip,dtol,bandv \n".format(str(azm[i]), str(atol[i]), str(bandh[i]), str(dip[i]), str(dtol[i]), str(bandv[i]))

	# Define the file parameter 

	gamvpar = '''
								Parameters for GAMV
									*******************

START OF PARAMETERS:
{datafile}               -file with data
{x}   {y}   {z}                         -   columns for X, Y, Z coordinates
2   {var_h}   {var_tail}                        -   number of variables,col numbers
{tmin}     {tmax}               -   trimming limits
{out}                           -file for variogram output
{nlags}                                -number of lags
{lagdist}                                -lag separation distance
{lagtol}                              -lag tolerance
{ndirections}                                 -number of directions
{directions}  
{standardize}                                 -standardize sills? (0=no, 1=yes)
1                                 -number of variograms
1   2   {variogram_type}                         -tail var., head var., variogram type
'''
	string_c = temp_dir_str.replace("\\", "/")
	map_dict = {'datafile':string_c+'tmp.dat',
				'x': col_number(string_c +'tmp.dat', x),
				'y': col_number(string_c +'tmp.dat', y),
				'z':col_number(string_c +'tmp.dat', z),
				'var_h':col_number(string_c +'tmp.dat', var_h),
				'var_t':col_number(string_c +'tmp.dat', var_tail),
				'out': string_c +'output.out',
				'tmin':str(tmin),
				'tmax':str(tmax),
				'nlags':nlags,
				'lagdist':lagdist,
				'lagtol' : lagtol,
				'ndirections': ndirections,
				'directions': directions,
				'standardize': standardize,
				'var_h':col_number(string_c +'tmp.dat',var_h),
				'var_tail':col_number(string_c +'tmp.dat', var_tail),
				'variogram_type': variogram_type}


	# Write the gslib par file 

	formatted_str = gamvpar.format(**map_dict)
	parfile = temp_dir_str+'gamv.par'
	f = open(parfile, 'w')
	f.write(formatted_str)
	f.close()

	# Run the gamv executable 

	program = DATA_PATH+"gamv.exe"
	call_program(program, parfile, usewine)

	# Open and read experimental variogram output 

	myfile = open(temp_dir_str+'output.out', 'r')
	data=myfile.readlines()
	count = 0
	dfs =[]

	distance = []
	number_of_pairs =[]
	variogram = [] 
	number = []
	count = 0 
	list_of_headers = ['Semivariogram', 'Cross', 'Covariance','Correlogram', 
	'Relative', 'General', 'Pairwise', 'Variogram', 'Semimadogram', 'Indicator' ]
	for i in range(len(data)):
		splited = data[i].split()
		if (splited[0] not in list_of_headers):
			distance.append(float(splited[1]))
			number_of_pairs.append(int(splited[3]))
			variogram.append(float(splited[2]))
			number.append(int(splited[0]))
		if (splited[0] in list_of_headers and i > 0) or (i == (len(data) -1)):
			df = pd.DataFrame(np.array([distance, variogram, number_of_pairs]).T, columns =['Average distance', 'Spatial continuity', 'Number of pairs'])
			dfs.append(df)
			number  =[] 
			distance = []
			number_of_pairs =[]
			variogram = [] 

	print(dfs)
	

	# If cross experimental variograms, only use the cross variograms 

	# Plot the experimental variogram functions 

	plots.plot_experimental_variogram(dfs, azm, dip)

	# Define the returning dictionary containing the experimental values 
	returning = {'Directions': [azm, dip],
				 'Values' : dfs}

	return returning 

##########################################################################################################################################################