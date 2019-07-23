import subprocess
import numpy as np
import pandas as pd
from pygslib import plots

#############################################################################################################

def call_program(program, parfile, usewine=False):
    if usewine == True:
        p = subprocess.Popen(['wine', program, parfile], stdout=subprocess.PIPE)
    else:
         p = subprocess.Popen([program, parfile], stdout=subprocess.PIPE)

    for line in p.stdout:    
        print(line.decode('utf-8'), end='')

def write_GeoEAS(df,x,y,z,vars=[]):
    """wirite a GeoEAS file from a DataFrame
    
    Arguments:
        df {DataFrame} -- pandas data frame object
        x {str} -- x coordinates string
        y {str} -- y coordinates string
    
    Keyword Arguments:
        z {str} -- z coordinates string (default: {None})
        vars {list} -- list of variables names as strings (default: {[]})
    """
    df.replace(float('nan'),-999,inplace=True)
    columns = [x,y]
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
    f = open('pygslib/gslib/tmp/tmp.dat', 'w')
    f.write(data)
    f.close()

def read_GeoEAS(file):
    f = open(file, 'r')
    col_names = []
    results = []
    
    for index, line in enumerate(f):
    	if index == 0:
    		continue
    	elif index == 1:
    		n_cols = int(line)
    	elif index <= n_cols+1:
    		col_names.append(line[:-1])
    	else:
    		values = [float(i) for i in line.split()]
    		results.append(values)

    f.close()

    return pd.DataFrame(results, columns = col_names)

#############################################################################################################

def celldeclus(df, x, y, z, var, tmin=-1.0e21, tmax=1.0e21, summary_file='pygslib/gslib/tmp/tmpsum.dat', output_file='pygslib/gslib/tmp/tmpfile.dat', x_anis=1, y_anis=1, n_cell=10, min_size=1, max_size=20, keep_min = True, specific_size=0, usewine=False):

    write_GeoEAS(df=df,x=x,y=y,z=z,vars=[var])
    
    celldecluspar = '''
                    Parameters for CELLDECLUS
                *************************

START OF PARAMETERS:
{datafile}                    -file with data
{x}   {y}   {z}   {var}               -  columns for X, Y, Z, and variable
{tmin}     {tmax}          -  trimming limits
{sum}          -file for summary output
{out}              -file for output with data and weights
{xanis}   {yanis}                   -Y and Z cell anisotropy (Ysize=size*Yanis)
{ncell}  {min}  {max}               -number of cell sizes, min size, max size
{kmin}  {size}                    -cell size to keep: -1 = minimum
                                                0 = specified
                                                +1 = maximum
    '''

    map_dict = {
        'datafile':'pygslib/gslib/tmp/tmp.dat',
        'x':'1',
        'y':'2',
        'z':0 if z == None else '3',
        'var':'3' if z == None else '4',
        'tmin':str(tmin),
        'tmax':str(tmax),
        'sum':summary_file,
        'out':output_file,
        'xanis':str(x_anis),
        'yanis':str(y_anis),
        'ncell':str(n_cell),
        'min':str(min_size),
        'max':str(max_size),
        'kmin': -1 if keep_min == True else 0,
        'size':str(specific_size)
    }

    formatted_str = celldecluspar.format(**map_dict)
    parfile = 'pygslib/gslib/tmp/partmp.par'
    f = open(parfile, 'w')
    f.write(formatted_str)
    f.close()
    program = "pygslib/gslib/CellDeclus.exe"

    call_program(program, parfile, usewine)

    df1 = read_GeoEAS('pygslib/gslib/tmp/tmpsum.dat')
    plots.cell_declus_sum(df1['Cell Size'],df1['Declustered Mean'])
    df2 = read_GeoEAS('pygslib/gslib/tmp/tmpfile.dat')
    df['Cell Declustering Weight'] = df2['Cell Declustering Weight']
