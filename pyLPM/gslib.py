import subprocess
import numpy as np
import pandas as pd
from pyLPM import plots

#############################################################################################################

def call_program(program, parfile, usewine=False):
    if usewine == True:
        p = subprocess.Popen(['wine', program, parfile], stdout=subprocess.PIPE)
    else:
         p = subprocess.Popen([program, parfile], stdout=subprocess.PIPE)

    for line in p.stdout:    
        print(line.decode('utf-8'), end='')

def write_GeoEAS(df,dh,x,y,z,vars=[]):
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
    f = open('pyLPM/gslib90/tmp/tmp.dat', 'w')
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

def col_number(file, col):
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
    if 'nugget' in varg:
        nst = len(varg) - 1
        nugget = varg['nugget']
    else:
        nst = len(varg)
        nugget = 0
    
    var_str = '{} {} \n'.format(nst, nugget)
    
    for struct in varg:
        if struct is not 'nugget':
            if struct is 'spherical':
                it = 1
            elif struct is 'exponetial':
                it = 2
            elif struct is 'gaussian':
                it = 3

            new_lines = '{} {} {} {} \n {} {} {}'.format(it, varg[struct]['cc'], varg[struct]['a1'], varg[struct]['a2'], varg[struct]['a3'], varg[struct]['r1'], varg[struct]['r2'], varg[struct]['r3'])

            var_str = var_str + new_lines
        
    return var_str

#############################################################################################################

def declus(df, x, y, z, var, tmin=-1.0e21, tmax=1.0e21, summary_file='pyLPM/gslib90/tmp/tmpsum.dat', output_file='pyLPM/gslib90/tmp/tmpfile.dat', x_anis=1, z_anis=1, n_cell=10, min_size=1, max_size=20, keep_min = True, number_offsets=4, usewine=False):

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
        'datafile':'pyLPM/gslib90/tmp/tmp.dat',
        'x':col_number('pyLPM/gslib90/tmp/tmp.dat', x),
        'y':col_number('pyLPM/gslib90/tmp/tmp.dat', y),
        'z':col_number('pyLPM/gslib90/tmp/tmp.dat', z),
        'var':col_number('pyLPM/gslib90/tmp/tmp.dat', var),
        'tmin':str(tmin),
        'tmax':str(tmax),
        'sum':summary_file,
        'out':output_file,
        'xanis':str(x_anis),
        'zanis':str(z_anis),
        'ncell':str(n_cell),
        'min':str(min_size),
        'max':str(max_size),
        'kmin': -1 if keep_min == True else 0,
        'noff':str(number_offsets)
    }

    formatted_str = decluspar.format(**map_dict)
    parfile = 'pyLPM/gslib90/tmp/partmp.par'
    f = open(parfile, 'w')
    f.write(formatted_str)
    f.close()
    program = "pyLPM/gslib90/declus.exe"

    call_program(program, parfile, usewine)

    df1 = read_GeoEAS(summary_file)
    plots.cell_declus_sum(df1['Cell Size'],df1['Declustered Mean'])
    df2 = read_GeoEAS(output_file)
    df['Declustering Weight'] = df2['Declustering Weight']

def kt3d(df, x, y, z, var):

    write_GeoEAS(df=df,dh=None,x=x,y=y,z=z,vars=[var])

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
{nx}    {oz}    {sz}                  -nz,zmn,zsiz
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
        'datafile':'pyLPM/gslib90/tmp/tmp.dat',
        'dh': col_number('pyLPM/gslib90/tmp/tmp.dat', dh),
        'x': col_number('pyLPM/gslib90/tmp/tmp.dat', x),
        'y': col_number('pyLPM/gslib90/tmp/tmp.dat', y),
        'z':col_number('pyLPM/gslib90/tmp/tmp.dat', z),
        'var':col_number('pyLPM/gslib90/tmp/tmp.dat', var),
        'tmin':str(tmin),
        'tmax':str(tmax),
        'option' 0 if option is 'grid' elif 1 if option is 'cross' elif 2 if option is 'jackknife'
    }

