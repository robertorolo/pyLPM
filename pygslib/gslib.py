import subprocess
import numpy as np

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

def read_GeoEAS(file, cols='all'):
    f = open(file, 'r')
    col_names = []
    for index, line in enumerate(f):
    	if index == 0:
    		continue
    	elif index == 1:
    		n_cols = int(line)
    	elif index <= n_cols:
    		col_names.append(line)

    print(col_names)




def celldeclus(df, x, y, z, var, tmin=-1.0e21, tmax=1.0e21, summary_file='tmp/tmpsum.dat', output_file='tmp/tmpfile.dat', x_anis=1, y_anis=1, n_cell=10, min_size=1, max_size=20, keep_min = True, specific_size=0):

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
        'datafile':'tmp/tmp.dat',
        'x':'1',
        'y':'2',
        'z':0 if z == None else '3',
        'var':'4',
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
    p = subprocess.Popen([program, parfile], stdout=subprocess.PIPE)

    for line in p.stdout:    
        print(line.decode('utf-8'), end='')



