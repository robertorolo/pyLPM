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
from sklearn import datasets, linear_model



from _plotly_future_ import v4_subplots
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats
import pandas as pd
from pyLPM import utils 
import math


#############################################################################################################

def plot_experimental_variogram(dfs, azm,dip):
    size_row = 1 if len(azm) < 4 else int(math.ceil(len(azm)/4))
    size_cols = 4 if len(azm) >= 4 else int(len(azm))

    titles = ["Azimuth {} - Dip {}".format(azm[j], dip[j]) for j in range(len(azm))]
    fig = make_subplots(rows=size_row, cols=size_cols, subplot_titles=titles)

    count_row = 1
    count_cols = 1


    for j, i in enumerate(dfs):

        fig.add_trace(go.Scatter(x=i['Average distance'], y=i['Spatial continuity'],
                mode='markers',
                name='Experimental' ,
                marker= dict(color =i['Number of pairs']),
                text =i['Number of pairs'],
                textposition='bottom center') , row=count_row, col=count_cols)
        fig.update_xaxes(title_text="Distance", row=count_row, col=count_cols, automargin = True)
        fig.update_yaxes(title_text="Variogram", row=count_row, col=count_cols, automargin=True)
        fig.update_layout(autosize=True)

        count_cols += 1
        if count_cols > 4:
            count_cols = 1
            count_row += 1
    fig.show()

    
    return (dfs)

#############################################################################################################

def plot_hscat(store, lagsize, lagmultiply, figsize):
    """Plots h scatterplot
    
    Args:
        store ([type]): [description]
        lagsize (float): lag size
        lagmultiply (int): how many lags to plot
    """
    distance =[i*lagsize for i in lagmultiply]

   
    fig = go.Figure()
    statitics = ""

    annotations = []
    for j, i in enumerate(store):

         # Create linear regression object

         if len(i[0]) != 0:
             slope, intercept, r_value, p_value, std_err = stats.linregress(i[0],i[1])

             statitics  = statitics +''' rho ({}) / distance {}  <br />'''.format(str(np.round(r_value,2)), str(distance[j]))

             x_val = np.linspace(0, np.max(i[0]), 100)
             y_val = slope*x_val + intercept
         else:
            x_val = []
            y_val =[]

         fig.add_trace(go.Scatter(x=i[0], y=i[1],
                mode='markers',
                name='Hscatter - distance {}'.format(str(distance[j]))))
         fig.add_trace(go.Scatter(x=x_val, y=y_val,
                mode= 'lines',
                name = 'Regression - distance {}'.format(str(distance[j])),
                line= dict(width=3)))
    fig.layout = {
        'title':'Hscatter plots',
        'xaxis':{'title':'Z(x)','zeroline':True,'autorange':True},
        'yaxis':{'title':'Z(x+h)','zeroline':True,'autorange':True},
        'annotations':[{'text':statitics,'showarrow':False,'x':0.98,'y':0.98,'xref':'paper','yref':'paper','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}],
        'width':figsize[0],
        'height':figsize[1],
        }
       
    fig['layout'].update(annotations =annotations)
    fig.show()

    
#############################################################################################################

def weighted_avg_and_var(values, weights):
    """Calculates weighted average and variance
    
    Args:
        values (array): data values array
        weights (array): weight values array
    
    Returns:
        array: average, variance
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, variance

def isotopic_arrays(arrays):
    """Extract isotopic group from arrays
    
    Args:
        arrays (array): data values nd array
    
    Returns:
        arrays: nd idotopic arrays
    """
    masks = []
    for array in arrays:
        masks.append(np.isnan(array))
    mask = sum(masks)
    masked_arrays = []
    for array in arrays:
        masked_var = np.ma.array(array, mask=mask).compressed()
        masked_arrays.append(masked_var)
    
    return masked_arrays

#############################################################################################################

def locmap(x, y, variable, categorical=False, title='', x_axis='Easting (m)', y_axis='Northing (m)', pointsize=8, colorscale='Jet', colorbartitle='', figsize=(700,700)):
    """Plots a location map for samples
    
    Args:
        x (array): x values array
        y (array): y values array
        variable (array): variables values array
        categorical (bool, optional): Categorical variable flag. Defaults to False.
        title (str, optional): Plot title. Defaults to ''.
        x_axis (str, optional): x axis title. Defaults to 'Easting (m)'.
        y_axis (str, optional): y axis title. Defaults to 'Northing (m)'.
        pointsize (int, optional): point size. Defaults to 8.
        colorscale (str, optional): color scale. Defaults to 'Jet'.
        colorbartitle (str, optional): colorbar title. Defaults to ''.
        figsize (tuple, optional): figure size. Defaults to (700,700).
    
    Returns:
        iplot: Location map
    """

    variable = np.where(variable == -999.0, float('nan'), variable)

    traces = []
    
    if categorical == True:

        cats  = np.unique(variable[~np.isnan(variable)])

        for cat in cats:

            mask = variable == cat
           
            trace = {
            'type':'scatter',
            'mode':'markers',
            'x':x[mask],
            'y':y[mask],
            'marker':{'size':pointsize},
            'text':variable[mask],
            'name':str(int(cat)),
            }

            traces.append(trace)

    else:

        trace = {
            'type':'scatter',
            'mode':'markers',
            'x':x,
            'y':y,
            'marker':{'size':pointsize,'color':variable,'colorscale':colorscale,'showscale':True,'colorbar':{'title':colorbartitle}},
            'text':variable
        }

        traces.append(trace)

    layout = {
        'title':title,
        'xaxis':{'title':x_axis,'scaleanchor':'y','zeroline':False},
        'yaxis':{'title':y_axis,'zeroline':False},
        'width':figsize[0],
        'height':figsize[1],
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def histogram(data, n_bins=20, wt=None, title='', x_axis='', y_axis='', cdf=False, figsize=(700,700)):
    """Histogram plot
    
    Args:
        data (array): data values array
        n_bins (int, optional): number of bins. Defaults to 20.
        wt (array, optional): weights values array. Defaults to None.
        title (str, optional): plot title. Defaults to ''.
        x_axis (str, optional): x axis title. Defaults to ''.
        y_axis (str, optional): y axis title. Defaults to ''.
        cdf (bool, optional): cdf flag. Defaults to False.
        figsize (tuple, optional): figure size. Defaults to (700,700).
    
    Returns:
        iplot: histogram plot
    """

    dataf = np.where(data == -999.0, float('nan'), data)
    dataf = data[~np.isnan(data)]

    statistics = '''
    n {}  <br />
    min {} <br />
    max {} <br />
    mean {}  <br />
    stdev {}  <br />
    cv {}  
    '''.format(round(len(dataf),0), round(dataf.min(),2), round(dataf.max(),2),  round(dataf.mean(),2), round(np.sqrt(dataf.var()),2), round(np.sqrt(dataf.var())/dataf.mean(),2))

    if wt is not None:

        mean, var = weighted_avg_and_var(dataf, wt)

        statistics = '''
        n {}  <br />
        min {} <br />
        max {} <br />
        mean {}  <br />
        stdev {}  <br />
        cv {}  <br />
        weighted
        '''.format(round(len(dataf),0),  round(dataf.min(),2), round(dataf.max(),2), round(mean,2), round(np.sqrt(var),2), round(np.sqrt(var)/mean),2)

    traces = []

    hist, bin_edges = np.histogram(dataf, bins=n_bins, weights=wt, density=True)
    hist = hist*np.diff(bin_edges)
    
    trace = {
        'type':'bar',
        'x':bin_edges,
        'y':hist,
        'name':'pdf'
    }

    traces.append(trace)

    if cdf == True:

        hist = np.cumsum(hist)

        trace = {
            'type':'bar',
            'x':bin_edges,
            'y':hist,
            'name':'cdf'
        }

        traces.append(trace)

    layout = {
        'title':title,
        'xaxis':{'title':x_axis},
        'yaxis':{'title':y_axis},
        'width':figsize[0],
        'height':figsize[1],
        'barmode':'group',
        'annotations':[{'text':statistics,'showarrow':False,'x':0.98,'y':0.98,'xref':'paper','yref':'paper','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}]
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def scatter2d(x, y, variable='kernel density', xy_line = True, best_fit_line=True, title='', x_axis='', y_axis='', pointsize=8, colorscale='Viridis', colorbartitle='', figsize=(700,700)):
    """2D scatter plot
    
    Args:
        x (array): x variable data array
        y (array): y variable data array
        variable (str, optional): variable to color points. Defaults to 'kernel density'.
        xy_line (bool, optional): x=y line flag. Defaults to True.
        best_fit_line (bool, optional): best fit line flag. Defaults to True.
        title (str, optional): plot title. Defaults to ''.
        x_axis (str, optional): x axis title. Defaults to ''.
        y_axis (str, optional): y axis title. Defaults to ''.
        pointsize (int, optional): point size. Defaults to 8.
        colorscale (str, optional): color scale. Defaults to 'Viridis'.
        colorbartitle (str, optional): color bar title. Defaults to ''.
        figsize (tuple, optional): figure size. Defaults to (700,700).
    
    Returns:
        iplot: 2D scatter plot
    """

    x = np.where(x == -999.0, float('nan'), x)
    y = np.where(y == -999.0, float('nan'), y)

    x, y = isotopic_arrays([x,y])[0], isotopic_arrays([x,y])[1]

    statistics = '''
    n {}  <br />
    rho {}
    '''.format(round(len(x),0), round(np.corrcoef([x,y])[1,0],2))

    if best_fit_line == True:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        statistics = '''
        n {}  <br />
        rho {} <br />
        slope {} <br />
        intercept {}
        '''.format(round(len(x),0), round(np.corrcoef([x,y])[1,0],2), round(slope,2), round(intercept,2))
    
    if type(variable) is not str:
        variable = np.where(variable == -999.0, float('nan'), variable)

    else:
        xy = np.vstack([x,y])
        variable = gaussian_kde(xy)(xy)
    
    traces = []
    
    trace = {
        'type':'scatter',
        'mode':'markers',
        'x':x,
        'y':y,
        'marker':{'size':pointsize,'color':variable,'colorscale':colorscale,'showscale':False,'colorbar':{'title':colorbartitle}},
        'text':variable,
        'name':'Scatter'
    }

    traces.append(trace)

    if xy_line == True:

        maxxy = [max(x), max(y)]

        trace = {
            'type':'scatter',
            'mode':'lines',
            'x':[0,max(maxxy)],
            'y':[0,max(maxxy)],
            'name':'x=y line',
            'line':{'dash':'dot','color':'red'}

        }

        traces.append(trace)

    if best_fit_line == True:

        maxxy = [max(x), max(y)]
        minxy = [min(x), min(y)]
        vals = np.arange(min(minxy),max(maxxy))

        trace = {
            'type':'scatter',
            'mode':'lines',
            'x':vals,
            'y':slope*vals+intercept,
            'name':'best fit line',
            'line':{'dash':'dot','color':'grey'}

        }

        traces.append(trace)

    layout = {
        'title':title,
        'xaxis':{'title':x_axis,'zeroline':True,'autorange':True},
        'yaxis':{'title':y_axis,'zeroline':True,'autorange':True},
        'width':figsize[0],
        'height':figsize[1],
        'annotations':[{'text':statistics,'showarrow':False,'x':0.98,'y':0.98,'xref':'paper','yref':'paper','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}],
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def cell_declus_sum(cell_size, mean, title='Cell declus summary', pointsize=8, figsize=(600,600)):
    """Cell declustering summary plot
    
    Args:
        cell_size (array): cell sizes values array
        mean (array): mean values array
        title (str, optional): plt title. Defaults to 'Cell declus summary'.
        pointsize (int, optional): point size. Defaults to 8.
        figsize (tuple, optional): figure size. Defaults to (600,600).
    
    Returns:
        iplot: cell declus summary plot
    """

    index = np.where(mean == min(mean))[0][0]

    text_annotation = '''
    cell size {} <br />
    mean {}
    '''.format(round(cell_size[index],2), round(min(mean),2))

    trace = {
    'type':'scatter',
    'mode':'markers',
    'x':cell_size,
    'y':mean,
    'marker':{'size':pointsize},
    }

    layout = {
    'title':title,
    'xaxis':{'title':'cell size','zeroline':True,'autorange':True},
    'yaxis':{'title':'mean','zeroline':True,'autorange':True},
    'width':figsize[0],
    'height':figsize[1],
    'annotations':[{'text':text_annotation,'showarrow':True,'arrowhead':7,'ax':0,'ay':-0.5*min(mean),'x':cell_size[index],'y':min(mean),'xref':'x','yref':'y','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}],
    }

    fig = go.Figure([trace], layout)

    return pyo.iplot(fig)

def pixelplot(grid_dic, variable, categorical=False, points=None, gap=0, title='', x_axis='Easting (m)', y_axis='Northing (m)', colorscale='Jet', colorbartitle='', figsize=(700,700)):
    """Pixel plot for gridded data
    
    Args:
        grid_dic (dict): grid definitions dictionary
        variable (array): data values array
        categorical (bool, optional): categorical data flag. Defaults to False.
        points (lst, optional): [x,y,z,var] list to plot points on top of grid. Defaults to None.
        gap (int, optional): gap between blocks. Defaults to 0.
        title (str, optional): plt title. Defaults to ''.
        x_axis (str, optional): x axis title. Defaults to 'Easting (m)'.
        y_axis (str, optional): y axis title. Defaults to 'Northing (m)'.
        colorscale (str, optional): color scale. Defaults to 'Jet'.
        colorbartitle (str, optional): color bar title. Defaults to ''.
        figsize (tuple, optional):figure size. Defaults to (700,700).
    
    Returns:
        iplot: pixel plot 
    """

    variable = np.where(variable == -999, float('nan'), variable)
    
    traces = []

    x = np.array([(grid_dic['ox']+i*grid_dic['sx']) for i in range(grid_dic['nx'])])
    y = np.array([(grid_dic['oy']+j*grid_dic['sy']) for j in range(grid_dic['ny'])])

    trace = {
    'type':'heatmap',
    'z':variable.reshape(grid_dic['ny'], grid_dic['nx']),
    'x':x,
    'y':y,
    'colorscale':colorscale,
    'xgap':gap,
    'ygap':gap
    }

    traces.append(trace)

    if points is not None:
        trace = {
            'type':'scatter',
            'mode':'markers',
            'x':points[0],
            'y':points[1],
            'marker':{'colorscale':colorscale,'size':6,'color':points[2],'line':{'color':'black','width':1}},
            'text':variable}

        traces.append(trace)
    
    layout = {
        'title':title,
        'xaxis':{'title':x_axis,'zeroline':False,'scaleanchor':'y'},
        'yaxis':{'title':y_axis,'zeroline':False},
        'width':figsize[0],
        'height':figsize[1],
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def qqplot(x,y, dicretization=100, title='', x_axis='', y_axis='', pointsize=8, figsize=(700,700)):
    """QQ plot
    
    Args:
        x (array): x variable data array
        y (array): y variable data array
        dicretization (int, optional): number of quantiles to plot. Defaults to 100.
        title (str, optional): plot title. Defaults to ''.
        x_axis (str, optional): x axis title. Defaults to ''.
        y_axis (str, optional): y axis title. Defaults to ''.
        pointsize (int, optional): point size. Defaults to 8.
        figsize (tuple, optional): figure size. Defaults to (700,700).
    
    Returns:
        iplot: QQ plot
    """
    
    x = np.where(x == -999.0, float('nan'), x)
    y = np.where(y == -999.0, float('nan'), y)

    x_quant = [np.nanquantile(np.array(x), q) for q in np.linspace(0,1,dicretization)]
    y_quant = [np.nanquantile(np.array(y), q) for q in np.linspace(0,1,dicretization)]

    traces = []

    maxxy = [max(x_quant), max(y_quant)]
    minxy = [min(x_quant), min(y_quant)]
    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':[min(minxy),max(maxxy)],
    'y':[min(minxy),max(maxxy)],
    'name':'reference',
    'line':{'dash':'dot','color':'grey'}
    }

    traces.append(trace)

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':x_quant,
    'y':y_quant,
    'name':'qq',
    #'line':{'dash':'dot','color':'grey'}
    }

    traces.append(trace)

    layout = {
    'title':title,
    'xaxis':{'title':x_axis,'zeroline':True,'autorange':False,'range':[min(minxy),max(maxxy)]},
    'yaxis':{'title':y_axis,'zeroline':True,'autorange':False,'range':[min(minxy),max(maxxy)]},
    'width':figsize[0],
    'height':figsize[1],
    #'annotations':[{'text':statistics,'showarrow':False,'x':0.98,'y':0.98,'xref':'paper','yref':'paper','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}],
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

def xval(real, estimate, error, pointsize=8, figsize=(500,900)):
    """Cross validation results summary
    
    Args:
        real (array): true values array
        estimate (array): estimated values array
        error (array): real - estimates array
        pointsize (int, optional): point size. Defaults to 8.
        figsize (tuple, optional): figure size. Defaults to (500,900).
    
    Returns:
        iplot: cross validation summary
    """

    fig = plotly.subplots.make_subplots(rows=1, cols=3)

    real = np.where(real == -999.0, float('nan'), real)
    estimate = np.where(estimate == -999.0, float('nan'), estimate)
    slope, intercept, r_value, p_value, std_err = stats.linregress(real,estimate)

    trace = {
    'type':'scatter',
    'mode':'markers',
    'x':real,
    'y':estimate,
    'marker':{'size':pointsize},
    'name':'True x Estimates'
    }

    fig.append_trace(trace, 1, 1)

    maxxy = [max(real), max(estimate)]
    minxy = [min(real), min(estimate)]
    vals = np.arange(min(minxy),max(maxxy))

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':vals,
    'y':slope*vals+intercept,
    'name':'best fit line',
    'line':{'dash':'dot','color':'grey'}
    }

    fig.append_trace(trace, 1, 1)

    maxxy = [max(real), max(estimate)]

    trace = {
    'type':'scatter',
    'mode':'lines',
    'x':[0,max(maxxy)],
    'y':[0,max(maxxy)],
    'name':'x=y line',
    'line':{'dash':'dot','color':'red'}
    }

    fig.append_trace(trace, 1, 1)

    hist, bin_edges = np.histogram(error, bins=20, density=True)
    hist = hist*np.diff(bin_edges)
    
    trace = {
    'type':'bar',
    'x':bin_edges,
    'y':hist,
    'name':'error histogram'
    }

    fig.append_trace(trace, 1, 2)

    trace = {
    'type':'scatter',
    'mode':'markers',
    'x':real,
    'y':error,
    'marker':{'size':pointsize},
    'name':'True x Error'
    }

    fig.append_trace(trace, 1, 3)

    statistics = '''
    n {}  <br />
    min {} <br />
    max {} <br />
    mean {}  <br />
    stdev {}  <br />
    cv {}  <br />
    rho {} <br />
    slope {}
    '''.format(round(len(error),0), round(error.min(),2), round(error.max(),2),  round(error.mean(),2), round(np.sqrt(error.var()),2), round(np.sqrt(error.var())/error.mean(),2), round(np.corrcoef([real,estimate])[1,0],2), round(slope,2))

    fig.layout.update(annotations=[{'text':statistics,'showarrow':False,'x':0.98,'y':0.98,'xref':'paper','yref':'paper','align':'left','yanchor':'top','bgcolor':'white','bordercolor':'black'}])

    fig.layout.update(title='Cross validation results')

    return pyo.iplot(fig)

def swath_plots(x,y,z,point_var,grid,grid_var,n_bins=10):
    """Swath plots in x, y and z
    
    Args:
        x (array): x coordinates array
        y (array): y coordinates array
        z (array): z coordinates array
        point_var (array): point variable array
        grid (dict): grid definitions dictionary
        grid_var (array): gridded variable array
        n_bins (int, optional): number of bins. Defaults to 10.
    
    Returns:
        iplot: swath plots
    """

    mask_pt = np.isfinite(point_var)
    mask_grid = np.isfinite(grid_var)
    if z is None:
        z = np.zeros(len(x))
    x, y, z = np.array(x)[mask_pt], np.array(y)[mask_pt], np.array(z)[mask_pt]
    point_var, grid_var = np.array(point_var)[mask_pt], np.array(grid_var)[mask_grid]

    points_df = pd.DataFrame(columns=['x','y','z','var'])
    points_df['x'], points_df['y'], points_df['z'], points_df['var'] = x, y, z, point_var

    grid_df = pd.DataFrame(columns=['x','y','z'], data=utils.add_coord(grid))
    #grid_df.sort_values(by=['z','y','x'], inplace=True)
    grid_df['var'] = grid_var

    x_linspace, y_linspace, z_linspace = np.linspace(min(grid_df['x']), max(grid_df['x']), n_bins), np.linspace(min(grid_df['y']), max(grid_df['y']), n_bins), np.linspace(min(grid_df['z']), max(grid_df['z']), n_bins)

    x_grades_pts = []
    x_grades_grid = []
    x_n_pts = []

    for idx, slice in enumerate(x_linspace):
        if idx != 0:
            pts_df_filter = (points_df['x'] >= x_linspace[idx-1]) & (points_df['x'] <= x_linspace[idx])
            grid_df_filter = (grid_df['x'] >= x_linspace[idx-1]) & (grid_df['x'] <= x_linspace[idx])

            x_grades_pts.append(np.mean(points_df[pts_df_filter]['var']))
            x_grades_grid.append(np.mean(grid_df[grid_df_filter]['var'])) 
            x_n_pts.append(len(points_df[pts_df_filter]['var']))

    y_grades_pts = []
    y_grades_grid = []
    y_n_pts = []

    for idx, slice in enumerate(y_linspace):
        if idx != 0:
            pts_df_filter = (points_df['y'] >= y_linspace[idx-1]) & (points_df['y'] <= y_linspace[idx])
            grid_df_filter = (grid_df['y'] >= y_linspace[idx-1]) & (grid_df['y'] <= y_linspace[idx])

            y_grades_pts.append(np.mean(points_df[pts_df_filter]['var']))
            y_grades_grid.append(np.mean(grid_df[grid_df_filter]['var'])) 
            y_n_pts.append(len(points_df[pts_df_filter]['var']))

    z_grades_pts = []
    z_grades_grid = []
    z_n_pts = []

    for idx, slice in enumerate(z_linspace):
        if idx != 0:
            pts_df_filter = (points_df['z'] >= z_linspace[idx-1]) & (points_df['z'] <= z_linspace[idx])
            grid_df_filter = (grid_df['z'] >= z_linspace[idx-1]) & (grid_df['z'] <= z_linspace[idx])

            z_grades_pts.append(np.mean(points_df[pts_df_filter]['var']))
            z_grades_grid.append(np.mean(grid_df[grid_df_filter]['var'])) 
            z_n_pts.append(len(points_df[pts_df_filter]['var']))

    if sum(z) == 0:

        fig = plotly.subplots.make_subplots(rows=2, cols=1)

        tracepts = {
        'type':'scatter',
        'mode':'lines',
        'x':x_linspace,
        'y':x_grades_pts,
        'name':'points in x',
        'line':{'dash':'solid','color':'red'},
        }

        tracegrid = {
        'type':'scatter',
        'mode':'lines',
        'x':x_linspace,
        'y':x_grades_grid,
        'name':'grid in x',
        'line':{'dash':'solid','color':'blue'},
        }

        tracenpts = {
        'type':'bar',
        'x':x_linspace,
        'y':np.array(x_n_pts)/max(x_n_pts)*max(x_grades_grid),
        'name':'number of points in x',
        'hovertext':x_n_pts
        }

        fig.append_trace(tracepts, 1, 1)
        fig.append_trace(tracegrid, 1, 1)
        fig.append_trace(tracenpts, 1, 1)

        tracepts = {
        'type':'scatter',
        'mode':'lines',
        'x':y_linspace,
        'y':y_grades_pts,
        'name':'points in y',
        'line':{'dash':'solid','color':'red'},
        }

        tracegrid = {
        'type':'scatter',
        'mode':'lines',
        'x':y_linspace,
        'y':y_grades_grid,
        'name':'grid in y',
        'line':{'dash':'solid','color':'blue'},
        }

        tracenpts = {
        'type':'bar',
        'x':y_linspace,
        'y':np.array(y_n_pts)/max(y_n_pts)*max(y_grades_grid),
        'name':'number of points in y',
        'hovertext':y_n_pts
        }

        fig.append_trace(tracepts, 2, 1)
        fig.append_trace(tracegrid, 2, 1)
        fig.append_trace(tracenpts, 2, 1)

        fig.layout.update(title='Swath plots')

        return pyo.iplot(fig)
















    




    