import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats

#############################################################################################################

def weighted_avg_and_var(values, weights):
    """returns weighted average and mean.
    
    Arguments:
        values {array} -- array of data values
        weights {array} -- array of weights
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, variance

def isotopic_arrays(arrays):
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
    """plots pdf and cdf.
    
    Arguments:
        data {array} -- data values array
    
    Keyword Arguments:
        n_bins {int} -- number of bins (default: {20})
        wt {array} -- array of weights (default: {None})
        title {str} -- plot title (default: {''})
        x_axis {str} -- x axis title (default: {''})
        y_axis {str} -- y axis title (default: {''})
        cdf {bool} -- cdf plot flag (default: {False})
        figsize {tuple} -- figure size (default: {(600,600)})
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

def scatter2d(x, y, variable='kernel density', xy_line = True, best_fit_line=True, regression_line = True, title='', x_axis='', y_axis='', pointsize=8, colorscale='Viridis', colorbartitle='', figsize=(700,700)):

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
        slope {}
        '''.format(round(len(x),0), round(np.corrcoef([x,y])[1,0],2), round(slope,2))
    
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




    




    