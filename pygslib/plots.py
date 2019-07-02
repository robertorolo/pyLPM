import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

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

#############################################################################################################

def locmap(x, y, variable, categorical=False, title='', x_axis='Easting (m)', y_axis='Northing (m)', pointsize=8, colorscale='Jet', colorbartitle='', figsize=(700,700)):

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

        mask = np.isnan(variable)

        trace = {
            'type':'scatter',
            'mode':'markers',
            'x':x[~mask],
            'y':y[~mask],
            'marker':{'size':pointsize,'color':variable,'colorscale':colorscale,'showscale':True,'colorbar':{'title':colorbartitle}},
            'text':variable[~mask]
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

    dataf = data[~np.isnan(data)]

    statistics = '''
    n {}  <br />
    min {} <br />
    max {} <br />
    mean {}  <br />
    stdev {}  <br />
    cv {}  <br />
    '''.format(round(len(dataf),0), round(dataf.min(),2), round(dataf.max(),2),  round(dataf.mean(),2), round(np.sqrt(dataf.var()),2), round(np.sqrt(dataf.var())/dataf.mean(),2))

    if wt != None:

        mean, var = weighted_avg_and_var(dataf, wt)

        statistics = '''
        n {}  <br />
        min {} <br />
        max {} <br />
        mean {}  <br />
        stdev {}  <br />
        cv {}  <br />
        weighted
        '''.format(round(len(dataf),0),  round(dataf.min(),2), round(dataf.max(),2), round(dataf.mean,2), round(np.sqrt(var),2), round(np.sqrt(var)/mean),2)

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

def scatter2d(x, y, variable='kernel density', xy_line = True, regression_line = True, title='', x_axis='', y_axis='', pointsize=8, colorscale='Jet', colorbartitle='', figsize=(700,700)):

    traces = []
    
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
        'xaxis':{'title':x_axis,'scaleanchor':'y','zeroline':True},
        'yaxis':{'title':y_axis,'zeroline':True},
        'width':figsize[0],
        'height':figsize[1],
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)



    