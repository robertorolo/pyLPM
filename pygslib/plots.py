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
    average = numpy.average(values, weights=weights)
    # Fast and numerically precise:
    variance = numpy.average((values-average)**2, weights=weights)
    return average, variance

#############################################################################################################

def histogram(data, n_bins=20, wt=None, title='', x_axis='', y_axis='', cdf=False, figsize=(600,600)):
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

    statistics = '''
    n = {}
    mean = {}
    variance = {}
    standard deviation = {}
    cv = {}
    not weighted
    '''.format(len(data), data.mean(), data.var(), np.sqrt(data.var()), np.sqrt(data.var())/data.mean())

    if wt != None:

        mean, var = weighted_avg_and_var(data, wt)

        statistics = '''
        n = {}
        mean = {}
        variance = {}
        standard deviation = {}
        cv = {}
        weighted
        '''.format(len(data), mean, variance, np.sqrt(variance), np.sqrt(variance)/mean)

    traces = []

    hist, bin_edges = np.histogram(data, bins=n_bins, weights=wt, density=True)
    
    trace = {
        'type':'bar',
        'x':bin_edges,
        'y':hist,
        'name':'pdf'
    }

    traces.append(trace)

    if cdf == True:

        hist = np.cumsum(hist*np.diff(bin_edges))
        hist = np.insert(hist, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)
        print(bin_edges, hist)

        trace = {
            'type':'scatter',
            'mode':'lines',
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
    }

    fig = go.Figure(traces, layout)

    return pyo.iplot(fig)

'''def scatter(x, y, z=None, variable=None, title='', x_axis='', y_axis=''):

    if z == None:

        trace = 

    else:

        trace = {
            'type':'scatter',
            'mode':'markers'
        }'''

    