
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import GridspecLayout
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from plotly.subplots import make_subplots
import plotly.express as px

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.express as px



def datasets():
	return ['Walker_Lake']






def descriptions(dataset): 
	descriptions = {}

	description['Walker_Lake'] = {'name': 'Walker_Lake',
								  'local': 'CAN', 
								  'type': '2D'}


def _location_data(local):
	gapminder = px.data.gapminder()
	gapminder['data_local'] = np.array([1000000 if gapminder["iso_alpha"].values[i] == 'local' else 0 for i in range(gapminder["iso_alpha"].shape[0])])
	gapminder['countries'] = gapminder["iso_alpha"]
	return gapminder


def describe(dataset_name):
	


	fig = px.scatter_geo(gapminder, locations='countries', hover_name="country", size='data_local',projection="natural earth")
	fig.show()


gapminder = px.data.gapminder()
gapminder['data_local'] = np.array([1000000 if gapminder["iso_alpha"].values[i] == 'CAN' else 0 for i in range(gapminder["iso_alpha"].shape[0])])
gapminder['countries'] = gapminder["iso_alpha"]

fig = px.scatter_geo(gapminder, locations='countries', hover_name="country", size='data_local',projection="natural earth")
fig.show()



 

from IPython.core.display import HTML
HTML('<a href="https://en.wikipedia.org/wiki/Walker_Lake_(Nevada)">link</a>')