import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd 
import numpy as np 
from IPython.core.display import HTML
import os 
import sys 
import ipywidgets as widgets
import pkg_resources

#defining datasets folder inside package folder
global DATA_PATH
DATA_PATH = pkg_resources.resource_filename('pyLPM', 'datasets/')

def dataset_list():
	
	"""Prints all avaliables datasets as a list
	"""
	
	print(['Walker_Lake'])

def dataset(dataset_name):
	"""Return dataset as a pandas DataFrame
	
	Args:
		dataset_name (str): dataset name
	
	Returns:
		DataFrame: dataset as a DataFrame
	"""
	if dataset_name == 'Walker_Lake':
		X, Y, Cu, Au = np.loadtxt(DATA_PATH+"Walker_Lake.txt" , skiprows = 6, unpack = True)
		df = pd.DataFrame(np.array([X,Y,Cu,Au]).T, columns =['X', 'Y', 'Cu', 'Au'])
		return df

def _description_html(dataset_name):
	"""Get a wikipedia description of the dataset
	
	Args:
		dataset_name (str): dataset name
	"""
	if dataset_name == "Walker_Lake":
		h1 = widgets.HTML('<h1> Walker Lake (Nevada) </h1>'\
		    '<p><cite> Walker Lake is a natural lake, in the Great Basin in western Nevada in the United States." \
			" It is 11 mi (17 km) long and 5 mi (8 km) wide, in northwestern Mineral County along the east " \
			" side of the Wassuk Range, about 75 mi (120 km) southeast of Reno. The lake is fed from the north " \
			" by the Walker River and has no natural outlet except absorption and evaporation. The community of " \
			" Walker Lake, Nevada, is found along the southwest shore. <br>  <br> </cite></p>'\
		    '<p><cite> The lakebed is a remnant of prehistoric Lake Lahontan that covered much of northwestern ' \
			"Nevada during the ice age. Although the ancient history of Walker Lake has been extensively " \
			"studied by researchers seeking to establish a climatic timeline for the region as part of " \
			"the Yucca Mountain Nuclear Waste Repository study, this research has raised many puzzling " \
			"questions. Unlike Pyramid Lake, the lake itself has dried up several times since the end of "\
			"the Pleistocene, probably due to natural diversions of the Walker River into the Carson Sink "\
			"approximately 2,100 years ago. Also, this research found no evidence that the Walker Lake basin "\
			"contained water during the Lake Lahontan highstand, although based on the surface elevation "\
			"of the highstand evidenced elsewhere in the region it must have. <br>  <br> </cite></p> "\
		    " <p><cite> Walker Lake is the namesake of the geological trough in which it sits, and which extends from "\
			 "Oregon to Death Valley and beyond, the Walker Lane. It was named after Joseph R. Walker, a mountain "\
			 "man who scouted the area with John C. Frémont in the 1840s. <br>  <br> </cite></p> "\
		     "<p><cite> The area around the lake has long been inhabited by the Paiute. Beginning in the mid-19th century "\
		     "the introduction of agriculture upstream of Walker Lake has resulted in the water from the Walker "\
		     "River and its tributaries being diverted for irrigation. These diversions have resulted in a severe " \
		     "drop in the level of the lake. Upstream water users have exploited the Walker River for profit, " \
		     "resulting in the destruction of Walker Lake. According to the USGS, the level dropped approximately " \
		     "181 ft (55 m) between 1882 and 2016. By June, 2016, the lake level was 3,909 feet above sea level. " \
		     "This is the lowest lake elevation since measurement began in 1882. <br>  <br> </cite></p> "\
		     " <p><cite> The lower level of the lake has resulted in a higher concentration of total dissolved solids (TDS). " \
			 " As of the spring of 2016, the TDS concentration had reached 26 g/L, well above the lethal limit for " \
			 " most of the native fish species throughout much of the lake. Lahontan cutthroat trout no longer occur " \
			 "in the lake and recent work by researchers indicates that the lake's tui chub have declined dramatically " \
			 "and may soon disappear as the salinity levels are lethal to tui chub eggs and young chubs. The decline of " \
			 " the lake's fishery is having a dramatic impact on the species of birds using the lake. By 2009, the town "\
			 " of Hawthorne canceled its Loon Festival because the lake, once a major stopover point for migratory loons, "\
			 " could no longer provide enough chub and other small fish to attract many loons.  <br> </cite></p> ")		
		h2 = widgets.HTML('<a href="https://en.wikipedia.org/wiki/Walker_Lake_(Nevada)">Font: Wikipedia</a>')
		display(h1,h2)


def descriptions(dataset_name):
	"""Show dataset info
	
	Args:
		dataset_name (str): show dataset info
	"""
	descriptions = {}
	descriptions['Walker_Lake'] = {'name': 'Walker_Lake',
								  'local': 'CAN'}

	if dataset_name == 'Walker_Lake':
		_description_html('Walker_Lake')
		_graphs([[56.1304],[106.3468]], "Walker_Lake")
		
def _graphs(local, dataset_name):

	if dataset_name == "Walker_Lake":

		df = dataset(dataset_name)

		#display(df.describe())

		fig = make_subplots(rows=2, cols=2,
			column_widths=[0.6, 0.4],row_heights=[0.4, 0.6],
			specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "scatter"}],
           [            None                    , {"type": "histogram"}]])

		fig.add_trace(go.Scattergeo(lat=pd.Series([38.695488]),
	                  lon=pd.Series([-118.71734]),
	                  mode="markers",
	                  marker=dict(size=30, opacity=0.8)), row=1, col=1)

		fig.add_trace(go.Histogram(x=df["Cu"], marker_color='#66CC66',opacity=0.75, name='Histogram Cu'),row=1, col=2)
		fig.add_trace(go.Scatter(x=df["X"], y=df["Y"], mode='markers', text =df["Cu"], marker=dict(color=df["Cu"], colorscale='Viridis')), row=2, col=2)

		fig.update_geos(
		projection_type="orthographic",
	    lakecolor="LightBlue",
	    landcolor="white",
    	oceancolor="MidnightBlue",
    	showocean=True)

		fig.update_layout(
	    template="plotly_white",
	    margin=dict(r=10, t=25, b=40, l=60))

		fig.show()