.. pyLPM documentation master file, created by
   sphinx-quickstart on Wed Jul 24 16:45:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyLPM's documentation!
=================================

pyLPM is a multi-plataform (windows and linux) free and open source python geostatistical library intended to be used in a jupyter notebook. Check `Walker Lake demo <https://pylpm.readthedocs.io/en/latest/demo%20wl.html#Walker-Lake-demo>`_ section in this documentation for a software overview. 
The software is composed by five modules:

* **GSLib** module is a wrapper on top of GSLib algorithms to permit a seamless integration with pyLPM.
* **Geostat algorithms** houses geostatistical algorithms developed by LPM or that are not available in the GSLib open source version.
* **Variography** module is packed with a full toolkit for spatial continuity analysis and interactive variogram modeling.
* **Utilities** is where the user will find functions to make the workflow easy and fast.
* **Plots** is a rich plotting functions library, based on plotly. All the plots are interactive.

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   Installation.rst
   jupyter.ipynb
   pandas.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   datasets.rst
   gslib.rst
   geostatalgo.rst
   variogram.rst
   utils.rst
   plots.rst

.. toctree::
   :maxdepth: 2
   :caption: Demos:
   
   demo wl.ipynb


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
