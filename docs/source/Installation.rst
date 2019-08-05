Installation
==============

The first step for installing pyLPM in your machine is to install anaconda.

.. image:: figs/anacondalogo.png
    :width: 350px
    :align: center

Anaconda is a free and open-source[5] distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.)

`Anaconda download link <https://www.anaconda.com/distribution/>`_

.. warning::

	Check the option  *Add Anaconda to my PATH enviroment variable* during installation.

After installing Anaconda you must download and install pyLPM package.

`GitHub  download page <https://github.com/robertorolo/pyLPM/releases>`_

After downloading Source code in `tar.gz` format:

* Open anaconda prompt from windows start menu;
* Navigate to your download folder with the command `cd`;
* Intal package:

.. code:: python

	pip install pyLPM-x.x.x.tar.gz

The same approach works under linux too.

Now you can open a jupyter notebook and import pyLPM.

.. code:: python

	import pyLPM

