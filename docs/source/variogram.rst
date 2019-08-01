Variography
=============

This module houses tools for spatial continuity analysis and variogram calculation and modeling. All interactive functions opens an interactive parameters input widget, non interactive versions are usefull for scripting your workflow.

.. currentmodule:: pyLPM.gammapy

Variogram map
---------------

This program will do a map of spatial continuity. It is not required to do in a estimation, but can save time when you have to know the directions of anisotropy.
More cold the direction is in the map less variance it have.

.. autofunction:: varmap

.. autofunction:: interactive_varmap

Experimental variogram
-------------------------

It is calcullated using the equation:
ȣ=1/2*mean((x(u)-x(u+h))^2)
where:
ȣ is the spatial variance
x(u) is the value of the variable at u position
x(u+h) is the value of the variable whith h of distance
This equation returns the mean of the variance 

.. autofunction:: experimental

.. autofunction:: interactive_experimental

Variogram modeling
-----------------------

.. autofunction:: modelling

.. autofunction:: interactive_modelling
