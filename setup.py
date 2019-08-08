 
from distutils.core import setup
setup(
  name = 'pyLPM',         # How you named your package folder (MyLib)
  packages = ['pyLPM'],   # Chose the same as "name"
  version = '1.3.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Educational geostatistical library based on GSLib algorithms',   # Give a short description about your library
  author = 'LPM - UFRGS',                   # Type in your name
  author_email = 'robertorolo@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/robertorolo/pyLPM',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/robertorolo/pyLPM/archive',    # I explain this later on
  keywords = ['geostatistics', 'kriging', 'variogram'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy==1.3.0',
          'numba',
          'plotly==4.0.0',
          'pandas',
          'IPython',
          'ipywidgets==7.5.0',
          'matplotlib',
          #'sklearn',
          'scikit-learn==0.21.2'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Education',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    #'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    #'Programming Language :: Python :: 3.4',
    #'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7'
  ],
  include_package_data = True,
  package_data={'pyLPM': ['gslib90/*exe', 'datasets/*txt']}
)
