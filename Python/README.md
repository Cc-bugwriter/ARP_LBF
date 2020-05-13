Version: 1.03
- optimize main func
- add data-version argument in main func
- fix save bug
----------------------------------------
Version: 1.02
- add adaptive hyperparameter search
- add model save and load function
----------------------------------------
Version: 1.01
- add random search, optimize default structure
----------------------------------------
# MLP Receptron
MLP Receptron is a Multi-layer artificial intelligence network，which base on python library.

![standard-readme compliant](https://github.com/Cc-bugwriter/ARP_LBF/blob/master/ARP_Programm_Prozess.png)

# Installation
MLP Receptron depends on a few of popular data science library. 
Before using MLP Receptron please install those necessary libraries in your local enviroment:
## Table of Libraries
- [math]
- [numpy] -- 1.16.4
- [pandas] -- 0.24.2
- [seaborn] -- 0.9.0
- [scipy] -- 1.2.1
- [keras] -- 2.2.4
- [zipfile]
- [sklearn] -- 0.20.3
- [warnings]
- [matplotlib]
- [tensorflow] -- 1.13.1

### Installing with Anaconda
The simplest way to install not only pandas, but Python and the most popular packages that make up the SciPy stack (IPython, NumPy, Matplotlib, …) is with [Anaconda](https://www.anaconda.com/distribution/), a cross-platform (Linux, Mac OS X, Windows) Python distribution for data analytics and scientific computing.
Installation instructions for [Anaconda](https://www.anaconda.com/distribution/) [can be found here.](https://docs.continuum.io/anaconda/install/)

### Installing from PyPI
All libraries can also be installed via pip from [PyPI](https://pypi.org/)
```bash
pip install numpy
pip install pandas
pip install sklearn
pip install matplotlib
```

### Installing in Pycharm (recommended)
If you have installed Pycharm IDE in your Laptop, Installing in Pycharm is the recommended installation method
- [1. clone this repository in your PC](https://github.com/Cc-bugwriter/ARP_LBF.git)
- [2. open Pycharm and create a project from Python folder, which is located in your cloned path]
- [3. after creating project, click file --> Project: Python --> Project Interpreter]
- [4. click the settings button --> Add --> Conda Environment --> Python Version 3.7 -> Ok]
- [5. back to Settings interface --> click  +  button --> Available Packages]
- [6. manually install the libraries --> speicial version must be check the same with above list ](##Table of Libraries)

## Usage
please open .py file in Pycharm IDE
choose [MLP_Preceptron] and enter ['shift + F10']

## Contributing
Pull requests are welcome. For major changes, please first to discuss what you would like to change in Whatsapp Group.

Please make sure to update tests as appropriate.

## Description
hier is a simple description for each script
### Function
[MLP_Preceptron] is already packaged script, which relay on node in each subfiles.
[MLP_Regression] and [MLP_Classifier] are debug version, which have same sub-funtion like in [MLP_Preceptron].

[MLP_Regression] check for :
- [load data set]
- [preprocessing]
- [model Regressor]
- [evaluation]
- [hyper Search]


[MLP_Classifier] check for :
- [load data set]
- [preprocessing]
- [model Regressor]
- [evaluation]
- [hyper Search]

### Nodes
all necessary nodes can belong to three blockes:
- [Preprocessing]
-- [pre_processing.py]
- [Regression]
-- [Regressor.py]
- [Classification]
-- [Classifier.py]
- [Evaluation]
-- [confusion_matrix.py], [hyper_search.py], [plot_learning_curve.py], [plot_regularization.py]
