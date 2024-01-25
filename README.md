# Implementation
Main source codes of DP training and DP prediction methods for random forest classifiers are in the ```mrf/src``` folder.
Source code used to generate the experimentation results and figures in the paper are in the ```mrf/exp``` folder. 
The data used in this study is provided in the ```data``` folder. 

# Dependencies
You'll need a working Python environment to run the code. 
We set up our environment through the Anaconda Python distribution which provides the conda package manager.
The required dependencies are specified in the file environment.yml.

Run the following command in the repository folder (where environment.yml is located) to create a separate environment 
and install all required dependencies in it:
```
conda env create --name mrf-paper --file=environment.yml
```

Our code uses external source codes that are in the ```external-code``` folder.

# Reproducing the results
The setup instructions are for Mac. Before running any code, activate the conda environment:
```
source activate ENVIRONMENT_NAME
```

To reproduce the results in the paper, set the current working directory to ```mrf/exp``` and run ```main.py``` with preferred configuration in ```mrf/exp/config.py```. The results will be saved under the ```temp``` folder as a csv format.
The following command runs with a default configuration: 
```
cd mrf/exp
python main.py
```

Figures for the results are generated via the Jupyber notebook ```pub-result.ipynb```. All figures are saved under the ```figures``` folder.

