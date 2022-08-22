[![DOI](https://zenodo.org/badge/527564756.svg)](https://zenodo.org/badge/latestdoi/527564756)

# AEM_appraisal
- Airborne EM methods generate enormous datasets for which fast quasi-2/3D inversion methods are prevailing, that may cause erroneous results.
- Our tool assesses quantitatively interpretable zones from inversion models obtained with 1D (approximate) forward models. 
- A forced modelling error approach allows the tool to work with an imperfect but computationally cheaper multidimensional modelling. 
 
 ## Contents
 This repository contains
 1. data and simulations: data, recoverd/inversion models from quasi-2D inversion, results from simulation scripts, intermediate results (so that they do not to be re-computed each time)
 2. scripts: those scripts generate the normalised gradient for the synthetic and field data case (for perfect and imperfect modelling)
 3. simulation codes: these codes maily use SimPEG to simulate the EM physics on a mesh. COARSE simulations can be easily computed on a single laptop. For other simulations, a HPC infrastructure is more suitable. All results of those simulations are in .cvs files in the data and simulations directory (no need to recompute)
 4. src: source files underlying the appraisal technique that is produced in the scrips. 
