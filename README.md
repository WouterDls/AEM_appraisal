[![DOI](https://zenodo.org/badge/527564756.svg)](https://zenodo.org/badge/latestdoi/527564756)

# The code AEM_appraisal in short
- Airborne EM methods generate enormous datasets for which fast quasi-2/3D inversion methods are prevailing, that may cause erroneous results.
- Our tool assesses quantitatively interpretable zones from inversion models obtained with 1D (approximate) forward models. 
- A forced modelling error approach allows the tool to work with an imperfect but computationally cheaper multidimensional modelling. 

For example, the following recovered model was obtained with a 1D forward model on Airborne EM data
![alt text](https://github.com/WouterDls/AEM_appraisal/blob/main/fieldcase_recovered_model.png)

Applying the appraisal method results in blanked out area's from the model that do not fit the multidimensionality of the data well.
![alt text](https://github.com/WouterDls/AEM_appraisal/blob/main/field_appraisal_imperfect.png)

## Summary
This code was used to create an appraisal tool which detects multidimensionality issues in Airborne EM inversion in which a 1D forward model was used. The results are published in Remote Sensing

Deleersnyder, W., Dudal, D., & Hermans, T. (2022). Novel Airborne EM Image Appraisal Tool for Imperfect Forward Modeling. Remote Sensing, 14(22), 5757. DOI: https://doi.org/10.3390/rs14225757

Abstract:

Full 3D inversion of time-domain Airborne ElectroMagnetic (AEM) data requires specialists’ expertise and a tremendous amount of computational resources, not readily available to everyone. Consequently, quasi-2D/3D inversion methods are prevailing, using a much faster but approximate (1D) forward model. We propose an appraisal tool that indicates zones in the inversion model that are not in agreement with the multidimensional data and therefore, should not be interpreted quantitatively. The image appraisal relies on multidimensional forward modeling to compute a so-called normalized gradient. Large values in that gradient indicate model parameters that do not fit the true multidimensionality of the observed data well and should not be interpreted quantitatively. An alternative approach is proposed to account for imperfect forward modeling, such that the appraisal tool is computationally inexpensive. The method is demonstrated on an AEM survey in a salinization context, revealing possible problematic zones in the estimated fresh–saltwater interface.
 
 ## Contents of the code
 This repository contains
 1. data and simulations: data, recoverd/inversion models from quasi-2D inversion, results from simulation scripts, intermediate results (so that they do not to be re-computed each time)
 2. scripts: those scripts generate the normalised gradient for the synthetic and field data case (for perfect and imperfect modelling)
 3. simulation codes: these codes maily use SimPEG to simulate the EM physics on a mesh. COARSE simulations can be easily computed on a single laptop. For other simulations, a HPC infrastructure is more suitable. All results of those simulations are in .cvs files in the data and simulations directory (no need to recompute)
 4. src: source files underlying the appraisal technique that is produced in the scrips. 
