# Experimental NIC-based ML preprocessing pipeline
This repository hosts our implementation of the Nature Inspired course group project. The latter is a humble attempt to build an (almost) fully-NIC based ML preprocessing pipeline considering 3 main preprocessing / data engineering steps:

* Eliminating multi-collinearity by means of clustering using PSO (Particle Swarm Optimization)
* Feature Selection with **Cuckoo Seach** Algorithm: with the ***evopreprocess*** library

* Feature Transformation: using Genetic algorithms to find the combination of functions (one function per feature) that maximizes the average linear correlation.

## More context
The initial goal that we have set in this project was applying nature-inspired algorithms as optimizers to the dataset that contained various features that were supposed to predict population growth. The resulting dataset would be evaluated by regression Machine Learning (referred to as ML) models, and results of our preprocessing would be evaluated. However, after starting working on the project, it very soon became evident that the project would have had a
very small scope, and few opportunities to explore interesting
applications of Nature Inspired Algorithms (referred to as NIA).  

Therefore, our final project represents a significant extension
of the initial proposal. The latter constitutes of predicting
the population’s growth using a combination of Machine
Learning and Nature Inspired Computing techniques (referred
to as NIC). The final project takes this first experimental
step into a more complex and sophisticated direction. Our
team attempts to build a ML preprocessing pipeline almost
fully-based on NIC techniques and evaluates its performance
with the considered dataset as per with the initial proposal.. 

## Directory Details
* databases: used to save the datasets file used in the project. Currently restricted to the **Population's Growth** [dataset](https://github.com/Daru1914/NIC_Project/blob/final/databases/final_dataset.xlsx).
* notesbooks: a directory of our working code, implementing the different parts of the project
    * Feature_transformer.ipynb/py: the code for the feature transformation step both as JupyterNotebook and a Python file
    * multicollinearity_PSO.ipynb: the code for the multicollinearity step
    * preparing_data.ipynb: the code for preparing the **Population's Growth** dataset.
    * test_pipeline.ipynb: the code used for testing our pipeline's performance on the dataset in question.
    * Tpot_Generator: code for creating a TPOT regressor for any dataset and applied to the dataset in question
* models with_test_pipelines: pickles files to save the models run during the testing phase
* test_results: the results of different tests with different datasets
* tex: folder containing the source code and the accompagning necessary files for the final report.

## Further details
For more detailed explanation of the project, please consider reading our report about our [experimental project](https://github.com/Daru1914/NIC_Project/blob/final/tex/final_report.pdf)
