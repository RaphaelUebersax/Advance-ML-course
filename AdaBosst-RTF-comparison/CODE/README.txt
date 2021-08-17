The report used mutliple different codes for testing different scenarios. An overview of the content of each 
code is given to falicitate the reader's task to find what he needs:

1) Adaboost_RTF_Shill_Bidding.py:

This code is the main python script that executes all different scenarios of the report on the Shill Bidding
dataset. It takes many parameters to be set in the begining of the code to decide what needs to be tested and
was results are desired to be printed to the consol or plotted.


2) Adaboost_RTF_Toy.py

This code is the equivalent for the first one but using the toy dataset rather than the Shill Bidding one. It is
thus more based on plotting 2d-decision boudaries rather than F-measure.


3) Heat_map.py

This script run the grid search for the heat_map (see hyperparameters tunning) and plots it.


4) plot_confusion_matrix.py

This script contains a helper function that plots the confusion matrix.


5) get_GMM_param.py

This script has been used to find optimal parameters of the GMM for the upsampling method. This is done by plotting
AIC and BIC metrics on the Shill data and using the elbow criterion.