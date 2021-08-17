"""
This file containes the grid search over max_features and max_depth hyperparameters
of Random Tree Forest. It is based on the original file Adaboost_RTF_Shill_Bidding.py
and plots a heat map of the F-measure.

Created on: 06 April 2021
Authors:    Jonas Perolini & Raphael Uebersax
Libraries:  Sklearn, Pandas, Numpy, matplotlib
"""


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as ticker


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score


from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Get the directory of this file
my_path = os.path.dirname(__file__)



############### PARAMETERS TO SET DEPENDING ON SCENARIO #######################

### Set the number of FFolds for Cross-validation ########
FFolds = 10

### Set train/test split ratio ###########################
ratio = 0.3

### How to deal with unbalanced data set #################
Balancing = True


### Hyperparameters to search ###########################
# Adaboost and RTF
n_estimators_ada = 10
n_estimators_rtf = 10

# RTF 
max_depth = [2,3,4,5,6,7,8,9,None]      # Max tree depth to avoid overfitting
max_features = [2,3,4,5,6,7,8,9]


# Default parameters
min_samples_leaf = 2   # Min sample leaf to split a node 
learning_rate = 1.0 






####################### PLOT AND PRINT MANAGMENT #############################
### Plot parameters
Plot_conf = True
Plot_F1 = True
Plot_dataset = False
Plot_accuracy = False
Plot_F1_CV = False
Plot_3D = False

save = True
my_name = "Heat_map"






####################### DATASET CHARACTERISTICS ###############################

#           SHILL BIDDING DATASET
# ------------------------------------------------
# Bidder Tendency   |   Float between 0 and 1
# Bidding Ratio     |   Float between 0 and 1
# Succ. Outbidding  |   Float either 0, 0.5 or 1
# Last Bidding      |   Float between 0 and 1
# Auction Bids      |   Float between 0 and 1
# Starting Price    |   Float between 0 and 1
# Early Bidding     |   Float between 0 and 1
# Winning Ratio     |   Float between 0 and 1
# Auction Duration  |   Integer (duration in hours)

# Class: 0 for normal behaviour
#        1 otherwise.



########################### READ DATA FROM CSV ################################

# Read the data from the Shill Biddin csv file (add heading)
# https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset 
data = pd.read_csv(my_path + "/Shill_Bidding_Dataset.csv",delimiter = ",")

X = data.values[:,3:12]  # Select the nine attributes, not considering ID's
y = data.values[:,12]    # Select the class (target)

# Change cathegorical values to be integers 
y = y.astype(int)       # Change class from object to int
X = X.astype(float)     # Change attributes from object to float





Ada_accuracy_GS, RTF_accuracy_GS = [], []
Ada_accuracy_std_GS, RTF_accuracy_std_GS = [], []
Ada_F1_train_GS, Ada_F1_test_GS = [], []
RTF_F1_train_GS, RTF_F1_test_GS = [], []
time_train_ada, time_train_rtf = [], []
time_test_ada, time_test_rtf = [], []

maximum_depth = []
maximum_features = []
F1_heat_map = []
Accuracy_heat_map = []

for max_d in max_depth:
    if Ada_accuracy_GS:
        F1_heat_map.append(RTF_F1_test_GS)
        Accuracy_heat_map.append(RTF_accuracy_GS)
        RTF_accuracy_GS = []
        RTF_F1_test_GS = []
    
    for max_f in max_features:
        print("max feature: ",max_f, "max depth: ", max_d )

        
        #################### F-FOLDS CROSS-VALIDATION #################################
        
        # Initialize lists to save results over kfolds
        adaboost_score, adaboost_missclassified  = [], []
        adaboost_F1_train, adaboost_F1_test = [], []
        rtf_score, rtf_missclassified = [], []
        rtf_F1_train, rtf_F1_test = [], []
        
        # Intialize empty list to save execution time 
        time_train_ada_kfold, time_train_rtf_kfold = [], []
        time_test_ada_kfold, time_test_rtf_kfold = [], []
    
        
        # Perform ffolds time the algorithm on different train /test splits
        for fold in range(FFolds):

            # Boolean in case the training set is balanced
            balanced = False
            
            # Split the data into train and test samples 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio,
                                                                random_state = None,
                                                                shuffle = True)
    
                
            
            ##################### CHECK IF BALANCED ###########################
            
            # Find attributes of the shill and not bidder in the training set
            X_train_shill = X_train[ np.where(y_train == 1)[0],: ]
            X_train_normal = X_train[ np.where(y_train == 0)[0],: ]
        
            # Find the number of sample of each class
            nb_shill_train = X_train_shill.shape[0]
            nb_normal_train = X_train_normal.shape[0]
            diff_samples = nb_shill_train-nb_normal_train
            
            # Find the over and underrepresented class
            if diff_samples > 20:
                over_represented = X_train_shill
                under_represented = X_train_normal
                under_class = 0
                over_class = 1
                
            elif diff_samples < 20:
                under_represented = X_train_shill
                over_represented = X_train_normal
                under_class = 1
                over_class = 0
                
            else:
                balanced = True
                
        
            ################### BLANCING THE TRAINING SET #############################
                
            if Balancing and not(balanced):
                
                # Fit a Gaussian mixture on the malign data
                BIC = []
                GM = []
                for gm_folds in range(10):
                    gm = GaussianMixture(n_components=3, covariance_type='full',
                                         init_params='kmeans',
                                         random_state=None).fit(under_represented)
                    BIC.append(gm.bic(under_represented))
                    GM.append(gm)
            
                idx = np.argmin(np.array(BIC))
                gm = GM[idx]
                new_points, _ = gm.sample(n_samples= abs(diff_samples))
                new_y = np.repeat(under_class,abs(diff_samples))
    
                X_train = np.vstack((X_train, new_points))
                y_train = np.hstack((y_train, new_y))
                
                #shuffle the training set
                X_train, y_train = shuffle(X_train,y_train)
                
        
        
            
            ###################### TRAIN ADABOOST AND RTF #########################
    
            # AdaBoost model 
            ada = AdaBoostClassifier(n_estimators=n_estimators_ada,
                                     learning_rate = learning_rate)
            
            start_time = time.time()
            ada.fit(X_train,y_train)    
            time_train_ada_kfold.append(time.time()-start_time)
            
            y_pred_train_ada = ada.predict(X_train)
            F1_ada_train = f1_score(y_train, y_pred_train_ada)
            
            
            
            # RTF model 
            forest = RandomForestClassifier(n_estimators=n_estimators_rtf,
                                            max_features=max_f,
                                            max_depth = max_d,
                                            min_samples_leaf=min_samples_leaf)
            start_time = time.time()
            forest.fit(X_train,y_train)
            time_train_rtf_kfold.append(time.time()-start_time)
            
            y_pred_train_forest = forest.predict(X_train)
            F1_forest_train = f1_score(y_train,y_pred_train_forest)
            
            
            ###################### TEST ADABOOST AND RTF ##########################
            
            # Test AdaBoost
            start_time = time.time()
            y_pred_test_ada = ada.predict(X_test)
            time_test_ada_kfold.append(time.time()-start_time)
            
            F1_ada_test = f1_score(y_test, y_pred_test_ada)
            
            # Result Adaboost
            diff_ada = y_test-y_pred_test_ada
            nb_diff_ada_test = np.flatnonzero(diff_ada).shape[0]
            Score_ada = 1.0 - nb_diff_ada_test/y_test.shape[0]
    
        
            
            
            
            # Test RTF
            start_time = time.time()
            y_pred_test_forest = forest.predict(X_test)
            time_test_rtf_kfold.append(time.time()-start_time)
            
            F1_forest_test = f1_score(y_test, y_pred_test_forest)

            
            # Result RTF
            diff_forest = y_test-y_pred_test_forest
            nb_diff_forest_test = np.flatnonzero(diff_forest).shape[0]
            Score_forest = 1.0 - nb_diff_forest_test/y_test.shape[0]
        
            
        
            
            # Retain score of this fold
            adaboost_score.append(Score_ada)
            adaboost_missclassified.append(nb_diff_ada_test)
            adaboost_F1_train.append(F1_ada_train)
            adaboost_F1_test.append(F1_ada_test)
    
            
            rtf_score.append(Score_forest)
            rtf_missclassified.append(nb_diff_forest_test)
            rtf_F1_train.append(F1_forest_train)
            rtf_F1_test.append(F1_forest_test)

    
    
    
        
        ######################### PRINTING RESULTS K_folds ########################
        
        # Change lists to numpy arrays for mean calculation
        adaboost_score = np.array(adaboost_score)
        rtf_score = np.array(rtf_score)
        adaboost_missclassified = np.array(adaboost_missclassified)
        rtf_missclassified = np.array(rtf_missclassified)
        
        
        # Save results for GS
        Ada_accuracy_GS.append(np.mean(adaboost_score))
        RTF_accuracy_GS.append(np.mean(rtf_score))
        Ada_accuracy_std_GS.append(np.std(adaboost_score,ddof=1))
        RTF_accuracy_std_GS.append(np.std(rtf_score, ddof=1))
        
        Ada_F1_train_GS.append(np.mean(adaboost_F1_train))
        Ada_F1_test_GS.append(np.mean(adaboost_F1_test))
        RTF_F1_train_GS.append(np.mean(rtf_F1_train))
        RTF_F1_test_GS.append(np.mean(rtf_F1_test))
        
        
        time_train_ada.append(np.mean(np.array(time_train_ada_kfold)))
        time_train_rtf.append(np.mean(np.array(time_train_rtf_kfold)))
        time_test_ada.append(np.mean(np.array(time_test_ada_kfold)))
        time_test_rtf.append(np.mean(np.array(time_test_rtf_kfold)))
        
        
F1_heat_map.append(RTF_F1_test_GS)
Accuracy_heat_map.append(RTF_accuracy_GS)
        
            
        
################################# HEAT MAP ####################################
F1_heat_map = np.array(F1_heat_map)
Accuracy_heat_map = np.array(Accuracy_heat_map)

print("max f: ",max_features)
print("max d: ",max_depth)
print("map: ",F1_heat_map)


fig, ax = plt.subplots()

im = ax.imshow(F1_heat_map, origin="lower", cmap="Blues", alpha=0.9)


def myfmt(x, pos):
    return '{0:.2f}'.format(x)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(myfmt))



# We want to show all ticks...
ax.set_xticks(np.arange(len(max_features)))
ax.set_yticks(np.arange(len(max_depth)))

if max_depth[-1] == None:
    max_depth= max_depth[:-1]
    max_depth.append("0")
    
# ... and label them with the respective list entries
ax.set_xticklabels(max_features, fontsize=8)
ax.set_yticklabels(max_depth,fontsize=8)

# Loop over data dimensions and create text annotations.
for i in range(len(max_depth)):
    for j in range(len(max_features)):
        text = ax.text(j, i, ('%.2f' % F1_heat_map[i,j]).lstrip('0'),
                       ha="center", va="center", color="k", fontsize = 8)

ax.set_title("F-measure (mean)", fontsize=10)
ax.set_xlabel("max features",fontsize=10)
ax.set_ylabel("max depth",fontsize=10)


fig.tight_layout()

if save:
    name = "Figures/" + my_name + "_F1.pdf"
    plt.savefig(name)





fig, ax = plt.subplots()
im = ax.imshow(Accuracy_heat_map, origin="lower", cmap="Blues", alpha=0.9)

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, format=ticker.FuncFormatter(myfmt))

# We want to show all ticks...
ax.set_xticks(np.arange(len(max_features)))
ax.set_yticks(np.arange(len(max_depth)))

if max_depth[-1] == None:
    max_depth= max_depth[:-1]
    max_depth.append("0")
    
# ... and label them with the respective list entries
ax.set_xticklabels(max_features)
ax.set_yticklabels(max_depth)

# Loop over data dimensions and create text annotations.
for i in range(len(max_depth)):
    for j in range(len(max_features)):
        text = ax.text(j, i, ('%.2f' % Accuracy_heat_map[i,j]).lstrip('0'),
                       ha="center", va="center", color="k")

ax.set_title("Accuracy (mean)",fontsize=15)
ax.set_xlabel("max features",fontsize=15)
ax.set_ylabel("max depth",fontsize=15)
fig.tight_layout()

if save:
    name = "Figures/" + my_name + "_accuracy.pdf"
    plt.savefig(name)









        

    


