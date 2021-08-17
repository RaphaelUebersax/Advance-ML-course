"""
This file containes code in order to analyse two classification method: Adaboost
and Random Tree Forest. In this file, both algorithms are tested on the Shill
Bidding dataset with multiple different scenarios.

Created on: 06 April 2021
Authors:    Jonas Perolini & Raphael Uebersax
Libraries:  Sklearn, Pandas, Numpy, matplotlib
"""

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import time

from plot_confusion_matrix import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

# Get the directory of this file
my_path = os.path.dirname(__file__)



############### PARAMETERS TO SET DEPENDING ON SCENARIO #######################

# This section needs to be looked with care before running experiments. Many 
# different parameters allow for changing the setup of the simulation depending
# on the hyperparameters and the different scenario that want to be tested. If
# the normal configuration using the basic dataset want to be executed, all
# boolean parameters need to be set to False
###############################################################################

### Set the number of FFolds for Cross-validation ########
FFolds = 10

### Set train/test split ratio ###########################
ratio = 0.3

### Hyperparameters to search ###########################
# Adaboost and RTF
nb_estimators_ada = [100]
nb_estimators_rtf = [100]
#nb_estimators = [2,10,25,50,75,100,200]
#nb_estimators = [2,10,25,50]

# Adaboost 
learning_rate = 1.0 

# RTF 
max_depth = None       # Max tree depth to avoid overfitting
max_features = "auto"
min_samples_leaf = 2   # Min sample leaf to split a node

### Boolean parameter that decides if PCA is performed on the dataset or not
DO_PCA = False

### How to deal with missing values ######################
Miss_Val = False                 # Boolean that decides if random values are set to Nan
replace_by_median = False        # 1) Median input for missing values
replace_using_prediction = False # 2) Use predictor to find fit for missing value

### How to deal with unbalanced data set #################
Balancing = True  # Boolean that decideds if the dataset is preproceed with a balancing method

### Add noise to data in percent
Noise = False    # Boolean that decides if noise is added at training on the dataset
noise = 0.5      # Amount of noise added (noise * Covariance)

### Sensitivity to outliers (misslabel some data)
outliers = False # Boolean that decides if some training data should be mislabelled
freq = 0.95      # ratio of data that is correctly labelled




####################### PLOT AND PRINT MANAGMENT #############################

# In this section, the user can decide what kind of metric and what kind of plot
# he is interested in.
###############################################################################

### Plot parameters
Plot_conf = False          # Plot confusion matrix
Plot_F1 = True             # Plot F1 measure (1-set of hyperparameters)
Plot_dataset = False       # Plot_dataset in 3d when using PCA
Plot_accuracy = False      # Plot the accuracy over multiple hyperparameters
Plot_F1_CV = False         # Plot F-measure over multiple hyperparameters
Plot_3D = False            # Plot 3d plot with PCA preprocessed

### Save plots or not
save = False
my_name = "Outliers_shill_5"

### Print parameters
Print_kfolds = False       # Print the k-folds during running time
Print_balancing = False    # Print the balancing method used while running
Print_results = True       # Print results of accuracy
Print_nb_sample_each_class = False   # Print number of samples of each class




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

# Normalize the data if PCA as auction duration take too much importance
if DO_PCA:
    X = (X)/np.std(X,axis=0, ddof=1)
    
# Remove information about some attributes of the data
if Miss_Val:
    missing = np.random.choice(a=[False, True], size=X.shape, p=[0.9,0.1])
    X[missing] = np.nan
    




##################### DEALING WITH MISSING VALUES #############################

if (replace_using_prediction and replace_by_median):
    raise Exception('Select only one method for replacing missing values.')

# Replace missing value with median of attribute (SimpleInputter)
elif replace_by_median:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(X)
    X = imp_mean.transform(X)
    print("Missing values replaced with median")

# Replace missing value with predictor (choose estimator model in begining of file)
elif replace_using_prediction:
    imp_mean = IterativeImputer(random_state=0, estimator = KNeighborsRegressor())
    data = np.c_[X, y]
    imp_mean.fit(data)
    data = imp_mean.transform(data)
    X = data[:,:-1]
    print("Missing values replaced with prediction")
    
else:
    print('No missing values')


# Number of estimators
total_est = len(nb_estimators_ada)

# Initialize empty lists to store results over grid search
Ada_accuracy_GS, RTF_accuracy_GS = [], []
Ada_accuracy_std_GS, RTF_accuracy_std_GS = [], []
Ada_F1_train_GS, Ada_F1_test_GS = [], []
RTF_F1_train_GS, RTF_F1_test_GS = [], []
time_train_ada, time_train_rtf = [], []
time_test_ada, time_test_rtf = [], []

for n_estimators_ada, n_estimators_rtf in zip(nb_estimators_ada, nb_estimators_rtf):
    #print("Computing results for hyperparameter ",iteration+1, " out of ", total_est )
    
    #################### F-FOLDS CROSS-VALIDATION #################################
    
    # Initialize lists to save results over kfolds
    adaboost_score, adaboost_missclassified  = [], []
    adaboost_F1_train, adaboost_F1_test = [], []
    rtf_score, rtf_missclassified = [], []
    rtf_F1_train, rtf_F1_test = [], []
    all_y_pred_test_ada, all_y_test = [], []
    all_y_pred_test_forest = []
    adaboost_TPR, adaboost_FPR = [], []
    rtf_TPR, rtf_FPR = [], []
    
    # Intialize empty list to save execution time 
    time_train_ada_kfold, time_train_rtf_kfold = [], []
    time_test_ada_kfold, time_test_rtf_kfold = [], []

    
    if Print_kfolds:
        print("")
        print("Start training and testing over ", str(FFolds),"-folds:")
        
        
        
    # Perform ffolds time the algorithm on different train /test splits
    for fold in range(FFolds):
        if Print_kfolds: print("  Fold number ", str(fold+1))
    
        # Boolean in case the training set is balanced
        balanced = False
        
        # Split the data into train and test samples 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio,
                                                            random_state = None,
                                                            shuffle = True)

        # Perform PCA on training set, keeping 3 dimension. Also applies
        # transformation of the testing set.
        if DO_PCA: 
            pca = PCA(n_components=3)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            print(pca.explained_variance_ratio_)
                 
        # Adds gaussian noise on the training data
        if Noise:
            X_train = X_train + np.random.normal(np.zeros(X_train.shape[1]),
                                                 noise*np.std(X_train,axis=0, ddof=1),
                                                 X_train.shape) 
            
        # Mislabells some training data
        if outliers:
            idx = np.random.choice(a=[0, 1], size=y_train.shape, p=[freq,1-freq])
            y_train = y_train + idx * (1 - 2*y_train)
            
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
            
            BIC = []
            GM = []
            
            # Fit a GMM on underrepresented class using 10-folds CV and taking
            # the optimal one
            for gm_folds in range(10):
                gm = GaussianMixture(n_components=3, covariance_type='full',
                                     init_params='kmeans',
                                     random_state=None).fit(under_represented)
                BIC.append(gm.bic(under_represented))
                GM.append(gm)
        
            idx = np.argmin(np.array(BIC))
            gm = GM[idx]
            
            # Generate new points using the GMM's distribution
            new_points, _ = gm.sample(n_samples= abs(diff_samples))
            new_y = np.repeat(under_class,abs(diff_samples))

            # Add new points to training data
            X_train = np.vstack((X_train, new_points))
            y_train = np.hstack((y_train, new_y))
            
            #shuffle the training set
            X_train, y_train = shuffle(X_train,y_train)
            
            if Print_balancing: print("      Balance training set with upsampling")
    
    
        if Print_nb_sample_each_class:
            X_train_shill = X_train[ np.where(y_train == 1)[0],: ]
            X_train_norm = X_train[ np.where(y_train == 0)[0],: ]
            nb_shill_train = X_train_shill.shape[0]
            nb_norm_train = X_train_norm.shape[0]
            
            print("nb_cheater training", nb_shill_train)
            print("nb_normal training", nb_norm_train)
            
            
            X_test_shill = X_train[ np.where(y_test == 1)[0],: ]
            X_test_norm = X_train[ np.where(y_test == 0)[0],: ]
            nb_shill_test = X_test_shill.shape[0]
            nb_norm_test = X_test_norm.shape[0]
            
            print("nb_cheater testing", nb_shill_test)
            print("nb_normal testing", nb_norm_test)
        
        
    
        ###################### TRAIN ADABOOST AND RTF #########################

        # AdaBoost model 
        ada = AdaBoostClassifier(n_estimators=n_estimators_ada,
                                 learning_rate = learning_rate)
        
        # Train the classifier
        start_time = time.time()
        ada.fit(X_train,y_train)    
        time_train_ada_kfold.append(time.time()-start_time)
        
        # Results on training set
        y_pred_train_ada = ada.predict(X_train)
        F1_ada_train = f1_score(y_train, y_pred_train_ada)
        
        
        
        # RTF model 
        forest = RandomForestClassifier(n_estimators=n_estimators_rtf,
                                        max_features=max_features,
                                        max_depth = max_depth,
                                        min_samples_leaf=min_samples_leaf)
        
        # Trai the classifier
        start_time = time.time()
        forest.fit(X_train,y_train)
        time_train_rtf_kfold.append(time.time()-start_time)
        
        # Results on training set
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

    
        
        # Store prediction adabooste for confusion matrix
        all_y_pred_test_ada.extend(y_pred_test_ada)
        all_y_test.extend(y_test)
        
        
        
        # Test RTF
        start_time = time.time()
        y_pred_test_forest = forest.predict(X_test)
        time_test_rtf_kfold.append(time.time()-start_time)
        
        F1_forest_test = f1_score(y_test, y_pred_test_forest)
        
        # Result RTF
        diff_forest = y_test-y_pred_test_forest
        nb_diff_forest_test = np.flatnonzero(diff_forest).shape[0]
        Score_forest = 1.0 - nb_diff_forest_test/y_test.shape[0]
        
        # Store prediction rtf for confusion matrix
        all_y_pred_test_forest.extend(y_pred_test_forest)
        
    
        
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
    
    Ada_F1_train_GS.append(adaboost_F1_train)
    Ada_F1_test_GS.append(adaboost_F1_test)
    RTF_F1_train_GS.append(rtf_F1_train)
    RTF_F1_test_GS.append(rtf_F1_test)
    
    time_train_ada.append(np.mean(np.array(time_train_ada_kfold)))
    time_train_rtf.append(np.mean(np.array(time_train_rtf_kfold)))
    time_test_ada.append(np.mean(np.array(time_test_ada_kfold)))
    time_test_rtf.append(np.mean(np.array(time_test_rtf_kfold)))
    
        
    
    # Confusion matrix
    classes = ["N", "S"]
    if Plot_conf:
        plot_confusion_matrix(all_y_pred_test_ada, all_y_test,
                              classes, "Adaboost", FFolds)
        plot_confusion_matrix(all_y_pred_test_forest, all_y_test,
                              classes, "Random Forest", FFolds)
    
    
    # Print results
    if Print_results:
        print("\n")
        print("Results:")
        print("AdaBoost wrongly classified at testing: ",np.round(np.mean(adaboost_missclassified)),
              "\nScore AdaBoost: ",np.mean(adaboost_score),"\n")
        print("RTF wrongly classified at testing: ",np.round(np.mean(rtf_missclassified)),
              "\nScore RTF: ",np.mean(rtf_score),"\n")
        
    

    if Plot_F1:
        # Create F1 measure lists for boxplot
        F1_adaboost = [adaboost_F1_train, adaboost_F1_test]
        F1_rtf = [rtf_F1_train, rtf_F1_test]
        
        # Plot boxplot of F1 measure
        fig, axes = plt.subplots(1,2,figsize = (10,5), sharey=True)
        axes[0].set_title('Adaboost', fontsize=22)
        axes[0].boxplot(F1_adaboost, 0,'',positions=[1,2],widths=0.5)
        axes[0].set_xticklabels(["Train","Test"], fontsize=25)
        axes[1].set_title('Random Forest', fontsize=22)
        axes[1].boxplot(F1_rtf ,0,'',positions=[1,2],widths=0.5)
        axes[1].set_xticklabels(["Train","Test"], fontsize=25)
        
        for ax in axes.flatten():
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_ylim(bottom=None, top=1.01, auto = True)
            ax.tick_params(axis='both', which='major', labelsize=14)
        if save:
            name = "Figures/" + my_name + ".pdf"
            plt.savefig(name)
        plt.show()
        

        

############################# PRINTING RESULTS GridSearch #####################

ada_score_lower = np.array(Ada_accuracy_GS)-np.array(Ada_accuracy_std_GS)
ada_score_upper = np.array(Ada_accuracy_GS)+np.array(Ada_accuracy_std_GS)
rtf_score_lower = np.array(RTF_accuracy_GS)-np.array(RTF_accuracy_std_GS)
rtf_score_upper = np.array(RTF_accuracy_GS)+np.array(RTF_accuracy_std_GS)

if Plot_accuracy:
    if (nb_estimators_ada != nb_estimators_rtf):
        raise Exception("Different estimators for ada and rtf!!!")
        
    ax1 = plt.subplot(211)
    ax1.plot(nb_estimators_ada, Ada_accuracy_GS,'ko-', label="Adaboost")
    ax1.fill_between(nb_estimators_ada, ada_score_lower, ada_score_upper,
                     color='black', alpha=0.3)
    ax1.plot(nb_estimators_rtf, RTF_accuracy_GS,'bo-', label="RTF")
    ax1.fill_between(nb_estimators_rtf, rtf_score_lower, rtf_score_upper,
                     color='blue', alpha=0.3)
    ax1.set_xticks(nb_estimators_rtf)
    ax1.set_xlabel("Nb Estimators", fontsize=14)
    ax1.set_ylabel("Accuracy", fontsize=14)
    ax1.legend(fontsize=12)
    
    ax2 = plt.subplot(223)
    ax2.plot(nb_estimators_ada, time_train_ada, 'ko-')
    ax2.plot(nb_estimators_rtf, time_train_rtf, 'bo-')
    ax2.set_xlabel("Nb Estimators",fontsize=14)
    ax2.set_xticks([nb_estimators_rtf[0]]+nb_estimators_rtf[2:])
    ax2.set_yticks([0.00,1.00, 2.00])
    ax2.set_yticklabels(['0.00', '1.00', '2.00'])
    ax2.set_title('Training',fontsize=14)
    ax2.set_ylabel('Time [s]',fontsize=14)
    
    ax3 = plt.subplot(224)
    ax3.plot(nb_estimators_ada, time_test_ada, 'ko-')
    ax3.plot(nb_estimators_rtf, time_test_rtf, 'bo-')
    ax3.set_xlabel("Nb Estimators",fontsize=14)
    ax3.set_xticks([nb_estimators_ada[0]]+nb_estimators_ada[2:])
    ax3.set_title('Testing',fontsize=14)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.75)
    if save:
        name = "Figures/" + my_name + ".pdf"
        plt.savefig(name)




if Plot_F1_CV:
    fig, axes = plt.subplots(2,2,sharey=True, figsize = (10,10),
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

        
    axes[0][0].set_ylabel('F-measure', fontsize=22)
    axes[0][0].set_title('Adaboost Train', fontsize=22)
    axes[0][0].boxplot(Ada_F1_train_GS, 0,'',positions=np.arange(1,len(nb_estimators_ada)+1),widths=0.5)
    axes[0][0].set_xticklabels(nb_estimators_ada)
    axes[0][1].set_title('Adaboost Test',fontsize=22)
    axes[0][1].boxplot(Ada_F1_test_GS, 0,'',positions=np.arange(1,len(nb_estimators_ada)+1),widths=0.5)
    axes[0][1].set_xticklabels(nb_estimators_ada)
    axes[1][0].set_ylabel('F-measure', fontsize=22)
    axes[1][0].set_title('RTF Train',fontsize=22)
    axes[1][0].boxplot(RTF_F1_train_GS ,0,'',positions=np.arange(1,len(nb_estimators_rtf)+1),widths=0.5)
    axes[1][0].set_xticklabels(nb_estimators_rtf)
    axes[1][0].set_xlabel("Nb Estimators",fontsize=22)
    axes[1][1].set_title('RTF Test',fontsize=22)
    axes[1][1].boxplot(RTF_F1_test_GS ,0,'',positions=np.arange(1,len(nb_estimators_rtf)+1),widths=0.5)
    axes[1][1].set_xticklabels(nb_estimators_rtf)
    axes[1][1].set_xlabel("Nb Estimators",fontsize=22)
    
    for ax in axes.flatten():
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_ylim(bottom=None, top=1.00, auto = True)
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    if save:
        name = "Figures/" + my_name + ".pdf"
        plt.savefig(name)
        

    
    
########################### PLOT 3D WITH PCA ################################## 
    
    
    
#plot 

# Creating plot
#ax.scatter3D(X_test[y_test == 1,0], X_test[y_test == 1,1],
#             X_test[y_test == 1,2], color = "green")
#ax.scatter3D(X_test[y_test == 0,0], X_test[y_test == 0,1],
#             X_test[y_test == 0,2], color = "red")
#plt.title("simple 3D scatter plot")

if Plot_3D:
    plt.figure()
    ax = plt.axes(projection ="3d")
    
    idx1 = y_test == 1
    idx2 = y_test == 0
    
    # Compute TP, TN, FP and FN
    Correct_X = X_test[(y_test-y_pred_test_forest) == 0,:]
    Correct_y = y_test[(y_test-y_pred_test_forest) == 0]
    Wrong_X = X_test[(y_test-y_pred_test_forest) != 0,:]
    Wrong_y = y_test[(y_test-y_pred_test_forest) != 0]
    
    # Plot the points with respective color
    ax.scatter3D(Wrong_X[Wrong_y == 1,0], Wrong_X[Wrong_y == 1,1], 
                 Wrong_X[Wrong_y == 1,2], color = "black",label = "FN")
    ax.scatter3D(Wrong_X[Wrong_y == 0,0], Wrong_X[Wrong_y == 0,1], 
                 Wrong_X[Wrong_y == 0,2], color = "blue",label = "FP")
    
    ax.scatter3D(Correct_X[Correct_y == 1,0],Correct_X[Correct_y == 1,1],
                 Correct_X[Correct_y == 1,2], color = "red",label = "TP")
    ax.scatter3D(Correct_X[Correct_y == 0,0],Correct_X[Correct_y == 0,1],
                 Correct_X[Correct_y == 0,2], color = "green",label = "TN")
    
    ax.legend()
    plt.title("Random forest final prediction test")
    plt.show()
    
    
    
    
    plt.figure()
    ax = plt.axes(projection ="3d")
    
    Correct_X_train = X_train[(y_train-y_pred_train_forest) == 0,:]
    #Correct_X_train = Correct_X_train[np.arange(0,y_train.shape[0],4),:]
    Correct_y_train = y_train[(y_train-y_pred_train_forest) == 0]
    #Correct_y_train = Correct_y_train[np.arange(0,y_train.shape[0],4)]
    
    Wrong_X_train = X_train[(y_train-y_pred_train_forest) != 0,:]
    Wrong_y_train = y_train[(y_train-y_pred_train_forest) != 0]
    
    ax.scatter3D(Wrong_X_train[Wrong_y_train == 1,0],Wrong_X_train[Wrong_y_train == 1,1],
                 [Wrong_y_train == 1,2], color = "black",label = "FN")
    ax.scatter3D(Wrong_X_train[Wrong_y_train == 0,0],Wrong_X_train[Wrong_y_train == 0,1], 
                 Wrong_X_train[Wrong_y_train == 0,2], color = "blue",label = "FP")
    
    ax.scatter3D(Correct_X_train[Correct_y_train == 1,0],Correct_X_train[Correct_y_train == 1,1],
                 Correct_X_train[Correct_y_train == 1,2], color = "red",label = "TP")
    ax.scatter3D(Correct_X_train[Correct_y_train == 0,0],Correct_X_train[Correct_y_train == 0,1],
                 Correct_X_train[Correct_y_train == 0,2], color = "green",label = "TN")
    
    ax.legend()
    plt.title("Random forest final prediction train ")
    plt.show()