"""
This file containes code in order to analyse two classification method: Adaboost
and Random Tree Forest. In this file, both algorithms are tested on the moon
dataset provided by SkLearn with multiple different scenarios.

Created on: 06 April 2021
Authors:    Jonas Perolini & Raphael Uebersax
Libraries:  Sklearn, Pandas, Numpy, matplotlib
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_moons, make_circles, make_blobs

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer

my_path = os.path.dirname(__file__) # Get the directory of this file
np.random.seed(15)

Spider = True

############### PARAMETERS TO SET DEPENDING ON SCENARIO #######################

### Set train/test split ratio ###########################
test_ratio = 0.3

### Step size in the mesh
h = .01

### Number estimator [ada, RTF]
nb_estimators = [50,100]

### Name of fig
save = True
my_name = "Toy_benchmark"

### How to deal with missing values ######################
replace_by_median = False       # 1) Median input for missing values
replace_using_prediction = False  # 2) Use predictor for missing value
remove_top_gray = False

### Number outliers (None if 0)
outlier = False
if outlier: 
    np.random.seed(90000)
nb_outliers = 3

### Delete
delete_top = False
delete_side = False

### For unbalanced dataset
delete_unbalancing = False
unbalanced_ratio = 0.94

### Show effect of training on few samples 
Few_samples = False

if Few_samples:    
    test_ratio = 0.95
    print('Train/test ratio switched to ', test_ratio)

#title of figure and title of plot 
Names = ["Ada Testing", "RTF Testing"]
fig_name = my_path +"/"+ "Figures_tight/" + "Outlier_3" + ".pdf"




########################### CREATE TOY DATASET ################################
#few samples: use noise = 0.07 and test_ratio of 0.9 

X, y = make_moons(n_samples=700, noise=0.07, random_state=0)

# Normalize the data           
X = StandardScaler().fit_transform(X) 

# preprocess dataset, split into training and test part
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_ratio, random_state=42)

#artificially unbalance dataset 
if delete_unbalancing:
    print('Imbalancing the dataset with an imbalancing ratio of ', unbalanced_ratio)
    X_grey = X_train[y_train==1,:]
    y_grey = y_train[y_train==1]
    nb = np.round(unbalanced_ratio*X_grey.shape[0]).astype(int)
    idx = np.random.permutation(X_grey.shape[0])[:nb]
    X_grey = np.delete(X_grey,idx,0)
    y_grey = np.delete(y_grey,idx,0)
    X_train = np.vstack((X_train[y_train==0,:], X_grey))
    y_train = np.hstack((y_train[y_train==0], y_grey))

# Create outliers
elif outlier:
    print('generating ', nb_outliers,' outliers')
    idx = np.random.permutation(X_train.shape[0])[:nb_outliers]
    y_train[idx] = 1-y_train[idx] 

# Delete
elif delete_side:
    print('deleting points in training set')
    a = np.int32(X_train[:,1] > -1.17)
    b = np.int32(X_train[:,1] < -0.5)
    c = a+b
    idx = np.where(c == 2)
    print(idx[0].shape[0]," points removed")
    X_train = np.delete(X_train,idx,0)
    y_train = np.delete(y_train,idx,0)

elif delete_top:
    print('deleting points in training set')
    idx = np.where(X_train[:,1]<-1.20)
    print(idx[0].shape[0]," points removed")
    X_train = np.delete(X_train,idx,0)
    y_train = np.delete(y_train,idx,0)

else:
    print('Un modified dataset')
    X_train = X_train
    y_train = y_train

# Define classifiers
classifiers = [AdaBoostClassifier(n_estimators = nb_estimators[0]),
               RandomForestClassifier(n_estimators = nb_estimators[0])]


##################### ADD MISSING VALUES #############################
if (replace_by_median or replace_using_prediction):
    if remove_top_gray:
        idx = np.where(X_train[:,1]<-1.25)
        X_train[idx,1]= np.nan
    else:
        idx = np.random.binomial(n=1, p=0.1, size=X_train.shape)
        idx = idx>0
        X_train[idx] = np.nan

##################### DEALING WITH MISSING VALUES #############################
if (replace_using_prediction and replace_by_median):
    raise Exception('Select only one method for replacing missing values.')

# Replace missing value with median of attribute (SimpleInputter)
elif replace_by_median:
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_mean.fit(X_train)
    X_train = imp_mean.transform(X_train)

# Replace missing value with predictor (choose estimator model in begining of file)
elif replace_using_prediction:
    imp_mean = IterativeImputer(random_state=0, estimator = KNeighborsRegressor())
    data = np.c_[X_train, y_train]
    imp_mean.fit(data)
    data = imp_mean.transform(data)
    X_train = data[:,:-1]
    
else:
    print('No missing values')


####################### PLOTTING ##############################################

fig, ax = plt.subplots(1,3, sharey=True, figsize=(15,6))

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Define colormap
cm = plt.cm.RdGy
cm_bright = ListedColormap(['#FF0000', '#606060'])

# Plot the training points
ax[0].set_title("Training",fontsize=35)
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')

ax[0].set_xlim(xx.min(), xx.max())
ax[0].set_ylim(yy.min(), yy.max())
ax[0].set_xticks(())
ax[0].set_yticks(())

for i, classifier in enumerate(classifiers):

    #train and predict 
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_train)
    #F1 = f1_score(y_train, y_pred)
    score = classifier.score(X_test, y_test)
  

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]     

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax[i+1].set_title(Names[i],fontsize=35)
    ax[i+1].contourf(xx, yy, Z, cmap=cm, alpha=.8)


    #plot the training points 
    ax[i+1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
            edgecolors='k',alpha = 0)

    # Plot the testing points
    ax[i+1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
            edgecolors='k', alpha=1)

    ax[i+1].set_xlim(xx.min(), xx.max())
    ax[i+1].set_ylim(yy.min(), yy.max())
    ax[i+1].set_xticks(())
    ax[i+1].set_yticks(())
    ax[i+1].text(xx.min() + .3, yy.min() + .2, ('%.2f' % score).lstrip('0'),
                size=12, horizontalalignment='left',fontsize = 30)
    

fig2 = plt.gcf()
plt.show()
if save:   
    fig2.tight_layout()
    fig2.savefig(fig_name)
    if Spider: 
        name = "Figures/" + my_name + ".pdf"
        plt.savefig(name)
    










       
            

        

