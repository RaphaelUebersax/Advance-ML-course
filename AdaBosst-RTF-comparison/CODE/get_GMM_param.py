"""
This file containes code in order to find optimal parameters for a GMM fitting 
the minority class of the Shill Bidding dataset.

Created on: 06 April 2021
Authors:    Jonas Perolini & Raphael Uebersax
Libraries:  Sklearn, Pandas, Numpy, matplotlib
"""

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

my_path = os.path.dirname(__file__)

data = pd.read_csv(my_path + "/Shill_Bidding_Dataset.csv", delimiter = ",")
data = data.dropna(axis = 1)

DO_PCA = False

label = 1

X = data.values[data.values[:,12] == label,3:12]  # Select the nine attributes
y = data.values[data.values[:,12] == label,12]    # Select the class (target)

# Change cathegorical values to be integers 
y = y.astype(int)       # Change class from object to int
X = X.astype(float)     # Change attributes from object to float (for np.nan)

X = X/np.std(X,axis=0, ddof=1)





#replace missing values by the mean 
median = False

if median:
    missing_id = X == 0
    X[missing_id] = np.nan
    mediandata = np.nanmedian(X,axis = 0)
    mediandata = np.tile(mediandata, (X.shape[0],1))
    X[missing_id] = mediandata[missing_id] 


folds = 10  #number of fold repetitions
it = 5    #number of GMM components

bicval = np.zeros((folds,3,it))
aicval = np.zeros((folds,3,it))

types = ['spherical', 'diag','full'] #types of covariance matrices

for t in range(folds):

    #split the data set into training and testing sets, only the training set will be used. Spliting the set helps avoiding overfitting
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = None,shuffle = True)

    if DO_PCA:
        pca = PCA(n_components=3)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print(pca.explained_variance_ratio_)

    #only keep the points with the label we are interested in (i.e. Malign or Benign)
    #X_train_label = X_train[ np.where(y_train == label)[0],: ]

    #find the optimal paramters of the GMM using bic and aic curves 
    i = 0
    for cov in types: 

        for j in range(it):

            #fit the GMM on the set with only one label 
            gm = GaussianMixture(n_components=j+1, covariance_type=cov, init_params='kmeans', random_state=None).fit(X_train)

            #store the aic and bic values
            aicval[t,i,j] = gm.aic(X_train)
            bicval[t,i,j] = gm.bic(X_train)

        i = i + 1
    print('fold: ',t+1 )

#keep the best run over the ten folds 
aic_min = np.min(aicval,axis = 0)
bic_min = np.min(bicval,axis = 0)


#test the upsampling 
nb_samples = 200

if DO_PCA:
    pca = PCA(n_components=3)
    pca.fit(X)
    X_train = pca.transform(X)
    print(pca.explained_variance_ratio_)
else: 
    X_train = X

BIC = []
GM = []

for gm_folds in range(10):
    gm = GaussianMixture(n_components=3, covariance_type='full',
                            init_params='kmeans',
                            random_state=None).fit(X_train)
    BIC.append(gm.bic(X_train))
    GM.append(gm)

idx = np.argmin(np.array(BIC))
gm = GM[idx]
new_points, _ = gm.sample(n_samples= nb_samples)

X_train = np.vstack((X_train, new_points))

plt.figure()

ax = plt.axes(projection ="3d")

X_class_zero_x = X_train[:,0]
X_class_zero_y = X_train[:,1]
X_class_zero_z = X_train[:,2]

ax.scatter3D(X_class_zero_x[0:nb_samples], X_class_zero_y[0:nb_samples], X_class_zero_z[0:nb_samples], color = "red")
ax.scatter3D(X_class_zero_x[nb_samples:-1], X_class_zero_y[nb_samples:-1], X_class_zero_z[nb_samples:-1], color = "green")

plt.title("Upsampling")
plt.show()
                

##plot the curves to find the elbow of the curve 

#Spherical covariance matrix 
plt.figure()
plt.title('best aic and bic over ten folds for spherical cov matrix')
plt.plot(range(1,it+1),aic_min[0,:],label = 'aic curve')
plt.plot(range(1,it+1),bic_min[0,:],label = 'bic curve')
plt.legend()
plt.xlabel('number of Gaussians')
plt.ylabel('metric value')
plt.show()

#Diag covariance matrix 
plt.figure()
plt.title('best aic and bic over ten folds for diagonal cov matrix')
plt.plot(range(1,it+1),aic_min[1,:],label = 'aic curve')
plt.plot(range(1,it+1),bic_min[1,:],label = 'bic curve')
plt.legend()
plt.xlabel('number of Gaussians')
plt.ylabel('metric value')
plt.show()

#Full covariance matrix 
plt.figure()
plt.title('best aic and bic over ten folds for full cov matrix')
plt.plot(range(1,it+1),aic_min[2,:],label = 'aic curve')
plt.plot(range(1,it+1),bic_min[2,:],label = 'bic curve')
plt.legend()
plt.xlabel('number of Gaussians')
plt.ylabel('metric value')
plt.show()