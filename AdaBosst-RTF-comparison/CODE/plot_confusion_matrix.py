
"""
This file containes a helper function to plot confusion matrices

Created on: 06 April 2021
Authors:    Jonas Perolini & Raphael Uebersax
Libraries:  Sklearn, Pandas, Numpy, matplotlib
Inspired from:
https://medium.com/analytics-vidhya/generation-of-a-concatenated-confusion-matrix-in-cross-validation-912485c4a972 
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(cnf_matrix, classes, title, TP_matrix):

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title, fontsize=16)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, "{:d} \n\n{:.2f}".format(TP_matrix[i,j],cnf_matrix[i, j]), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black", fontsize=13)
        # plt.text(j, i, format(TP_matrix[i,j], 'd'), 
        #          color="white" if cnf_matrix[i, j] > thresh else "black", 
        #          verticalalignment = 'bottom')

    #plt.tight_layout()
    plt.ylabel('True label', fontsize=13)
    plt.xlabel('Predicted label', fontsize=13)


    name = "Figures/" + title + ".pdf"
    plt.savefig(name)

        
    
    return cnf_matrix

def plot_confusion_matrix(predicted_labels_list, y_test_list, classes, classifier, Ffolds):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list, normalize = "true" )
    np.set_printoptions(precision=2)
    TP_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    TP_matrix = np.round(TP_matrix/Ffolds).astype(int)

    
    title = classifier

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes, title, TP_matrix)
    plt.show()
