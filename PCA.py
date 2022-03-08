"""
This code runs a PCA, and plots the results based on the classifications
defined in a Target Column

NOTE: The strings in this code ARE CASE SENSITIVE - make sure you
      match the case for any column names, file names, etc.


Author: Tom Burns
email : tom.burns@canada.ca
date  : Feb 24th 2021

VERSION 1.1.0
"""

VERSION = '%i.%i.%i' % (1, 1, 0)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# This is the input file that contains your
# descriptors and your target values
#INFILE = 'test.csv'
INFILE = 'BinaryMLFile.csv'
#INFILE = 'TrinaryMLFile.csv'

# This is the heading of the "target" column
# that should contain your classifications
#TARGET = 'Target'
TARGET = 'Class'

# If there are any columns you want the code
# to ignore in the csv file, add them to OMIT
# OMIT = ['Example1', 'Example2']
OMIT = ['CAS']

# If you want to the data to be scaled, set this
# value to True (Recommended),
# otherwise set this value to False
SCALE = True


# ====================================================================
# DO NOT EDIT following lines
LOC = '\\'.join(os.path.realpath(__file__).split('\\')[:-1]) + '\\'
INFILE = LOC + INFILE
# ====================================================================


def import_data(infile=INFILE):
    """Imports and formats the data"""
    # Import the raw data from the file
    raw = pd.read_csv(infile)

    # Filter out the columns to grab only the desired descriptors
    rcols, cols = [c for c in raw], []
    for col in rcols:
        if col not in OMIT and col != TARGET:
            cols.append(col)

    # Format the final arrays
    X, Y = np.array(raw[cols]), np.array(raw[TARGET])
    return X, Y, cols


def plot_results(X, Y, Z, C):
    """Generates the PCA Plots"""
    
    # This defines the colours of each "classification", if you
    # want to add more than 4 classifications, you'll need to add
    # some colours to this list
    clrs = ['b', 'r', 'g', 'y']
    
    # Finds all of the unique classifications
    u = np.unique(Y)
    # Code will error out if length of u > length of clrs
    if len(u) > len(clrs):
        print("Error: Number of unique classifications > Number of defined colours in code")
        exit()
    
    # separate the set into the different classes
    allX, allY = [], []
    for y in u:
        allX.append(X[np.where(Y==y)])
        allY.append(Y[np.where(Y==y)])

    # Plot the final data PCA1 vs PCA2
    plt.subplot(221)
    for i, x in enumerate(allX):
        one, two = x[:,0], x[:,1]
        plt.scatter(one, two, color='none', edgecolor=clrs[i], label='Class %i' % u[i])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()

    # Plot the contribution of each PC to the variance
    plt.subplot(222)
    expl, csum = Z[0], Z[1]
    if len(expl) > 15:
        expl_X = [i+1 for i in range(0, len(expl[:15]))]
        expl_Y = expl[:15]
    else:
        expl_X = [i+1 for i in range(0, len(expl))]
        expl_Y = expl[:]

    plt.title('PC1 + PC2 = %.2f' % csum + '% of the Variance')
    plt.bar(expl_X, 100* expl_Y, edgecolor='b', lw=2, fc=(0, 0, 1, 0.3))
    plt.xlabel('PC')
    plt.ylabel('Contribution to Variance [%]')

    # Descriptor contribution to PC 1
    loadings = Z[2]
    plt.subplot(223)
    first = loadings[0, :]
    first = np.array([abs(i) for i in first])
    dic = pd.DataFrame(first, dtype='float64')
    dic = dic.sort_values([0], ascending=False)
    f_ids = []
    for i in dic.index:
        f_ids.append(i)
    if len(dic) > 15:
        f_y = np.array(dic[0][:15])
        f_x = np.array([i for i in range(len(f_y))][:15])
    else:
        f_y = np.array(dic[0][:])
        f_x = np.array([i for i in range(len(f_y))][:])
    plt.bar(f_x, 100 * f_y, edgecolor='b', lw=2, fc=(0, 0, 1, 0.3))
    plt.ylabel('PC1 Contribution [%]')
    f_labels = [C[i] for i in f_ids]
    plt.xlabel('Variable')
    if len(f_x) < len(f_labels):
        f_labels = f_labels[:len(f_x)]
    plt.xticks(f_x, f_labels, rotation=90)

    # Descriptor Contribution to PC2
    plt.subplot(224)
    first = loadings[1, :]
    first = np.array([abs(i) for i in first])
    dic = pd.DataFrame(first, dtype='float64')
    dic = dic.sort_values([0], ascending=False)
    f_ids = []
    for i in dic.index:
        f_ids.append(i)
    if len(dic) > 15:
        f_y = np.array(dic[0][:15])
        f_x = np.array([i for i in range(len(f_y))][:15])
    else:
        f_y = np.array(dic[0][:])
        f_x = np.array([i for i in range(len(f_y))][:])
    plt.bar(f_x, 100 * f_y, edgecolor='b', lw=2, fc=(0, 0, 1, 0.3))
    plt.ylabel('PC2 Contribution [%]')
    f_labels = [C[i] for i in f_ids]
    plt.xlabel('Variable')
    if len(f_x) < len(f_labels):
        f_labels = f_labels[:len(f_x)]
    plt.xticks(f_x, f_labels, rotation=90)
    #print(sum(100 * f_y))
    #print(f_y)
    plt.show()


def run_pca(X):
    """Runs the PCA fitting"""
    # Loading in the PCA module
    #print(X.shape[1])
    #exit()
    #pca = PCA(n_components=X.shape[1])
    pca = PCA()
    
    # Here is where the descriptor scaling takes place
    if SCALE:
        scaler = StandardScaler()
        sX = scaler.fit_transform(X)
    else:
        sX = X[:]
    
    # Run the PCA
    pcaX = pca.fit_transform(sX)

    # Post Analysis on PCA results
    loadings = pca.components_
    explained = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    components = np.transpose(pca.components_[0:2, :])
    c_sum = sum(explained_ratio[:2]) * 100

    return pcaX, (explained_ratio, c_sum, loadings)


def welcome():
    """Adds a welcome message"""
    print('=' * 75)
    print(' Starting PCA.py Version %s' % VERSION)
    print('-' * 75)
    print(' Code Parameters:\n')
    print(' Input File       :', INFILE.split('\\')[-1])
    print(' Target Column    :', TARGET)
    print(' Omitting Columns :', OMIT)
    print(' Scale Descriptors:', SCALE)
    print('=' * 75)


def main():
    """Main Execution of Script"""
    welcome()
    X, Y, cols = import_data()
    pcaX, pca_info = run_pca(X)
    plot_results(pcaX, Y, pca_info, cols)


if __name__ in '__main__':
    main()
    print('\n Code Terminated Normally.\n')
    input(' Press enter to close window.\n')
