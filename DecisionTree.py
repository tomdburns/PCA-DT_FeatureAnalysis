"""
Runs a Decision Tree Classifier on the data set returns
analysis of first two nodes in the tree or array of trees.

Goal: to inform the user of the usefulness of each individual
      descriptor during the classification

This code will run several decision tree fittings with a random
set of the descriptors, and return the distributions of those
descriptors in final array of trees. The number of trees run
is determined by the NTREES variable. If only one tree is selected
the code will use all available descriptors in the fitting

NOTE: if the PCA fails to distinguish between any of the descriptors
      the results of these decision trees are not always useful.


Author: Tom Burns
email : tom.burns@canada.ca
date  : Feb 24th 2021

VERSION 1.1.0
"""

VERSION = '%i.%i.%i' % (1, 1, 0)


import os
import sys
import numpy as np
from math import floor
from PCA import import_data
import sklearn.tree as TREE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DT


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
#OMIT = ['Example1', 'Example2']
OMIT = ['CAS']

# If you want to the data to be scaled, set this
# value to True (Recommended),
# otherwise set this value to False
SCALE = True

# How many trees will you run?
NTREES = 10000

# Run the boostrapping? (Default=False)
BOOT   = False
# If boostrap being run, how many steps
BSTEPS = 10000

# ====================================================================
# DO NOT EDIT following lines
LOC = '\\'.join(os.path.realpath(__file__).split('\\')[:-1]) + '\\'
INFILE = LOC + INFILE
# ====================================================================


def tree_to_code(tree, feature_names):
    """Converts the tree to working code, uses that code to extract the node identities"""
    _tree = TREE._tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print("def tree({}):".format(", ".join(feature_names)))
    node_dist = {}

    def recurse(node, depth, node_dist):
        #print(depth)
        indent = "  " * depth
        Depth = 'Depth %i' % depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            if Depth not in node_dist:
                node_dist[Depth] = []
            if name not in node_dist[Depth]:
                node_dist[Depth].append(name)
            #print("{}if {} <= {}:".format(indent, name, threshold))
            node_dist = recurse(tree_.children_left[node], depth + 1, node_dist)
            #print("{}else:  # if {} > {}".format(indent, name, threshold))
            node_dist = recurse(tree_.children_right[node], depth + 1, node_dist)
        else:
            #print("{}return {}".format(indent, tree_.value[node]))
            pass
        return node_dist

    node_dist = recurse(0, 1, node_dist)
    return node_dist


def fit_dt(x, y, c, feats):
    """Runs the decision tree fitting"""
    tree = DT(max_depth=2, max_features=feats)
    if SCALE:
        scaler = StandardScaler()
        sx = scaler.fit_transform(x)
    else:
        sx = x[:]
    tree.fit(x, y)
    pred = tree.predict(x)
    ba  = accuracy(pred, np.array(y))
    nodes = tree_to_code(tree, c)
    return nodes, ba


def bootstrap(bset, N=BSTEPS):
    """Runs a boostrapping of the values in bset with N steps"""
    vals, b = [], len(bset)
    print('\n Running Boostrap:')
    for n in range(N):
        sub = []
        for m in range(b):
            sub.append(np.random.choice(bset))
        vals.append(np.mean(sub))
        printProgress(n+1, N)
    print('\nDone.')
    u, s = np.mean(vals), np.std(vals)
    l, h = u - 1.96*s, u + 1.96*s # Ranges for 95% CI
    print('\n' + '~'*75 + '\n Boostrap Results:\n')
    print(' Mean                    = %.2f' % u + ' %')
    print(' Standard Error          = %.2f' % s + ' %')
    print(' 95% Confidence Interval =' + ' %.2f - %.2f' % (l, h) + ' %')
    print('~'*75)
    return u, s, l, h


def accuracy(pred, actu):
    """Calculates the BA"""
    TC, FC, AC = [], [], [] # True Calls, False Calls, All Calls
    UQ = np.unique(actu)
    for i in UQ:
        TC.append(0)
        FC.append(0)
        AC.append(0)
    for i, a in enumerate(actu):
        j = np.where(UQ == a)[0][0]
        p = pred[i]
        if p == a:
            TC[j] += 1
        else:
            k = np.where(UQ == p)[0][0]
            FC[j] += 1
        AC[j] += 1
    BA = []
    for i, v in enumerate(AC):
        if v == 0:
            continue # only a valid skip here because I'm looking at full set.
        BA.append(100 * TC[i] / v)
    return np.mean(BA)


def run_trees(x, y, c):
    """controls the running and analysis of the dt fittings"""
    # These two objects will contain a running tally of the identiies
    # of the first two nodes in the trees
    node_one, node_two = {}, {}

    # How many features will each decision tree consider?
    # If only one tree is being run, it will consider all
    # features, otherwise it will consider N=sqrt(all features),
    # selecting at random.
    if NTREES == 1:
        feats = x.shape[1]
    else:
        feats = floor(x.shape[1]**0.5)

    # Run the actual trees, sort resulting nodes
    print(' Starting Decision Tree Fittings:')
    accs = []
    for run in range(NTREES):
        nodes, ba = fit_dt(x, y, c, feats)
        accs.append(ba)
        for node in nodes['Depth 1']:
            if node not in node_one:
                node_one[node] = 0
            node_one[node] += 1
        for node in nodes['Depth 2']:
            if node not in node_two:
                node_two[node] = 0
            node_two[node] += 1
        printProgress(run+1, NTREES)
    print('\n Done.')
    if BOOT:
        uBA, sBA, lBA, hBA = bootstrap(accs)
    else:
        uBA, sBA, lBA, hBA = None, None, None, None
        print('\n' + '~'*75 + '\n Balanced Accuracy Distribution Statistics:\n')
        print(' Mean               = %.2f' % np.mean(accs) + ' %')
        print(' Standard Deviation = %.2f' % np.std(accs) + ' %')
        print('~'*75)
    return node_one, node_two, (accs, uBA, sBA, lBA, hBA)


def sort_data(iset):
    """Sorts the node counts into usable arrays for plotting"""
    rX, rY = [], []
    for i in iset:
        rX.append(i)
        rY.append(iset[i])
    ri = np.argsort(np.array(rY))
    ri = np.flip(ri)
    X, Y = [], []
    for i in ri:
        X.append(rX[i])
        Y.append(rY[i])
    return np.array(X), np.array(Y)
    

def printProgress (iteration, total, prefix='Progress', suffix='Complete ',
                   decimals=2, barLength=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write(' %s [%s] %.2f%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()


def plot_results(one, two, accs):
    """Plots the distributions from the first two nodes in the DTs"""
    # Format the data from first two nodes
    x_one, y_one = sort_data(one)
    x_two, y_two = sort_data(two)
    
    # The maximum number of values in the x axis of plots 121 and 122
    MAX_X = 20

    # Plot node 1 distribution
    plt.subplot(131)
    plt.bar(x_one[:MAX_X], 100 * y_one[:MAX_X] / NTREES, lw=2, fc=(0, 0, 1, 0.3), edgecolor='b')
    plt.title('First Node Distribution')
    plt.xlabel('Descriptor')
    plt.ylabel('Frequency [%]')
    plt.xticks(x_one[:MAX_X], x_one[:MAX_X], rotation=90)

    # Plot node 2 distribution
    plt.subplot(132)
    plt.bar(x_two[:MAX_X], 100 * y_two[:MAX_X] / NTREES, lw=2, fc=(0, 0, 1, 0.3), edgecolor='b')
    plt.title('Second Node Distribution')
    plt.xlabel('Descriptor')
    plt.ylabel('Frequency [%]')
    plt.xticks(x_two[:MAX_X], x_two[:MAX_X], rotation=90)
    
    # Plot accuracy distribution
    plt.subplot(133)
    plt.hist(accs[0], lw=2, fc=(0,0,1,0.3), edgecolor='b')
    if BOOT:
        plt.axvline(accs[1], color='k', linestyle='-', label='Bootstrap Mean')
        plt.axvline(accs[3], color='k', linestyle=':', label='95% CI')
        plt.axvline(accs[4], color='k', linestyle=':')

    else:
        plt.axvline(np.mean(accs[0]), color='k', linestyle='-', label='Mean')
        plt.axvline(np.mean(accs[0]) - np.std(accs[0]), color='k',
                    linestyle=':', label='Standard Deviation')
        plt.axvline(np.mean(accs[0]) + np.std(accs[0]), color='k', linestyle=':')
    plt.legend()   
    plt.ylabel('Frequency')
    plt.xlabel('Balanced Accuracy [%]')
    plt.title('Balanced Accuracy Distribution')

    plt.show()


def welcome():
    """Adds a welcome message"""
    print('=' * 75)
    print(' Starting DecisionTree.py Version %s' % VERSION)
    print('-' * 75)
    print(' Code Parameters:\n')
    print(' Input File       :', INFILE.split('\\')[-1])
    print(' Target Column    :', TARGET)
    print(' Omitting Columns :', OMIT)
    print(' Scale Descriptors:', SCALE)
    print(' Number of Trees  :', NTREES)
    print(' Run Bootstrap    :', BOOT)
    if BOOT:
        print(' Bootstrap Steps  :', BSTEPS)
    if BOOT and NTREES > 1000:
        print('\n WARNING: You are running a bootstrap on %i Trees,' % NTREES)
        print('          this will be a long calculation and is')
        print('          unlikely to provide useful insights.')
    elif BOOT and NTREES < 8:
        print('\n WARNING: You are running a bootstrap on %i Trees,' % NTREES)
        print('          a minimum of 8 Trees will be required to')
        print('          provide useful insights.')
    print('=' * 75)


def main():
    """Main Execution of Script"""
    welcome()
    X, Y, cols = import_data(infile=INFILE)
    node_one, node_two, accs = run_trees(X, Y, cols)
    plot_results(node_one, node_two, accs)


if __name__ in '__main__':
    main()
    print('\n Code Terminated Normally.\n')
    input(' Press enter to close window.\n')
