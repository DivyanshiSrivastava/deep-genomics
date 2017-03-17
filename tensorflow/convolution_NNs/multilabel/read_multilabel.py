import sys
import numpy as np
from itertools import chain
from random import random
import sklearn.metrics

feature_dict = { 'A' : [1, 0, 0, 0], 'T' : [0,1,0,0], 'G' : [0,0,1,0],'C' : [0,0,0, 1], 'N': [0,0,0,0]}

def get_data_from_file(infile):
    """ get sequence data """ 
    features = list()
    labels = list()
    with open(infile) as data:
        for line in data: 
            for bp in line.strip():
                features.append(feature_dict[bp])
    return list(chain.from_iterable(features)) 

def get_matrix(infile):
    features = get_data_from_file(infile)
    n = int(sys.argv[3])
    m = len(features)/ (n * 4)
    features_onehot = np.reshape(features, (m, n*4))
    print features_onehot.shape
    return features_onehot

def get_labels(infile):
    return np.genfromtxt(infile, delimiter = ",")

# functions to measure metrics:

def metrics(probas, labels, num_classes, fp):
    PRC = list()
    for i in range(num_classes):
        fp.write("Label %d:%f\t" % (i,sklearn.metrics.average_precision_score(labels[:,i], probas[:,i])))
    fp.write("\n")
