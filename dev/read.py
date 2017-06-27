import numpy as np
from itertools import chain
from random import random
import sklearn.metrics

feature_dict = { 'A' : [1, 0, 0, 0], 'T' : [0,1,0,0], 'G' : [0,0,1,0],'C' : [0,0,0, 1], 'N': [0,0,0,0]}
label_dict = {0 : [1,0], 1 : [0,1]}

def get_data_from_file(infile):
    """ get sequence data """ 
    features = list()
    labels = list()
    with open(infile) as data:
        for line in data: 
            sample = list(line.split()[0])
            for bp in sample:
                features.append(feature_dict[bp])
            label = line.strip().split()[1]
            labels.append(label_dict[int(label)])
    return list(chain.from_iterable(features)), list(chain.from_iterable(labels))  

def get_matrix(infile):
    features, labels = get_data_from_file(infile)
    m = len(labels)/2
    n = 200
    features_onehot = np.reshape(features, (m, n*4))
    labels_onehot =  np.reshape(labels, (m, 2))
    return features_onehot, labels_onehot[:,1]

# functions to measure metrics:

def metrics(probas, labels):
    PRC = list()
    return sklearn.metrics.average_precision_score(labels[:,1], probas[:,1])
