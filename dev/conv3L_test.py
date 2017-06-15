from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping
from keras.models import model_from_yaml
from keras.models import load_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from itertools import islice
from itertools import chain
import numpy as np
import sklearn.metrics
import sys

def test(X_test):
    probas = model.predict_on_batch(X_test)
    return probas

def map_bases(line):
    features = []
    for bp in line.strip():
        features.append(feature_dict[bp])
    return list(chain.from_iterable(features))

# load model 

model = load_model("conv3L_model")
feature_dict = { 'A' : [1, 0, 0, 0], 'T' : [0,1,0,0], 'G' : [0,0,1,0],'C' : [0,0,0, 1], 'N': [0,0,0,0]}
n = 1000

# iterate over test
probas = []
lc = 0
with open("/group/lab/divyanshi/data/EB/factors/Ascl1/train_test_split/seqfiles/test.dat", "r") as f:
    while True:
        chunk = list(islice(f, n))
        if chunk != []:
            lc += 1
            chunk = [map_bases(x) for x in next_n_lines]
            x = np.array(chunk)
            x_rs = np.reshape(x,(-1,200,4))
            print x_rs.shape
            probas.append(test(x_rs))
	if not chunk:
            break

probas = np.ravel(np.array(probas))
y_test = np.loadtxt("/group/lab/divyanshi/data/EB/factors/Ascl1/train_test_split/seqfiles/test.labels")

# measure performance
roc_auc = sklearn.metrics.roc_auc_score(y_test, probas)
prc = sklearn.metrics.average_precision_score(y_test, probas)
print "test roc", roc_auc
print "test prc", prc
