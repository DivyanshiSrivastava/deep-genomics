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

model = load_model(sys.argv[2])
feature_dict = { 'A' : [1, 0, 0, 0], 'T' : [0,1,0,0], 'G' : [0,0,1,0],'C' : [0,0,0, 1], 'N': [0,0,0,0]}
n = 5000

# iterate over test
lc = 0
with open(sys.argv[1] + ".seq", "r") as f:
    while True:
        chunk = list(islice(f, n))
        if chunk != []:
            lc += 1
            print lc
            chunk = [map_bases(x) for x in chunk]
            x = np.array(chunk)
            x_rs = np.reshape(x,(-1,200,4))
            print n*(lc-1),n*lc
            try:
                hold = np.ravel(np.array(test(x_rs)))
                probas = np.hstack((probas, hold))
            except:
                probas = np.ravel(np.array(test(x_rs)))
	if not chunk:
            break

np.savetxt(sys.argv[1] + "probas.txt", probas)
y_testn = np.loadtxt((sys.argv[1]+ ".labels"))
y_testb = np.loadtxt((sys.argv[1] + ".broadlabels"))
threshold = lambda t: 1 if t >= 0.5 else 0 
npthresh = np.vectorize(threshold)
y_pred = npthresh(probas)

# measure performance
roc_auc_n = sklearn.metrics.roc_auc_score(y_testn, probas)
prc_n = sklearn.metrics.average_precision_score(y_testn, probas)
print "test roc", roc_auc_n
print "test prc", prc_n
print sklearn.metrics.confusion_matrix(y_testn, y_pred)

# measure performance
roc_auc_b = sklearn.metrics.roc_auc_score(y_testb, probas)
prc_b = sklearn.metrics.average_precision_score(y_testb, probas)
print "test roc", roc_auc_b
print "test prc", prc_b
print sklearn.metrics.confusion_matrix(y_testb, y_pred)

