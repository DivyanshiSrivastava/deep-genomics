from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping
import read
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from keras.models import model_from_yaml
from keras import backend as K
import sys

def keras_graphmodel():
    
    seq_input = Input(shape = (n,4,), name = 'seq')
    xs = Conv1D(16, 20, padding="same")(seq_input)
    xs = Conv1D(32,20,padding = "same")(xs)
    xs = Conv1D(64,20,padding = "same")(xs)
    xs = Activation('relu')(xs)
    xs = Flatten()(xs)
    xs = Dense(512, activation = 'relu')(xs)
    
    acc_input = Input(shape = (3,), name = 'accessibility')
    xa = Dense(1,activation = 'sigmoid')(acc_input)
    merge = concatenate([xs,xa])
    result = Dense(256, activation = 'relu')(merge)
    result = Dropout(0.5)(result)
    result = Dense(1, activation = 'sigmoid')(result)
    model = Model(inputs=[seq_input, acc_input],outputs = result)
    return model

def train_and_test(X_train, y_train, X_test, y_test, acc_train, acc_test, n):

    model = keras_graphmodel()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    hist = model.fit([X_train,acc_train], y_train, validation_split=0.1, callbacks=[early_stopping], batch_size= 200, nb_epoch= 20)
    probas = model.predict_on_batch([X_test,acc_test])
    return model, probas

n = 200
X_train,y_train = read.get_matrix( sys.argv[1] + ".seq.train.txt")
X_test, y_test = read.get_matrix( sys.argv[1] + ".seq.test.txt")
acc_train = np.loadtxt( sys.argv[1] + ".acc.train.txt")
acc_test = np.loadtxt( sys.argv[1] + ".acc.test.txt")
X_train_rs = np.reshape(X_train, (-1,200,4))
X_test_rs = np.reshape(X_test, (-1,200,4))

model, probas = train_and_test(X_train_rs, y_train, X_test_rs, y_test, acc_train, acc_test, n)

# measure performance
roc_auc = sklearn.metrics.roc_auc_score(y_test, probas)
prc = sklearn.metrics.average_precision_score(y_test, probas)
print "test roc", roc_auc
print "test prc", prc

# saving model
model.save("conv3L_model.h5py")
