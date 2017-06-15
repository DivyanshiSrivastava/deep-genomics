from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD, Adagrad, Adam
from keras.callbacks import EarlyStopping
from keras.models import model_from_yaml
from keras import backend as K
from sklearn.model_selection import train_test_split
import read
import numpy as np
import sklearn.metrics
import sys

def keras_model():

    model = Sequential()
    
    #model.add(Reshape((n, 4), input_shape=(n*4,)))
    
    model.add(Conv1D(16, 20, padding="same", input_shape=(n, 4,)))
    print model.input_shape
    print model.output_shape
    model.add(Activation('relu'))
    model.add(Conv1D(32,20,padding = "same"))
    model.add(Conv1D(64,20,padding = "same"))
    model.add(Flatten())
    print model.output_shape
    model.add(Dense(512))
    # Flattened 
    print model.output_shape
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # Finally
    print model.output_shape
    model.add(Activation('sigmoid'))
    return model

def train_and_test(X_train, y_train, X_test, y_test, n):

    model = keras_model()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    hist = model.fit(X_train, y_train, validation_split=0.1, callbacks=[early_stopping], batch_size= 200, nb_epoch= 10)
    probas = model.predict_on_batch(X_test_rs)
    return model, probas

# read data
n = 200
X,y = read.get_matrix(sys.argv[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train_rs = np.reshape(X_train, (-1,200,4))
X_test_rs = np.reshape(X_test, (-1,200,4))

# train
model, probas = train_and_test(X_train_rs, y_train, X_test, y_test, n)

# measure performance
roc_auc = sklearn.metrics.roc_auc_score(y_test, probas)
prc = sklearn.metrics.average_precision_score(y_test, probas)
print "test roc", roc_auc
print "test prc", prc

# saving model
model.save("conv3L_model.h5py")
