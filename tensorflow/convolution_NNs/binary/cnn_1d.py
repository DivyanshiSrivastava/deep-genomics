# Train a single layer neural network ( CNN ) using sequence.
# Impementation: tensorflow v. 0.11

import sys
import tensorflow as tf
import numpy as np
import read
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

def variables(shapeW, shapeB, nameW, nameB):
    """ Defining the initial convolution """
    initialW = tf.truncated_normal( shapeW , stddev=0.1)
    initialB = tf.constant(0.1, shape=shapeB)
    return tf.Variable(initialW, name = nameW), tf.Variable(initialB, name = nameB)

def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, 1, n, 1],
                        strides=[1, 1, n, 1], padding='SAME')

def nn(xd,yd,zd,bd,fc_nodes,x,n,keep_prob):
    """ convolve over feature matrix """

    W_conv1, b_conv1 = variables([20,4,32], [32], 'W_conv1', 'b_conv1')
    x_image = tf.reshape(x, [-1,n,4])                          # [batch, in_width, in_channels]
    h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)   # [batch, out_width, out_channels] // eq // [batch, 1, out_width, out_channels]
    x_image2 = tf.reshape(h_conv1,[-1,1,n,32])
    h_pool1 = max_pool(x_image2,n)

    # do a fully connected layer
    W_fc1, b_fc1 = variables( [1 * 1 * 32,fc_nodes], [fc_nodes], 'W_fc1', 'b_fc1') 
    h_pool1_flat = tf.reshape(h_pool1, [-1, 1 * 1 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # final layer
    W_fc2, b_fc2 = variables([fc_nodes, 2], [2], 'W_fc2', 'b_fc2')
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv


def run_conv(xd,yd,zd,bd,fc_nodes,dropout,n,x,y_, keep_prob, X_train, X_test, y_train, y_test):
    """ calculating loss & training """
    y_conv = nn( xd, yd, zd, bd, fc_nodes,x,n,keep_prob)
    y_conv_softmax = tf.nn.softmax(y_conv) # for auROC/auPRC calculations.
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv,labels = y_))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    #saver = tf.train.Saver()

    # running the tensorflow graph - training

    with tf.Session() as sess:
        
        print "In session"
        sess.run(init)
        batch_size = 100
    
	epochs = 0

        while epochs < 20:
		
	    idx = 0
	    print len(X_train)
	    while idx + batch_size < len(X_train):
                batch_xs, batch_ys = X_train[idx:idx + batch_size], y_train[idx:idx+batch_size]
	        idx = idx + batch_size
		# train
                sess.run(train_step, feed_dict = { x: batch_xs, y_ : batch_ys, keep_prob: dropout})
            
	    # test_accuracy = accuracy.eval(feed_dict={x: X_test, y_ : y_test, keep_prob: 1})
	    # print "Epoch %d, test accuracy %f" % (epochs, test_accuracy)
            test_probas = y_conv_softmax.eval(feed_dict= {x: X_test, y_ : y_test, keep_prob: 1})
            auprc = average_precision_score(y_test, test_probas)
            with open(sys.argv[2], "a") as fp:
                fp.write("Epoch%d:%f\n"%(epochs,auprc))             
            epochs += 1
        save_path = saver.save(sess, "trained_model.ckpt")

def main():
    
    # reading the feature matrix & splitting into test and train. 

    X, y= read.get_matrix(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print y_test.shape
    print X_test.shape
    print y_test[:10]
    
    n = int(sys.argv[3])

    # creating placeholders for the features (x) , and the labels (y).

    x  = tf.placeholder(tf.float32, shape = (None, n*4)) # One-hot, so n*4
    y_ = tf.placeholder(tf.float32, shape = (None, 2))   # k = no_of_classes = 2
    keep_prob = tf.placeholder(tf.float32)
 
    run_conv(4,25,1,16, 512, 0.5, n, x, y_, keep_prob, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
