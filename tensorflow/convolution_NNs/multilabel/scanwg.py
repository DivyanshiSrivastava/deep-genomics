# Train a single layer neural network ( CNN ) using sequence.
# Impementation: tensorflow v. 0.11

import sys
import tensorflow as tf
import numpy as np
import read
# from sklearn.model_selection import train_test_split

def read_single_example(filename, record_defaults):
    # construct a queue for all the files.
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs = 1,
                                                    name = "myQueue")
    # create a reader. 
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    int_example = tf.decode_csv(value, record_defaults = record_defaults)
    example = tf.cast(int_example, tf.float32)
    return example

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

def nn(xd,yd,zd,bd,fc_nodes,examples_batch,n,keep_prob):
    """ convolve over feature matrix """

    W_conv1, b_conv1 = variables([20,4,32], [32], 'W_conv1', 'b_conv1')
    x_image = tf.reshape(examples_batch, [-1,n,4])                          # [batch, in_width, in_channels]
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


def run_conv(xd,yd,zd,bd,fc_nodes,dropout,n, examples_batch, keep_prob):
    """ calculating loss & training """
    y_conv = nn( xd, yd, zd, bd, fc_nodes,examples_batch,n,keep_prob)
    y_conv_softmax = tf.nn.softmax(y_conv) # for auROC/auPRC calculations.
    
    init = tf.initialize_all_variables()
    init_local = tf.initialize_local_variables()
    saver = tf.train.Saver()

    # running the tensorflow graph - training

    with tf.Session() as sess:
        
        print "In session"
        sess.run(init)
        sess.run(init_local)
        saver.restore(sess, tf.train.latest_checkpoint('./'))
    
        # start enque threads:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord = coord)
        steps = 1
        try:
          while not coord.should_stop():
            temp = y_conv_softmax.eval(feed_dict = { keep_prob: 1})
            f = "probas/outfile" + str(steps)
            np.savetxt(f, temp)
            print "Done step %d" % steps
            steps = steps + 1
        except tf.errors.OutOfRangeError:
          print "Reached End"
        finally:
          coord.request_stop()

        # Wait for the threads to finish
        coord.join(threads)
        sess.close()

def main():
    
    n = 200
    record_defaults = list()
    for i in range(800):
        record_defaults.append([0])

    # get single examples
    example = read_single_example(sys.argv[1], record_defaults)
    # groups examples into batches randomly
    examples_batch = tf.train.batch([example], batch_size=50000,capacity=200000, allow_smaller_final_batch=True)
    
    keep_prob = tf.placeholder(tf.float32)

    run_conv(4,25,1,16, 512, 0.5, n, examples_batch, keep_prob)

if __name__ == "__main__":
    main()

