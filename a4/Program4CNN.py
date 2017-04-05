#!/usr/bin/env python2
# Ni Zhang
"""
Created on Sun Apr  2 21:13:14 2017

@author: nizhang
"""
#from __future__ import print_function
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import cv2

debug = True

def normalize(train, test):
    train_mean = tf.reduce_mean(train,keep_dims = True)
    test_mean = tf.reduce_mean(test,keep_dims = True)

    train = (train - train_mean)
    test = (test - test_mean)
    return train, test

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    #if debug: print('x type is ', x)
    #if debug: print('W type is ', W)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding = 'VALID')



# Create model
def conv_net(x, weights, biases, dropout):
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    #conv3 = maxpool2d(conv2, k=1)
                      
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    #print('conv3 shape ', conv3.get_shape())
    #print('have ', weights['wd1'].get_shape().as_list()[0])
    fc1 = tf.reshape(conv3, [-1, 3 * 3 * 64])
    
    
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    #print('fc1 is ', tf.argmax(fc1, 1))
    return fc1

def main():
    data_file = file("cifar_10_tf_train_test.pkl", "rb")
    #data_file = file("small.pkl", "rb")

    train_x, train_y, test_x, test_y = pickle.load(data_file)
    data_file.close()
    #cv2.imshow("test", train_x[0])
    #cv2.waitKey(1)
    
    #floats converting and scaling
    train_x = tf.cast(train_x, tf.float32)/ 255.0
    test_x = tf.cast(test_x, tf.float32) / 255.0
    #normalizing
    train_x, test_x = normalize(train_x, test_x)
    
    
    # Parameters
    learning_rate = 0.005
    training_iters = 200000
    batch_size = 128
    display_step = 10
    
    # Network Parameters
    n_input = 50000 #
    n_classes = 10 # MNIST total classes (0-9 digits)
    dropout = 1.0 #0.5 # Dropout, probability to keep units

    min_after_dequeue = n_input
    capacity = min_after_dequeue + 3 * batch_size
    batch_x, batch_y = tf.train.shuffle_batch([train_x, train_y], batch_size=batch_size, capacity=capacity, enqueue_many = True, min_after_dequeue=min_after_dequeue)

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.001)), 
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 32],stddev=0.001)),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.001)),
        # fully connected, 3*3*64 inputs, 10 outputs
        'wd1': tf.Variable(tf.truncated_normal([3 * 3 * 64, 10],stddev=0.001)),
        # 1024 inputs, 10 outputs (class prediction)
        #'out': tf.Variable(tf.random_normal([3 * 3 * 64, 10]))
    }

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'bc2': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'bc3': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
        'bd1': tf.Variable(tf.truncated_normal([10], stddev=0.001)),
        #'out': tf.Variable(tf.random_normal([10]))
    }
    '''
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([32])),
    'bc3': tf.Variable(tf.truncated_normal([64])),
    'bd1': tf.Variable(tf.truncated_normal([10])),
    '''    
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.int64, [None])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    pred = conv_net(x, weights, biases, keep_prob);

    #if debug : print('pred is ', tf.argmax(pred, 1))
    
    # Define loss and optimizer
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=train_y))
    #if debug: print('labels is ', train_y)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    #print('pred is ', pred)
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 1
        
        tf.train.start_queue_runners(sess=sess)
        # Keep training until reach max iterations
        while step < training_iters:
            #if debug: print('step is ', step)
            '''
            indices = np.random.choice(n_input, batch_size)
            print 'len of indices ', len(indices)
            print ('indices', indices)
            print 'train_x type ', train_x

            batch_x = train_x[indices]
            #print ('batch_x', batch_x)
            batch_y = np.asarray(train_y)[indices]
            
            '''
            # collect batches of images before processing
            #batch_x, batch_y = tf.train.batch([train_x, train_y], batch_size = batch_size, enqueue_many = True)
            #if debug: print 'batch further'
            #batch_x, batch_y = xs.eval(), ys.eval()
            #if debug: print 'batch finished'
            #print batch_x.eval()
            #
            # Run optimization op (backprop)
            #print 'batch_y', batch_y
            py_images, py_labels = sess.run([batch_x, batch_y])
            sess.run(optimizer, feed_dict={x: py_images, y: py_labels, keep_prob: dropout})
            #if debug: print 'run'
            #print 'wc1 ', weights['wc1'].eval()
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: py_images, y: py_labels, keep_prob: 1.})
                print("Iter ", str(step), ", Minibatch Loss= ", "{:.6f}".format(loss), ", Training Accuracy= " + "{:.5f}".format(acc))
                #print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                
            step += 1
            
        coord.request_stop()
        coord.join(threads)

    
if __name__ == '__main__':
    main()