#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:20:02 2017

@author: nizhang
"""

import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np

np.set_printoptions(threshold=np.inf)


def load_data():
    npzfile = np.load("train_and_val.npz")
    
    train_eye_left = np.float32(npzfile["train_eye_left"])/255.0
    train_eye_right = np.float32(npzfile["train_eye_right"])/255.0
    train_face = np.float32(npzfile["train_face"])/255.0
    train_face_mask = np.float32(npzfile["train_face_mask"])/255.0
    train_y = np.float32(npzfile["train_y"])

    val_eye_left = np.float32(npzfile["val_eye_left"])/255.0
    val_eye_right = np.float32(npzfile["val_eye_right"])/255.0
    val_face = np.float32(npzfile["val_face"])/255.0
    val_face_mask = np.float32(npzfile["val_face_mask"])/255.0
    val_y = np.float32(npzfile["val_y"])  
    
    return train_eye_left, train_eye_right, train_face, \
            train_face_mask, train_y, val_eye_left, val_eye_right, \
            val_face, val_face_mask, val_y

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and leaky relu activation
    
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding = 'VALID')
    x = tf.nn.bias_add(x, b)
    
    return tf.maximum(0.01*x,x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding = 'VALID')

# Create model
def conv_net(eye_left, eye_right, face, face_mask, weights, biases):
    
    # Left eye training
    # First Convolution Layer
    eyeleft_conv1 = conv2d(eye_left, weights['eyeleft_wc1'], biases['eyeleft_bc1'])
    
    # First Max Pooling (down-sampling)
    eyeleft_conv1 = maxpool2d(eyeleft_conv1, k=2)

    # Second Convolution Layer
    eyeleft_conv2 = conv2d(eyeleft_conv1, weights['eyeleft_wc2'], biases['eyeleft_bc2'])
    # Second Max Pooling (down-sampling)
    eyeleft_conv2 = maxpool2d(eyeleft_conv2, k=2)

    # Third Convolution Layer
    eyeleft_conv3 = conv2d(eyeleft_conv2, weights['eyeleft_wc3'], biases['eyeleft_bc3'])
                      
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    eyeleft_fc1 = tf.reshape(eyeleft_conv3, [-1, 11 * 11 * 64])
    eyeleft_out = tf.add(tf.matmul(eyeleft_fc1, weights['eyeleft_wout']), biases['eyeleft_bout'])
    
    # Right eye training
    # First Convolution Layer
    eyeright_conv1 = conv2d(eye_right, weights['eyeright_wc1'], biases['eyeright_bc1'])
    
    # First Max Pooling (down-sampling)
    eyeright_conv1 = maxpool2d(eyeright_conv1, k=2)

    # Second Convolution Layer
    eyeright_conv2 = conv2d(eyeright_conv1, weights['eyeright_wc2'], biases['eyeright_bc2'])
    # Second Max Pooling (down-sampling)
    eyeright_conv2 = maxpool2d(eyeright_conv2, k=2)

    # Third Convolution Layer
    eyeright_conv3 = conv2d(eyeright_conv2, weights['eyeright_wc3'], biases['eyeright_bc3'])
                      
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    eyeright_fc1 = tf.reshape(eyeright_conv3, [-1, 11 * 11 * 64])
    eyeright_out = tf.add(tf.matmul(eyeright_fc1, weights['eyeright_wout']), biases['eyeright_bout'])
    
    concat = tf.concat([eyeleft_out, eyeright_out], 1)
    
    fc1_eye = tf.add(tf.matmul(concat, weights['eye_w']), biases['eye_b'])
    
    # Face training
    # First Convolution Layer
    face_conv1 = conv2d(face, weights['face_wc1'], biases['face_bc1'])
    
    # First Max Pooling (down-sampling)
    face_conv1 = maxpool2d(face_conv1, k=2)

    # Second Convolution Layer
    face_conv2 = conv2d(face_conv1, weights['face_wc2'], biases['face_bc2'])
    # Second Max Pooling (down-sampling)
    face_conv2 = maxpool2d(face_conv2, k=2)

    # Third Convolution Layer
    face_conv3 = conv2d(face_conv2, weights['face_wc3'], biases['face_bc3'])
                      
    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    face_fc1 = tf.reshape(face_conv3, [-1, 11 * 11 * 64])
    face_out = tf.add(tf.matmul(face_fc1, weights['face_wout']), biases['face_bout'])
    fc2_face = tf.add(tf.matmul(face_out, weights['fc2_wface']), biases['fc2_bface'])
    
    concat_1 = tf.concat([fc1_eye, fc2_face], 1)
    
    reshape_mask = tf.reshape(face_mask, [-1, 25 * 25])
    
    fc1_mask = tf.add(tf.matmul(reshape_mask, weights['fc1_wmask']), biases['fc1_bmask'])
    
    fc2_mask = tf.add(tf.matmul(fc1_mask, weights['fc2_wmask']), biases['fc2_bmask'])
    concat_2 = tf.concat([concat_1, fc2_mask], 1)

    fc1_face_eye_mask = tf.add(tf.matmul(concat_2, weights['fc1_face_eye_wmask']), biases['fc1_face_eye_bmask'])
    
    output_final = tf.add(tf.matmul(fc1_face_eye_mask, weights['output_wfinal']), biases['output_bfinal'])

    return output_final


def train(train_eye_left, train_eye_right, train_face, \
            train_face_mask, train_y, val_eye_left, val_eye_right, \
            val_face, val_face_mask, val_y ):
    # Parameters
    rate = 0.001
    #training_iters = 100
    batch_size = 256
    #display_step = 10
    hm_epoches = 1
    # Network Parameters
    n_input = train_eye_left.shape[0]
    schedule = 800
    decay = 0.6
    
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'eyeleft_wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.001)), 
        # 5x5 conv, 32 inputs, 64 outputs
        'eyeleft_wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 32],stddev=0.001)),
        'eyeleft_wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.001)),
        # fully connected, 11*11*64 inputs, 10 outputs
        'eyeleft_wout': tf.Variable(tf.truncated_normal([11 * 11 * 64, batch_size],stddev=0.001)),
        # 1024 inputs, 10 outputs (class prediction)
        # 5x5 conv, 1 input, 32 outputs
        'eyeright_wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.001)), 
        # 5x5 conv, 32 inputs, 64 outputs
        'eyeright_wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 32],stddev=0.001)),
        'eyeright_wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.001)),
        # fully connected, 11*11*64 inputs, 10 outputs
        'eyeright_wout': tf.Variable(tf.truncated_normal([11 * 11 * 64, batch_size],stddev=0.001)),
        
        #concatenate left eye with right eye 
        'eye_w': tf.Variable(tf.truncated_normal([2*batch_size, batch_size],stddev=0.001)),
                            
                                    
        # 1024 inputs, 10 outputs (class prediction)
        # 5x5 conv, 1 input, 32 outputs
        'face_wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.001)), 
        # 5x5 conv, 32 inputs, 64 outputs
        'face_wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 32],stddev=0.001)),
        'face_wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.001)),
        # fully connected, 11*11*64 inputs, 10 outputs
        'face_wout': tf.Variable(tf.truncated_normal([11 * 11 * 64, batch_size],stddev=0.001)),
        
        'fc2_wface': tf.Variable(tf.truncated_normal([batch_size, 2*batch_size], stddev=0.001)),

        'fc1_wmask': tf.Variable(tf.truncated_normal([25 * 25, batch_size], stddev=0.001)),
        'fc2_wmask': tf.Variable(tf.truncated_normal([batch_size, batch_size/2], stddev=0.001)),
        'fc1_face_eye_wmask': tf.Variable(tf.truncated_normal([896, 256], stddev=0.001)), 
        'output_wfinal': tf.Variable(tf.truncated_normal([256, 2], stddev=0.001))                              
        
    }

    biases = {
        'eyeleft_bc1': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'eyeleft_bc2': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'eyeleft_bc3': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
        'eyeleft_bout': tf.Variable(tf.truncated_normal([batch_size], stddev=0.001)),
        
        'eyeright_bc1': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'eyeright_bc2': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'eyeright_bc3': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
        'eyeright_bout': tf.Variable(tf.truncated_normal([batch_size], stddev=0.001)),
                           
        'eye_b': tf.Variable(tf.truncated_normal([batch_size], stddev=0.001)),

        'face_bc1': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'face_bc2': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
        'face_bc3': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
        'face_bout': tf.Variable(tf.truncated_normal([batch_size], stddev=0.001)),
        
        'fc2_bface': tf.Variable(tf.truncated_normal([2*batch_size], stddev=0.001)),
        
        'fc1_bmask': tf.Variable(tf.truncated_normal([batch_size], stddev=0.001)),
        'fc2_bmask': tf.Variable(tf.truncated_normal([batch_size/2], stddev=0.001)),
        'fc1_face_eye_bmask': tf.Variable(tf.truncated_normal([256], stddev=0.001)), 

        'output_bfinal': tf.Variable(tf.truncated_normal([2], stddev=0.001))                              

    }
    
     # tf Graph input
    eyeleft_inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    eyeright_inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    face_inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    mask_inputs = tf.placeholder(tf.float32, [None, 25, 25])
    
    y = tf.placeholder(tf.float32, [None, 2])
    eta = tf.placeholder(tf.float32)

    pred = conv_net(eyeleft_inputs, eyeright_inputs, face_inputs, mask_inputs, weights, biases)
    #pred_op = tf.nn.softmax(pred)
    #pred_op = tf.argmax(pred_op, 1)
    # create the collection
    tf.get_collection("validation_nodes")
    # Add to the collection
    tf.add_to_collection("validation_nodes", eyeleft_inputs)
    tf.add_to_collection("validation_nodes", eyeright_inputs)
    tf.add_to_collection("validation_nodes", face_inputs)
    tf.add_to_collection("validation_nodes", mask_inputs)
    tf.add_to_collection("validation_nodes", pred)
    # Define loss and optimizer
    #cost = tf.reduce_mean(tf.metrics.mean_squared_error(labels=y, predictions=pred))
    #err = np.mean(np.sqrt(np.sum((pred - y)**2, axis=1)))
    err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred, y)))))
    optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(err)

    # Evaluate model
    #correct_pred = tf.equal(pred_op, y)
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    # Initializing the variables
    init = tf.global_variables_initializer()
    train_err_set = []
    test_err_set = []
    # Create a saver.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(hm_epoches):
    
            for step in range(int(n_input/batch_size)):
                choices = np.random.choice(n_input, batch_size, replace = False)
                batch_eye_left = train_eye_left[choices]
                batch_eye_right = train_eye_right[choices]
                batch_face = train_face[choices]
                batch_face_mask = train_face_mask[choices]
    
                batch_y = train_y[choices]
                print 'training start'
                if step % schedule == 0:
                    rate = rate*decay
                    print 'learning rate decreased to: '+ str(rate)
                sess.run(optimizer, feed_dict={eyeleft_inputs: batch_eye_left, eyeright_inputs: batch_eye_right, \
                                               face_inputs:batch_face, mask_inputs: batch_face_mask, y: batch_y, eta: rate})
            train_err = sess.run(err, feed_dict={eyeleft_inputs: train_eye_left[:batch_size], eyeright_inputs: train_eye_right[:batch_size], \
                                           face_inputs:train_face[:batch_size], mask_inputs: train_face_mask[:batch_size], y: train_y[:batch_size]})
            print "Iteration: "+str(step)  + ", train error = "+"{:.5f}".format(train_err)
            
            test_err = sess.run(err, feed_dict={eyeleft_inputs: val_eye_left[:batch_size], eyeright_inputs: val_eye_right[:batch_size], \
                                               face_inputs:val_face[:batch_size], mask_inputs: val_face_mask[:batch_size], y: val_y[:batch_size]})
            print "Iteration: "+str(step) + ", test accuracy= "+"{:.5f}".format(test_err)
            train_err_set.append(train_err)

            test_err_set.append(test_err)        
                    
    
        saver.save(sess, "my_model")    
        sess.close()
    

def main():
    train_eye_left, train_eye_right, train_face, \
            train_face_mask, train_y, val_eye_left, val_eye_right, \
            val_face, val_face_mask, val_y = load_data()
    train(train_eye_left, train_eye_right, train_face, \
            train_face_mask, train_y, val_eye_left, val_eye_right, \
            val_face, val_face_mask, val_y )
    
if __name__ == '__main__':
    main()
