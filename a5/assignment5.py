#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 22:59:25 2017

@author: Yixuan Tan
"""

import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np

np.set_printoptions(threshold=np.inf)


def load_data():
    
    npzfile = np.load("train_and_val.npz")
    train_x = npzfile["train_x"]
    #print 'train_x is ', train_x.shape
    train_y = npzfile["train_y"]
    train_y = np.reshape(train_y, [-1, 1]);
    #print 'train_y is ', train_y[1:10]
    train_mask = npzfile["train_mask"]
    #print 'train_mask is ', train_mask[0, :]
    
    #Validation filenames follow the same pattern
    val_x = npzfile["val_x"]
    val_y = npzfile["val_y"]
    val_y = np.reshape(val_y, [-1, 1]);
    val_mask = npzfile["val_mask"]
    
    return train_x, train_y, train_mask, val_x, val_y, val_mask

def train(train_x, train_y, train_mask, val_x, val_y, val_mask):
    #set the structure variables
    vocab_size = train_y.shape[0]
    embedding_size = 300 # word_embedding_size
    max_sequence_length = 25 
    rnn_cell_size = 64
    val_size = val_x.shape[0]

    #set the training variables
    hm_epochs = 10
    batch_size = 1000
    rate = 0.01
    decay = 0.6
    schedule = 800

    #placeholder for the variables
    #size = tf.placeholder(tf.int32)
    #sequence_placeholder = tf.placeholder(tf.int32, [batch_size, max_sequence_length])
    sequence_placeholder = tf.placeholder(tf.int32, [None, max_sequence_length])
    
    y = tf.placeholder(tf.float32)
    #mask_placeholder = tf.placeholder(tf.float32, [batch_size, max_sequence_length])
    mask_placeholder = tf.placeholder(tf.float32, [None, max_sequence_length])


    
    min_after_dequeue = vocab_size
    capacity = min_after_dequeue + 3 * batch_size
    batch_x, batch_y, batch_mask = tf.train.shuffle_batch([train_x, train_y, train_mask], \
                                    batch_size=batch_size, capacity=capacity, \
                                    enqueue_many = True, min_after_dequeue=min_after_dequeue)


    ##########################################
    #initialization 
    w_embed = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.001))
    rnn_input = tf.cast(tf.nn.embedding_lookup(w_embed, sequence_placeholder), tf.float32)
    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_cell_size)
    print 'lstm_cell shape is ', lstm_cell
    #lstm_cell = tf.contrib.rnn.LSTMCell(em)
    outputs, states = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = rnn_input, dtype = tf.float32)
    length = tf.cast(tf.reduce_sum(mask_placeholder, reduction_indices=1), tf.int32)
    batch_len = tf.shape(outputs)[0]
    max_length = tf.shape(outputs)[1]
    out_size = tf.shape(outputs)[2]
    flat = tf.reshape(outputs, [-1, out_size])
    index = tf.range(0, batch_len) * max_length + (length - 1)
    relevant = tf.gather(flat, index) # relevant[total_input, rnn_cell_size]
    
    layer = {'weights':tf.Variable(tf.truncated_normal([rnn_cell_size, 1], stddev=0.001)),
                      'biases':tf.Variable(tf.truncated_normal([1], stddev=0.001))}
    pred = tf.matmul(relevant, layer['weights']) + layer['biases']

    ##########################################
    size = tf.placeholder(tf.int32)
    #prediction = recurrent_neural_network()
    label = tf.reshape(y, [size, 1])
    eta = tf.placeholder(tf.float32)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = label) )
    # NOTE: must use sigmoid_cross_entropy .... ; if using softmax_cross_entropy_... the output loss is 0.0????
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels = label, logits = pred) )
    optimizer = tf.train.AdamOptimizer(learning_rate = eta).minimize(cost)
    label = tf.cast(label, tf.int64)
    predict_op = tf.cast(tf.nn.sigmoid(pred)*2, tf.int64) #sigmoid function less than 0.5 return prediction 0
    correct = tf.equal(predict_op, label)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    
    tf.get_collection("validation_nodes")
    tf.add_to_collection("validation_nodes", sequence_placeholder)
    tf.add_to_collection("validation_nodes", mask_placeholder)
    tf.add_to_collection("validation_nodes", predict_op)
    saver = tf.train.Saver()
    
    # Launch the graph
    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tf.train.start_queue_runners(sess=sess)
        
        for epoch in range(hm_epochs):
            iter = 1
            for _ in range(int(vocab_size/batch_size)):
    
                epoch_x, epoch_y, epoch_mask = sess.run([batch_x, batch_y, batch_mask])
                #val_x, val_y, val_mask = sess.run([val_x, val_y, val_mask])
                #epoch_x = epoch_x.reshape((batch_size, max_sequence_length))
                if iter % schedule == 0:
                    rate = rate*decay
                _, cost_, pred_, accuracy_ = sess.run([optimizer, cost, pred, accuracy], \
                                                      feed_dict={sequence_placeholder: epoch_x, y: epoch_y, \
                                                                 mask_placeholder: epoch_mask, eta: rate, size: batch_size})
                val_accuracy_ = sess.run(accuracy, feed_dict ={sequence_placeholder: val_x, y: val_y, mask_placeholder: val_mask, size: val_size})
                #print 'cost: ', cost_, ' pred:', pred_, ' accuracy: ', accuracy_
                print 'iter: ', iter, 'cost: ', cost_, 'training accuracy: ', accuracy_, 'validation_accuracy', val_accuracy_ 
                #print 'iter: ', iter, '   loss: ', epoch_loss, '  correct: ', correct, ' accuracy: ', accuracy
                iter += 1
    
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', cost_)
    
        saver.save(sess, "my_model")
        #print('Accuracy:',accuracy.eval({va, y:mnist.test.labels}))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    return;

def main():
    train_x, train_y, train_mask, val_x, val_y, val_mask = load_data();
    train(train_x, train_y, train_mask, val_x, val_y, val_mask);


    
if __name__ == '__main__':
    main()
    
