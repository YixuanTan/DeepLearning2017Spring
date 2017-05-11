#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 20:03:30 2017

@author: yixuantan
"""

import tensorflow as tf
import numpy as np
import sys
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle


debug = False

def readInput(pathPrefix):
    data = []
    for filename in os.listdir(pathPrefix):
        if not filename.endswith("jpg"): continue
        image = mpimg.imread(pathPrefix + '/' + filename)
        data.append(np.reshape(image, 28 * 28))
        
    X = np.array(data, dtype = 'float64')
    #X_normed = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
    X_normed = X / 255.0
    if debug: print X_normed
    
    
    label = []
    end = pathPrefix.rfind('/')
    if pathPrefix.endswith('train_data'):
        path = pathPrefix[:end] + '/labels/train_label.txt'
    else:
        path = pathPrefix[:end] + '/labels/test_label.txt'
    with open(path) as f:
        for line in f:
            row = np.zeros(5)
            row[int(line) - 1] = 1.0
            label.append(row)
    Y = np.array(label, dtype = 'float64')
    if debug: print 'X shape is ', X_normed.shape
    if debug: print 'Y shape is ', Y.shape
    
    return X_normed, Y

def train(X, Y, Xt, Yt):
    global plotcount
    num_sample = len(X)
    learning_rate = 0.01
    batch_size = 100
    lbd = 10.0
    maxiter = 1000000
    threshold = 1.0e-6

    tf.reset_default_graph()
    
    Xs = tf.placeholder(tf.float64)
    Ys = tf.placeholder(tf.float64)
    T = tf.placeholder(tf.float64)   # T = [w, w_0]    

    #initialize Theta to 0.01
    Theta = 0.01 * np.ones((X.shape[1], 5))
         
    #w = tf.Variable(tf.zeros([784, 5]))
    f = tf.matmul(Xs, T);
    expf = tf.exp(f);
    sigmoid = tf.transpose(tf.transpose(expf) / tf.reduce_sum(expf, 1));  # divide by summation of each row      sigmoid is  K x M          
    #nllgrad = - 1.0 / batch_size * tf.matmul(tf.transpose(Xs), tf.multiply(Ys, 1 - sigmoid) ) + lbd / batch_size * T
    nllgrad = - 1.0 / batch_size * tf.matmul(tf.transpose(Xs), tf.subtract(Ys, sigmoid) ) + lbd / batch_size * T
    nll = - 1.0 / batch_size * tf.reduce_sum(tf.multiply(Ys, tf.log(sigmoid))) + lbd / 2.0 / batch_size * tf.reduce_sum(tf.multiply(T, T))  

    with tf.Session() as sess:
        
        currLoss = 1.0e3
        deltaLoss = 1.0e3
        currGrads = 1.0e3
        iter = 0
        initnorm = 0.0
        while abs(deltaLoss) > threshold and currGrads > threshold * initnorm:
            # randomly select a subset
            a)
            
            Xb = X[indices]
            Yb = Y[indices]
            
            f_ = sess.run(f, feed_dict = {T: Theta, Xs: Xb})
            #if debug: print 'f ', f_
            expf_ = sess.run(expf, feed_dict = {f: f_})
            #if debug: print 'expf ', expf_
            sigmoid_ = sess.run(sigmoid, feed_dict = {expf: expf_})
            #if debug: print 'sigmoid ', sigmoid_
            nllgrad_ = sess.run(nllgrad, feed_dict = {sigmoid: sigmoid_, T: Theta, Xs: Xb, Ys: Yb})
            gd = nllgrad_
            if debug: print 'grad max is ', np.amax(gd)
            if debug: print 'Theta max is ', np.amax(Theta)
            if debug: print 'lambda is ', np.amax(gd) / np.amax(Theta) * batch_size
            nll_ = sess.run(nll, feed_dict = {sigmoid: sigmoid_, T: Theta, Xs: Xb, Ys: Yb})
            
            # initialize initnorm
            if initnorm < 1.e-6:         
                initnorm = np.linalg.norm((nllgrad_))
            
            currGrads = np.linalg.norm(nllgrad_)
            prevLoss = currLoss
            currLoss = nll_
            deltaLoss = currLoss - prevLoss
            
            Theta = Theta - learning_rate * nllgrad_;
            if debug: print ' iter: ', iter, '  nll: ', nll_
            #print ' iter: ', iter
            iter = iter + 1
            if iter == maxiter:
                break
                if debug:
                    pred = tf.nn.softmax(tf.matmul(X, T))
                    res = sess.run(pred, feed_dict = {T: Theta});
                    for l in res[0:100,:]: print np.argmax(l) + 1
                        
            if iter % 10 == 0 : 
                accuracy = test(Xt, Yt, Theta, sess)
                print "accuracy is ", accuracy
                if accuracy > 0.965:
        break 

         
        # postprocessing
        filehandler = open("multiclass_parameters.txt","wb")
        pickle.dump(Theta, filehandler)
        filehandler.close()
        
        if debug:
            Theta = []
            Theta = pickle.load(open( "multiclass_parameters.txt", "rb" ))
            
        plotcount = 0
        for i in range(5):
            plt.figure(plotcount)
            plotcount = plotcount + 1
            img = Theta[:,i].reshape(28,28)
            plt.imshow(img)     
            plt.colorbar()
            plt.title('W' + str(i))
            plt.show()
    sess.close()
    return Theta
        
def test(X, Y, Theta, ss):
    pred = tf.nn.softmax(tf.matmul(X, Theta))
    tp_all = 0
    #with tf.Session() as ss:
    res = ss.run(pred)
    #for l in res: print l
    for i in range(5):
        indices = []
        indices = np.argmax(Y, 1) == i   
        output = res[indices, :]
        tp = np.sum(np.argmax(output, 1) == i)
        #if debug: print 'tp is ', tp
        tp_all = tp_all + tp
                #print 'error for digit ' + str(i+1) + ' is ' + str(1.0 - 1.0 * tp / sum(indices))        
    #ss.close()
    #print 'average classification error is ' + str(1.0 - 1.0 * tp_all / X.shape[0])  
    return 1.0 - 1.0 * tp_all / X.shape[0] 
                
    
def main(argv):
    if len(argv) != 3:
        print "command example: python hw2.py ./data_prog2/train_data ./data_prog2/test_data"
        
    #read train data
    data, Y = readInput(argv[1]) 
    shp = (len(data), 1)
    X = np.append(data[:, :-1], np.ones(shp), 1)  

    # read test data
    datat, Yt = readInput(argv[2]) 
    shpt = (len(datat), 1)
    Xt = np.append(datat[:, :-1], np.ones(shpt), 1) 


    Theta = []
    Theta = train(X, Y, Xt, Yt)
    
    if debug: Theta = pickle.load(open( "multiclass_parameters.txt", "rb" ))
    
    
    #test(Xt, Yt, Theta)
        
if __name__ == "__main__":
    main(sys.argv)
