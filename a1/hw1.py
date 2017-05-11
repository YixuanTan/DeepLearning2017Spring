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
import matplotlib.pyplot as plt


threshold = 1.0e-6
debug = False;
maxiter = 500
plotcount = 1;
def p1(X, Y):
    # problem 1
    with tf.Session() as sess:
        model = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)), tf.transpose(X)), Y)
        result = sess.run(model)
        print 'question 1: Theta is '
        for l in result: print l
        sess.close()


def p2(X, Y):
    # problem 2
    global plotcount
    num_sample = len(X)
    
    T = tf.placeholder(tf.float64)           
        
    #define lose function
    loss = tf.matmul(tf.transpose(tf.matmul(X, T) - Y), tf.matmul(X, T) - Y) / num_sample
    #define gradient
    grads = 2. * tf.matmul(tf.transpose(X), tf.matmul(X, T) - Y) / num_sample 

    for learning_rate in [0.001]:
        iteration = []
        lossval = []
        #initialize Theta to 0.01
        Theta = 0.01 * np.ones((X.shape[1], 5)) # 5 columns for 5 labels
        
        with tf.Session() as sess:
            currLoss = 1.0e3
            deltaLoss = 1.0e3
            currGrads = 1.0e3
            initnorm = np.linalg.norm(sess.run(grads, feed_dict = {T: Theta}))
            iter = 0
            while deltaLoss > threshold or np.linalg.norm(currGrads) > threshold * initnorm:
                prevLoss = currLoss
                iter = iter + 1
                
                currGrads = sess.run(grads, feed_dict = {T: Theta})
                Theta = Theta - learning_rate * currGrads
                currLoss = sess.run(loss, feed_dict = {T: Theta})[0]
                deltaLoss = abs(currLoss - prevLoss)
                if debug: print ' iter: ', iter, 'currLoss: ', currLoss, 'deltaLoss: ', deltaLoss, '  grad norm: ', np.linalg.norm(currGrads)
                iteration.append(iter)
                lossval.append(currLoss)
                if iter == maxiter:
                    break
                
            print 'question 2: Theta is (learning rate = ', learning_rate, ') '
            for l in Theta: print l 
            
            plt.figure(plotcount)
            plotcount = plotcount + 1
            plt.plot(iteration, lossval)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.title('learning rate = ' + str(learning_rate))

            #plt.hold(True)
            plt.show()
            sess.close()
        



def p3(X, Y):
    global plotcount
    num_sample = len(X)
    learning_rate = 0.001
    
    Xs = tf.placeholder(tf.float64)
    Ys = tf.placeholder(tf.float64)
    T = tf.placeholder(tf.float64)           
    
    
    
    for batch_size in [10, 15, 20]:
        #initialize Theta to 0.01
        Theta = 0.01 * np.ones((X.shape[1], 1))

        
        #define lose function
        loss = tf.matmul(tf.transpose(tf.matmul(Xs, T) - Ys), tf.matmul(Xs, T) - Ys) / batch_size
        #define gradient
        #grads = tf.gradients(loss, [T]) 
        grads = 2. * tf.matmul(tf.transpose(Xs), tf.matmul(Xs, T) - Ys) / batch_size

        with tf.Session() as sess:
            iteration = []
            lossval = []

            currLoss = 1.0e3
            deltaLoss = 1.0e3
            currGrads = 1.0e3
            iter = 0
            initnorm = 0
            while deltaLoss > threshold or np.linalg.norm(currGrads) > threshold * initnorm :
                # randomly select a subset
                indices = random.sample(xrange(num_sample), batch_size)
                
                Xb = X[indices]
                Yb = Y[indices]
                
                # initialize initnorm
                if initnorm == 0:         
                    initnorm = np.linalg.norm(sess.run(grads, feed_dict = { T: Theta, Xs: Xb, Ys: Yb}))
    
                prevLoss = currLoss
                iter = iter + 1
                
                currGrads = sess.run(grads, feed_dict = { T: Theta, Xs: Xb, Ys: Yb})
                Theta = Theta - learning_rate * np.array(currGrads)
                currLoss = sess.run(loss, feed_dict = {T: Theta, Xs: Xb, Ys: Yb})[0]
                deltaLoss = abs(currLoss - prevLoss)
                if debug: print ' iter: ', iter, 'currLoss: ', currLoss, 'deltaLoss: ', deltaLoss, '  grad norm: ', np.linalg.norm(currGrads)
                iteration.append(iter)
                lossval.append(currLoss)
                if iter == maxiter:
                    break
                
            print 'question 3:  Theta is (batch size = ', batch_size, ') '
            for l in Theta: print l 
            
            plt.figure(plotcount)
            plotcount = plotcount + 1
            plt.plot(iteration, lossval)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.title('batch size = ' + str(batch_size))
            plt.show()
            sess.close()

                
    
def main(argv):
    
    data = np.loadtxt(argv[1]) 
    shp = (len(data), 1)
    X = np.append(data[:, :-1], np.ones(shp), 1)  
    Y = np.transpose(np.matrix(data[:, -1]))  #Y = np.transpose(data[:, -1])
    p1(X, Y)
    p2(X, Y)
    p3(X, Y)
    
if __name__ == "__main__":
    main(sys.argv)
