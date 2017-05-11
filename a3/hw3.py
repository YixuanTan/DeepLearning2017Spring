#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:52:21 2017

@author: yixuantan
"""
import tensorflow as tf
import numpy as np
#np.set_printoptions(threshold=np.inf)
import sys
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import pickle


debug = False

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def train(X, Y):
    num_sample = len(X)
    eta1 = 1
    eta10 = eta1
    eta2 = eta1
    eta20 = eta1
    eta3 = eta1
    eta30 = eta1
    batch_size = 50
    maxIter = 30
    tf.reset_default_graph()
    
    H0 = tf.placeholder(tf.float64, shape=[50, 784, 1])
    W1 = tf.placeholder(tf.float64, shape=[50, 784, 100]) 
    W10 = tf.placeholder(tf.float64, shape=[50, 100, 1]) 
    H1 = tf.placeholder(tf.float64, shape=[50, 100, 1]) 
    W2 = tf.placeholder(tf.float64, shape=[50, 100, 100]) 
    W20 = tf.placeholder(tf.float64, shape=[50, 100, 1]) 
    H2 = tf.placeholder(tf.float64, shape=[50, 100, 1]) 
    W3 = tf.placeholder(tf.float64, shape=[50, 100, 10]) 
    W30 = tf.placeholder(tf.float64, shape=[50, 10, 1]) 
    H3 = tf.placeholder(tf.float64, shape=[50, 10, 1]) 
    
    
    Z1 = tf.matmul(tf.transpose(W1, perm=[0, 2, 1]), H0) + W10
    H1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(tf.transpose(W2, perm=[0, 2, 1]), H1) + W20
    H2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(tf.transpose(W3, perm=[0, 2, 1]), H2) + W30
    H3 = tf.nn.softmax(Z3, dim = 1)
    
    with tf.Session() as sess:
        # initialization
        W1_avg = 0.01*tf.random_normal(shape = [784, 100], stddev = .1).eval()
        W2_avg = 0.01*tf.random_normal(shape = [100, 100], stddev = .1).eval()
        W3_avg = 0.01*tf.random_normal(shape = [100, 10], stddev = .1).eval()
        W1_ = tf.zeros(shape = [50, 784, 100]).eval()
        W2_ = tf.zeros(shape = [50, 100, 100]).eval()
        W3_ = tf.zeros(shape = [50, 100, 10]).eval()
        for m in range(50):
            W1_[m] = np.copy(W1_avg)
            W2_[m] = np.copy(W2_avg)
            W3_[m] = np.copy(W3_avg)
            
        W10_ = tf.zeros(shape = [50, 100, 1]).eval()
        W20_ = tf.zeros(shape = [50, 100, 1]).eval()
        W30_ = tf.zeros(shape = [50, 10, 1]).eval()
        W10_avg = np.copy(W10_[0])
        W20_avg = np.copy(W20_[0])
        W30_avg = np.copy(W30_[0])
        
        '''
        W1_ = 0.01*tf.truncated_normal(shape = [50, 784, 100], stddev = 0.1).eval()
        W10_ = tf.zeros(shape = [50, 100, 1]).eval()
        W1_avg = W1_[0]
        W10_avg = W10_[0]
        
        W2_ = 0.01*tf.truncated_normal(shape = [50, 100, 100], stddev = 0.1).eval()
        W20_ = tf.zeros(shape = [50, 100, 1]).eval()
        W2_avg = W2_[0]
        W20_avg = W20_[0]
        
        W3_ = 0.01*tf.truncated_normal(shape = [50, 100, 10], stddev = 0.1).eval()
        W30_ = tf.zeros(shape = [50, 10, 1]).eval()
        W3_avg = W3_[0]
        W30_avg = W30_[0]
        '''
        
        iter = 0
        while True:
            # batching
            indices = random.sample(xrange(num_sample), batch_size)
            Xs = X[indices]
            Ys = Y[indices]
    
            H0_ = tf.reshape(Xs, [50, 784, 1]).eval()
            #reshprint 'H0_ ', np.reshape(H0_, (50, 784))
            #Forward propagation
            Z1_ = sess.run(Z1, feed_dict = {W1 : W1_ , H0 : H0_, W10 : W10_})
            if debug: print 'Z1_ ', Z1_.shape
            H1_ = sess.run(H1, feed_dict = {Z1 : Z1_})
            Z2_ = sess.run(Z2, feed_dict = {W2 : W2_, H1: H1_, W20 : W20_})
            if debug: print 'Z2_ ', Z2_.shape
            H2_ = sess.run(H2, feed_dict = {Z2 : Z2_})
            if debug: print 'H2_ ', H2_.shape
    
            Z3_ = sess.run(Z3, feed_dict = {W3 : W3_, H2: H2_, W30 : W30_})
            if debug: print 'Z3_ ', Z3_.shape
            H3_ = sess.run(H3, feed_dict = {Z3 : Z3_})
            
            if debug: print 'H3_ ', H3_.shape
            loss = 0.5 * np.linalg.norm(np.reshape(H3_,[50,10]) - Ys)
            #if iter % 5 == 0 : print np.reshape(H3_,[50,10])
            print 'loss ', loss 
            # backward propagation
            dY = tf.subtract(H3_, tf.reshape(Ys, shape=[50, 10, 1])).eval()
            if debug: print 'dY', dY.shape
            
            #dW3 = tf.matmul(H3_, tf.transpose(dY, perm=[0, 2, 1])) 
            #Sigma3 = tf.nn.softmax(Z3_).eval()
            #if debug : print 'Sigma3', Sigma3
            
            #output layer
            dW3 = np.zeros([50, 100, 10])
            dW30 = np.zeros([50, 10, 1])
            dH2 = np.zeros([50, 100, 1]);
            for m in range(50):
                for k in range(10):
                    for i in range(10):    
                        if i == k:
                            #if debug: print 'H2[m,:,0] shape ', H2_[m,:,0].shape
                            dW3[m,:,k] = dW3[m,:,k] + H3_[m,k,0] * (1 - H3_[m,k,0]) * H2_[m,:,0] * dY[m,i,0]
                            dW30[m,k,0] = dW30[m,k,0] + H3_[m,k,0] * (1 - H3_[m,k,0]) * dY[m,i,0]
                        else:
                            dW3[m,:,k] = dW3[m,:,k] - H3_[m,k,0] * H3_[m,i,0] * H2_[m,:,0] * dY[m,i,0]
                            dW30[m,k,0] = dW30[m,k,0] - H3_[m,k,0] * H3_[m,i,0] * dY[m,i,0]
                            
                    dH2[m,:,0] = dH2[m,:,0] + (H3_[m,k,0] * (W3_[m,:,k] - np.add.reduce(np.reshape(H3_[m], 10) * W3_[m], 1))) * dY[m,k,0]
                
                
            
            # 2nd hidden layer 
            dW2 = np.zeros([50, 100, 100])
            dW20 = np.zeros([50, 100, 1])
            dH1 = np.zeros([50, 100, 1])
    
            for m in range(50):     
                for k in range(100):
                    for i in range(100):
                        if i == k and Z2_[m,i,0] > 0:
                            dW2[m,:,k] = dW2[m,:,k] + H1_[m,:,0] * dH2[m,i,0]
                            dW20[m,k,0] = dW20[m,k,0] + dH2[m,i,0] 
                            
                if Z2_[m,k,0] > 0:
                    dH1[m,:,0] = dH1[m,:,0] + W2_[m,:,k] * dH2[m,k,0]
                
            
            # 1st hidden layer
            dW1 = np.zeros([50, 784, 100])
            dW10 = np.zeros([50, 100, 1])
            for m in range(50):     
                for k in range(100):
                    for i in range(100):
                        if i == k and Z1_[m,i,0] > 0:
                            dW1[m,:,k] = dW1[m,:,k] + H0_[m,:,0] * dH1[m,i,0]
                            dW10[m,k,0] = dW10[m,k,0] + dH1[m,i,0] 
            
            dW1_avg = tf.cast(tf.reduce_mean(dW1, 0), tf.float64).eval();
            #print 'dW1 is ', dW1[0]
            #print 'dW1-- is ', dW1[1]
            #print 'dW1_avg is ', dW1_avg
            dW10_avg = tf.cast(tf.reduce_mean(dW10, 0), tf.float64).eval();
            #print 'dW10_avg is ', dW10_avg 
            W1_avg = W1_avg - eta1 * dW1_avg;
            W10_avg = W10_avg - eta10 * dW10_avg;
    
            dW2_avg = tf.cast(tf.reduce_mean(dW2, 0), tf.float64).eval();
            
            dW20_avg = tf.cast(tf.reduce_mean(dW20, 0), tf.float64).eval();
            W2_avg = W2_avg - eta2 * dW2_avg;
            W20_avg = W20_avg - eta20 * dW20_avg;
    
            dW3_avg = tf.cast(tf.reduce_mean(dW3, 0), tf.float64).eval();
            #print 'dW3_avg is ', dW3_avg
            dW30_avg = tf.cast(tf.reduce_mean(dW30, 0), tf.float64).eval();
            W3_avg = W3_avg - eta3 * dW3_avg;
            W30_avg = W30_avg - eta30 * dW30_avg;
                                               
            for m in range(50):
                W1_[m] = np.copy(W1_avg)
                W10_[m] = np.copy(W10_avg)
                W2_[m] = np.copy(W2_avg)
                W20_[m] = np.copy(W20_avg)
                W3_[m] = np.copy(W3_avg)
                W30_[m] = np.copy(W30_avg) 
                         
            iter = iter + 1
            if iter == maxIter:
                break
    
        # test
        test = []
        testDataPath = '/Users/yixuantan/Documents/RPIcourses/DeepLearning2017/assignments/a3/Prog3_data/test_data'
        testLabelPath = '/Users/yixuantan/Documents/RPIcourses/DeepLearning2017/assignments/a3/Prog3_data/labels/'
        for filename in os.listdir(testDataPath):
            if not filename.endswith("jpg"): continue
            image = mpimg.imread(testDataPath + '/' + filename)
            test.append(np.reshape(image, 28 * 28))
            
        Draw = np.array(test, dtype = 'float64')
        #X_normed = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
        D = Draw / 255.0
        
        label = []
        with open(testLabelPath + '/test_label.txt') as f:
            for line in f:
                row = np.zeros(10)
                row[int(line) - 1] = 1.0
                label.append(row)
        Ytest = np.array(label, dtype = 'float64')
        
        shp = (len(D), 1)
        Xtest = np.append(D[:, :-1], np.ones(shp), 1)  
    
        # forward progagation
        H0_ = tf.reshape(Xtest, [5000, 784, 1]).eval()
    
        #Forward propagation
        Z1_ = sess.run(Z1, feed_dict = {W1 : W1_ , H0 : H0_, W10 : W10_})
        if debug: print 'Z1_ ', Z1_.shape
        H1_ = sess.run(H1, feed_dict = {Z1 : Z1_})
        Z2_ = sess.run(Z2, feed_dict = {W2 : W2_, H1: H1_, W20 : W20_})
        if debug: print 'Z2_ ', Z2_.shape
        H2_ = sess.run(H2, feed_dict = {Z2 : Z2_})
        if debug: print 'H2_ ', H2_.shape
        Z3_ = sess.run(Z3, feed_dict = {W3 : W3_, H2: H2_, W30 : W30_})
        if debug: print 'Z3_ ', Z3_.shape
        H3_ = sess.run(H3, feed_dict = {Z3 : Z3_})
        
        tp_all = 0
        for i in range(10):
            indices = []
            indices = np.argmax(Ytest, 1) == i   
            output = np.reshape(H3_,(5000,10))[indices, :]
            tp = np.sum(np.argmax(output, 1) == i)
            #if debug: print 'tp is ', tp
            tp_all = tp_all + tp
            print 'accuray for ', i + 1, ' is ', tp / output.shape[0]
    
        print 'overall accuracy ', 1.0 - 1.0 * tp_all / Xtest.shape[0] 
    
    sess.close()
    return [W1, W10, W2, W20, W3, W30]
    
def main(argv):
    
    # read trainning
    data = []
    trainDataPath = '/Users/yixuantan/Documents/RPIcourses/DeepLearning2017/assignments/a3/Prog3_data/fake_train_data'
    trainLabelPath = '/Users/yixuantan/Documents/RPIcourses/DeepLearning2017/assignments/a3/Prog3_data/labels'
    for filename in os.listdir(trainDataPath):
        if not filename.endswith("jpg"): continue
        image = mpimg.imread(trainDataPath + '/' + filename)
        data.append(np.reshape(image, 28 * 28))
        
    Draw = np.array(data, dtype = 'float64')
    #X_normed = (X - X.min(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0))
    D = Draw / 255.0
    
    label = []
    with open(trainLabelPath + '/fake_train_label.txt') as f:
        for line in f:
            row = np.zeros(10)
            row[int(line) - 1] = 1.0
            label.append(row)
    Y = np.array(label, dtype = 'float64')
    
    shp = (len(D), 1)
    X = np.append(D[:, :-1], np.ones(shp), 1)  

    Theta = train(X, Y)
    
    
if __name__ == "__main__":
    main(sys.argv)

