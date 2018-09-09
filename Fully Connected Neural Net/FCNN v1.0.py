#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:52:49 2018

@author: abhigyan
"""

"""This is an implementation of a Fully Connected Neural Network Architecture 
    with tf.contrib.layers API. Some non-mandatory parameters to the tf.contrib.layers
    is missing. This is only a Vanilla implementation. The user can easily add
    the missing customisations"""


#Importing Relevant packages
    
import tensorflow as tf


"""Neural Network Class which will be applied on a given problem, in nodes input
the Neural Network structure, with 'D' standing for dropout
Example: nn = Neural_Network(4, [10, 'D', 10, 3], [tf.nn.leaky_relu, 'D', tf.nn.leaky_relu, tf.nn.leaky_relu], 0.1, optimiser = 'Adam')
         nn.train_Neural_Net(dataset, labels, 2000)    
         nn.evaluate_Neural_Network(dataset, labels)"""

class Neural_Network(object):
    
    def __init__(self, numberOfLayers, nodes, activations, learningRate,  
                 optimiser = 'GradientDescent', regularizer = None, 
                 dropout = 0.5, initializer = tf.contrib.layers.xavier_initializer(),
                 featureNormalize = True):
        
        #Few parameters
        self.numberOfLayers = numberOfLayers
        self.nodes = nodes
        self.activations = activations
        self.learningRate = learningRate
        self.regularizer = regularizer
        self.dropout = dropout
        self.initializer = initializer
        self.featureNormalize = featureNormalize
        
        #Different optimisers. Required optimisers may be added by the user
        if(optimiser == 'GradientDescent'):
            self.optimiser = tf.train.GradientDescentOptimizer(self.learningRate)
        elif(optimiser == 'Adam'):
            self.optimiser = tf.train.AdamOptimizer(self.learningRate)
        elif(optimiser == 'RMSProp'):
            self.optimiser = tf.train.RMSPropOptimizer(self.learningRate)
        elif(optimiser == 'AdaGrad'):
            self.optimiser = tf.train.AdagradOptimizer(self.learningRate)
        elif(optimiser == 'AdaGrad'):
            self.optimiser = tf.train.MomentumOptimizer(self.learningRate)

        
        
    #Creating the computational grpah of the Neural Network    
    def create_Neural_Net(self, numberOfFeatures):
        self.numberOfFeatures = numberOfFeatures
        self.X = tf.placeholder(dtype = tf.float32, shape = (None, self.numberOfFeatures), name = 'Input_Dataset')
        
        for i in range(0, self.numberOfLayers):
            if(i == 0 and self.nodes[i]!= 'D'):
                layer = tf.contrib.layers.fully_connected(self.X, self.nodes[i], 
                                                          activation_fn = self.activations[i],
                                                          weights_initializer = self.initializer, 
                                                          biases_initializer = self.initializer)  
                
            elif(i == self.numberOfLayers-1 and self.nodes[i]!= 'D'):
                self.output = tf.contrib.layers.fully_connected(layer, self.nodes[i], 
                                                                activation_fn = self.activations[i],
                                                                weights_initializer = self.initializer, 
                                                                biases_initializer = self.initializer)
            elif(self.nodes[i]!= 'D'):
                layer = tf.contrib.layers.fully_connected(layer, self.nodes[i], 
                                                          activation_fn = self.activations[i],
                                                          weights_initializer = self.initializer, 
                                                          biases_initializer = self.initializer)
                
            elif(self.nodes[i] == 'D' and self.mode == 'TRAIN'):
                tf.nn.dropout(layer, self.dropout)
                
            elif(self.nodes[i] == 'D' and (self.mode == 'EVALUATE' or self.mode == 'PREDICT')):
                layer = layer * self.dropout
                
    
    #Method to normalise the dataset, if unnormalised.
    def featureNormalizer(self, dataset):
        mean = np.mean(dataset, axis = 0)
        stddev = np.sqrt(np.var(dataset, axis = 0))
        return (dataset - mean)/stddev
                
    
    #Method to train the Neural Network. Takes in dataset, labels and number of epochs
    def train_Neural_Net(self, dataset, labels, epochs):
        if(self.featureNormalize == True):
            dataset = self.featureNormalizer(dataset)
        
        self.mode = 'TRAIN'
        self.create_Neural_Net(numberOfFeatures = len(dataset[0]))
            
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.output, labels = labels, name = 'cross_entropy')
        loss = tf.reduce_mean(entropy, name = 'loss')
        hypothesis = tf.nn.softmax(self.output)
        correct_preds = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
        train_op = self.optimiser.minimize(loss)
        
        self.loss=[]
        self.accuracy = []
        self.saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(0, epochs):
                _, l, acc = sess.run([train_op, loss, accuracy], feed_dict = {self.X:dataset}) 
                print('Loss in epoch ' + str(i) + ' is: ' + str(l))
                self.loss.append(l)
                self.accuracy.append(acc)
            self.saver.save(sess, './Neural_Net.ckpt')          #Saves model at the current directory
                                                                #with the name Neural_Net.ckpt
        return self.loss, self.accuracy
    
    
    #Method to Cross Validate the Neural Network. Takes in dataset, labels
    def evaluate_Neural_Network(self, dataset, labels):
        if(self.featureNormalize == True):
            dataset = self.featureNormalizer(dataset)
            
        self.mode = 'EVALUATE'
        
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.output, labels = labels, name = 'cross_entropy')
        loss = tf.reduce_mean(entropy, name = 'loss')
        hypothesis = tf.nn.softmax(self.output)
        correct_preds = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
        
        with tf.Session() as sess:
            self.saver.restore(sess, './Neural_Net.ckpt')
            l, acc = sess.run([loss, accuracy], feed_dict = {self.X:dataset})  
            self.loss = l
            self.accuracy = acc
                
        return self.loss, self.accuracy
    
    
    
    #Method to make predictions based on trained weights
    def predict_Neural_Network(self, dataset):
        if(self.featureNormalize == True):
            dataset = self.featureNormalizer(dataset)
            
        self.mode = 'PREDICT'
            
        hypothesis = tf.nn.softmax(self.output)
        predictions = tf.argmax(hypothesis, 1)
        
        with tf.Session() as sess:
            self.predictions = sess.run([predictions], feed_dict = {self.X:dataset})  
            
        return self.predictions
    


"""END"""    
    
