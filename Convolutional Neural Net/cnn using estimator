#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:17:10 2018

@author: abhigyan
"""
import tensorflow as tf


class CNN(object):
    
    def __init__(self, numberOfLayers, typeOfLayer, layerActivations, filtersInEachLayer, 
                 kernelSizeInEachLayer, stridesInEachLayer, paddingInEachLayer):
        
        self.numberOfLayers = numberOfLayers
        self.typeOfLayer = typeOfLayer
        self.layerActivations = layerActivations
        self.filtersInEachLayer = filtersInEachLayer
        self.kernelSizeInEachLayer = kernelSizeInEachLayer
        self.stridesInEachLayer = stridesInEachLayer
        self.paddingInEachLayer = paddingInEachLayer
        
        
    def create_a_CNN(self, features, labels, mode, params):
        self.X = features
        self.labels = labels
        
        for i in range(0, self.numberOfLayers):
            if(self.typeOfLayer[i] == 'conv' and i == 0):
                output = tf.layers.conv2d(self.X, filters = self.filtersInEachLayer[i], 
                                     kernel_size = self.kernelSizeInEachLayer[i], 
                                     strides = self.stridesInEachLayer[i], 
                                     padding = self.paddingInEachLayer[i], 
                                     activation = self.layerActivations[i])
                
            elif(self.typeOfLayer[i] == 'conv' and i != self.numberOfLayers-1):
                output = tf.layers.conv2d(output, filters = self.filtersInEachLayer[i], 
                                     kernel_size = self.kernelSizeInEachLayer[i], 
                                     strides = self.stridesInEachLayer[i], 
                                     padding = self.paddingInEachLayer[i], 
                                     activation = self.layerActivations[i])
                
            elif(self.typeOfLayer[i] == 'conv' and i == self.numberOfLayers-1):
                self.output = tf.layers.conv2d(output, filters = self.filtersInEachLayer[i], 
                                     kernel_size = self.kernelSizeInEachLayer[i], 
                                     strides = self.stridesInEachLayer[i], 
                                     padding = self.paddingInEachLayer[i], 
                                     activation = self.layerActivations[i])
                
            elif(self.typeOfLayer[i] == 'pool' and i != self.numberOfLayers-1):
                output = tf.layers.max_pooling2d(inputs = output, 
                                                 pool_size = self.kernelSizeInEachLayer[i],
                                                 strides = self.stridesInEachLayer[i])
                
            elif(self.typeOfLayer[i] == 'pool' and i == self.numberOfLayers-1):
                self.output = tf.layers.max_pooling2d(inputs = output, 
                                                      pool_size=self.kernelSizeInEachLayer[i], 
                                                      strides = self.stridesInEachLayer[i])
                
            
            elif(self.typeOfLayer[i] == 'fully_connected' and i != self.numberOfLayers-1):
                output = tf.contrib.layers.flatten(output)
                output = tf.contrib.layers.fully_connected(output, 
                                                       num_outputs = self.filtersInEachLayer[i],
                                                       activation_fn = self.layerActivations[i])
                
            elif(self.typeOfLayer[i] == 'fully_connected' and i == self.numberOfLayers-1):
                output = tf.contrib.layers.flatten(output)
                self.output = tf.contrib.layers.fully_connected(output, 
                                                       num_outputs = self.filtersInEachLayer[i],
                                                       activation_fn = self.layerActivations[i])

                                                         
        self.hypothesis = tf.nn.softmax(self.output)
        self.y_pred_class = tf.argmax(self.hypothesis, axis = 1)
        #self.sparseLabels = tf.argmax(labels, axis = 1)
        
    
        if(mode == tf.estimator.ModeKeys.PREDICT):
            spec = tf.estimator.EstimatorSpec(mode = mode, predictions = self.y_pred_class)
        
        else:
            self.entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.output, 
                                                                  labels = self.labels, 
                                                                  name = 'cross_entropy')
            self.loss = tf.reduce_mean(self.entropy, name = 'loss')
            self.optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
            train_op = self.optimizer.minimize(loss = self.loss, global_step=tf.train.get_global_step())
            metrics = {'accuracy':tf.metrics.accuracy(labels = self.labels, predictions = self.y_pred_class)}
            spec = tf.estimator.EstimatorSpec(mode = mode, predictions = self.y_pred_class, 
                                              loss = self.loss, train_op = train_op, eval_metric_ops = metrics)
            print(spec)
    
        return spec
    
    
    def create_estimators_CNN(self, dataset, labels, CVDataset, CVLabels, parameters):
        self.dataset = dataset
        self.labels = labels
        self.CVDataset = CVDataset
        self.CVLabels = CVLabels
        self.parameters = parameters
        #self.create_a_CNN(self.dataset, self.labels, )
        
        self.train_input_fn = tf.estimator.inputs.numpy_input_fn(x = self.dataset, y = self.labels, 
                                                                 num_epochs = None, 
                                                                 shuffle = True)
        self.eval_input_fn = tf.estimator.inputs.numpy_input_fn(x = self.dataset, y = self.labels, 
                                                                num_epochs = 1, 
                                                                shuffle = False)
        self.predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = self.dataset, y = self.labels, 
                                                                num_epochs = 1, 
                                                                shuffle = False)
        self.model = tf.estimator.Estimator(model_fn = self.create_a_CNN, params = self.parameters, 
                                            model_dir = './cnn_checkpoint')
        
    def use_CNN(self, mode):
        
        if(mode == 'Train'):
            self.train = self.model.train(input_fn = self.train_input_fn, steps = 1000)
            print(self.train)
        elif(mode == 'Evaluate'):
            self.evaluate = self.model.evaluate(input_fn = self.eval_input_fn)
        elif(mode == 'Predict'):
            self.predict = self.model.predict(input_fn = self.predict_input_fn)
