# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile


import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', 'test/', 'Summaries directory')

from reading_csv_files import creating_dataset
from dataset import DataSet
import pandas as pd
import numpy as np
from examination_checks import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




class configuration(object):
    
    # num_epochs = 128
    # num_ex_per_epoch = 8000
    # K = 128
    # batch_size = 258
    steps_per_epoch = 8000
    num_epochs = 200
    batch_size = 250
    dense_keep_prob = 0.5
    adv_keep_prob = 0.9
    adv_n_layers = 2
    dense_n_layers = 5
    dense_hidden_layers = [30,30,30,30,30]
    #dense_hidden_layers = [100,100,100,100,100]
    
    adv_hidden_layers = [128,128]
    adv_lambda = 0
    K = 120
    scope_adv_name = 'adv_net'
    scope_dense_name = 'dense_net'
    num_input_vab = 22
    adv_learning_rate = 0.01
    ema_decay = 0.999
    #learning_rate = 1e-5
    learning_rate = 1e-4
    epsilon = 1e-8
    mode = 'adv'
    adversary =True
    nTrainExamples = 399000
    adv_n_gaussian = 4
    num_adv_classes = 40
    val_batch_size = 100
    scope_adv_name1 = 'adv_net_1'
    scope_adv_name2 = 'adv_net_2'
    weight_decay = 1e-3

class directories(object):
    
    
    #data_train ='~/tensorflow_ex/data_csv_files/data_train.csv'
    #data_test = '~/tensorflow_ex/data_csv_files/data_test.csv'
    #data_val = '~/tensorflow_ex/data_csv_files/data_validation.csv'
    
    data_train = 'parquet_files/train_data.gzip'
    data_test = 'parquet_files/test_data.gzip'
    data_val = 'parquet_files/val_data.gzip'

    tensorboard = 'tensorboard'
    checkpoints = 'checkpoint'
    graphs = 'graphs'

def load_data_csv():
    df_test = pd.read_csv(directories.data_test)
    df_train = pd.read_csv(directories.data_train)
    df_val = pd.read_csv(directories.data_val)

    vabs_not_input = ['my_mbc', 'my_deltae','my_cc1','my_cc2', 'my_cc3', 'my_cc4',  
                      'my_cc9', 'my_hso02', 'my_hso04', 'my_hso12',
                      'my_thrustB', 'my_thrustO', 'my_cosbto','my_R2']

    features_test = df_test.drop(vabs_not_input, axis=1)
    features_train = df_train.drop(vabs_not_input, axis=1)
    features_val = df_val.drop(vabs_not_input, axis=1)
    
    labels_test = df_test['labels']
    labels_train = df_train['labels']
    labels_val = df_val['labels']
    
    aux_test = df_test[['my_mbc','my_deltae']]
    aux_train = df_train[['my_mbc','my_deltae']]
    aux_val = df_val[['my_mbc','my_deltae']]

    return [np.float32(np.nan_to_num(features_test.values)),
            np.float32(np.nan_to_num(features_train.values)),
            np.float32(np.nan_to_num(features_val.values)),
            aux_test.values.astype(np.float32),
            aux_train.values.astype(np.float32),
            aux_val.values.astype(np.float32),
            labels_test.values.astype(np.int32),
            labels_train.values.astype(np.int32),
            labels_val.values.astype(np.int32)]
            


def read_parquet_data():
    
    train_data = pd.read_parquet("parquet_files/train_data.gzip")
    test_data = pd.read_parquet("parquet_files/test_data.gzip")
    val_data = pd.read_parquet("parquet_files/val_data.gzip")


    # train_data = train_data.drop(train_data[np.absolute(train_data.deltaE) < 10.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.Mbc) < 20.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.R2) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cosTBTO) < 10.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cosTBz) < 10.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.thrustBm) < 10.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.thrustOm) < 10.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.et) < 1000.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.mm2) < 1000.0].index)

    # train_data = train_data.drop(train_data[np.absolute(train_data.hso00) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso01) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso02) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso03) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso04) < 100.0].index)
    
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso10) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso12) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso14) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso20) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso22) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hso24) < 100.0].index)


    # train_data = train_data.drop(train_data[np.absolute(train_data.hoo1) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hoo2) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hoo3) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.hoo4) < 100.0].index)

    # train_data = train_data.drop(train_data[np.absolute(train_data.cc1) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc2) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc3) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc4) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc5) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc6) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc7) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc8) < 100.0].index)
    # train_data = train_data.drop(train_data[np.absolute(train_data.cc9) < 100.0].index)
    

    training_feat1 = ['cosTBz',
                      'et',
                      'mm2',
                      'hso00',
                      'hso01',
                      'hso03',
                      'hso10',
                      'hso14',
                      'hso20',
                      'hso22',
                      'hso24',
                      'hoo0',
                      'hoo1',
                      'hoo2',
                      'hoo3',
                      'hoo4',
                      'cc5',
                      'cc6',
                      'cc7',
                      'cc8']
    
    training_feat2 = ['R2',
                      'cosTBTO',
                      'cosTBz',
                      'thrustBm',
                      'thrustOm',
                      'et',
                      'mm2',
                      'hso00',
                      'hso01',
                      'hso02',
                      'hso03',
                      'hso04',
                      'hso10',
                      'hso12',
                      'hso14',
                      'hso20',
                      'hso22',
                      'hso24',
                      'hoo0',
                      'hoo1',
                      'hoo2',
                      'hoo3',
                      'hoo4',
                      'cc1',
                      'cc2',
                      'cc3',
                      'cc4',
                      'cc5',
                      'cc6',
                      'cc7',
                      'cc8',
                      'cc9']

                                          
                     
    train_features = train_data[training_feat2]
    train_labels = train_data['label']
    train_aux = train_data[['deltaE', 'Mbc']]
    train_deltaElabels = pd.qcut(train_data['deltaE'], q=configuration.num_adv_classes, labels=False)
    train_Mbclabels = pd.qcut(train_data['Mbc'], q=configuration.num_adv_classes, labels=False)
    

    test_features = test_data[training_feat2]
    test_labels = test_data['label']
    test_aux = test_data[['deltaE', 'Mbc']]
    test_deltaElabels = pd.qcut(test_data['deltaE'], q=configuration.num_adv_classes, labels=False)
    test_Mbclabels = pd.qcut(test_data['Mbc'], q=configuration.num_adv_classes, labels=False)


    val_features = val_data[training_feat2]
    val_labels = val_data['label']
    val_aux = val_data[['deltaE', 'Mbc']]
    val_deltaElabels = pd.qcut(val_data['deltaE'], q=configuration.num_adv_classes, labels=False)
    val_Mbclabels = pd.qcut(val_data['Mbc'], q=configuration.num_adv_classes, labels=False)
    


    return [test_features.to_numpy(dtype=np.float32),
            train_features.to_numpy(dtype=np.float32),
            val_features.to_numpy(dtype=np.float32),
            test_aux.to_numpy(dtype=np.float32),
            train_aux.to_numpy(dtype=np.float32),
            val_aux.to_numpy(dtype=np.float32),
            test_labels.to_numpy(dtype=np.int32),
            train_labels.to_numpy(dtype=np.int32),
            val_labels.to_numpy(dtype=np.int32),
            train_deltaElabels.to_numpy(dtype=np.int32),
            train_Mbclabels.to_numpy(dtype=np.int32),
            test_deltaElabels.to_numpy(dtype=np.int32),
            test_Mbclabels.to_numpy(dtype=np.int32),
            val_deltaElabels.to_numpy(dtype=np.int32),
            val_Mbclabels.to_numpy(dtype=np.int32),]

            



def custom_elu(x):
    return tf.nn.elu(x) + 1.00

def log_sum_exp_trick(x, axis=1):
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    lse = x_max + tf.log(tf.reduce_sum(tf.exp(x - x_max), axis=1, keepdims=True))
    return lse


def MDN(config, x, latent_vab, 
        n_layers, hidden_nodes, 
        keep_prob, name, actvf=tf.nn.elu, 
        training=True):
    
    init = tf.contrib.layers.xavier_initializer()
    layers = [x]

    with tf.variable_scope(name, reuse=False):
        hidden_0 = dense_layer_b(x, n_nodes=hidden_nodes[0], name='hidden_0', keep_prob=keep_prob, 
                                 training=training, actvf=actvf)
        layers.append(hidden_0)
        
        for n in range(0, n_layers -1):
            hidden_n = dense_layer_b(layers[-1], n_nodes=hidden_nodes[n+1], name='hidden{}'.format(n+1), 
                                     keep_prob=keep_prob, training=training, actvf=actvf)
            layers.append(hidden_n)
        
        fc = tf.layers.dense(layers[-1], units=90, activation=tf.nn.tanh, kernel_initializer=init)
        fc_logits, fc_mu, fc_sigma = tf.split(fc, 3, axis=1)
        logits = tf.layers.dense(fc_logits, units=config.adv_n_gaussian, activation=tf.identity, name='mixing_fractions')
        centers = tf.layers.dense(fc_mu, units=config.adv_n_gaussian, activation=tf.identity, name='mean')
        variances= tf.layers.dense(fc_sigma, units=config.adv_n_gaussian, activation=tf.math.softplus, name='variances')
        #mixing_coefs = tf.nn.softmax(logits)
        
        #variances = tf.math.exp(variances)
        mixing_coefs = tf.nn.softmax(logits)
        y_true = tf.expand_dims(latent_vab,1)
        
        term1 = tf.log(mixing_coefs)
        term2 =  - 0.5*tf.log(2*np.pi)
        term3 = - tf.log(variances)
        term4 = - ((centers - y_true)*(centers - y_true))/(2*(variances)*(variances))
        exp =  term1 + term2 + term3 + term4  
        #exp = tf.reduce_sum(exp, axis=1)
        loss = tf.reduce_logsumexp(exp, axis=1, keepdims=True)
        
    #return -tf.reduce_logsumexp(exp, axis=1) 
    return loss
    
def dense_layer_b(x, n_nodes, name, keep_prob, training, actvf=tf.math.tanh ):
    bnargs = {'center':True, 'scale':True, 'training': training, 'fused': True, 'renorm':True}
    init = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(configuration.weight_decay)


    
    with tf.variable_scope(name, initializer=init) as scope:
        layer = tf.layers.dense(x, units=n_nodes, activation=actvf, 
                                kernel_initializer=init, 
                                kernel_regularizer=regularizer)
        bn = tf.layers.batch_normalization(layer, **bnargs)
        layer_out = tf.layers.dropout(bn, keep_prob, training=training)
        #layer_out = bn
    return layer_out
    

def dense_net(x, n_layers, hidden_layer_nodes, keep_prob, training, name, builder=dense_layer_b, reuse=False):
    

    init = tf.contrib.layers.xavier_initializer() 
    layers = [x]
    
    with tf.variable_scope(name, reuse=reuse):
        hidden0 = builder(x, n_nodes=hidden_layer_nodes[0], 
                          name='hidden0', 
                          keep_prob=keep_prob, 
                          training=training)
        
        layers.append(hidden0)
        
        for n in range(0, n_layers - 1):
            hidden_n = builder(layers[-1], n_nodes=hidden_layer_nodes[n + 1], name='hidden{}'.format(n+1),
                               keep_prob=keep_prob, training=training)
            layers.append(hidden_n)
    
        output = tf.layers.dropout(layers[-1], keep_prob, training=training)
        output = tf.layers.dense(output, units=1, kernel_initializer=init)
    
    return output

def residual_block(input_tensor, n_nodes, name, keep_prob, training, actvf=tf.math.tanh):
    bnargs = {'center':True, 'scale':True, 'training': training, 'fused': True, 'renorm':True}
    init = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(name, initializer=init) as scope:
        
        x = input_tensor
        x = tf.layers.dense(x, units=n_nodes, activation=actvf, kernel_initializer=init)
        x = x + input_tensor
        x = tf.layers.batch_normalization(x, **bnargs)
        x = tf.layers.dropout(x, keep_prob, training=training)
        
    return x


def residual_net(x, n_layers, hidden_layer_nodes, keep_prob, training, name, builder=residual_block, reuse=False):
    

    init = tf.contrib.layers.xavier_initializer() 
    layers = [x]
    
    with tf.variable_scope(name, reuse=reuse):
        hidden0 = dense_layer_b(x, n_nodes=hidden_layer_nodes[0], 
                                name='hidden0', 
                                keep_prob=keep_prob, 
                                training=training)
        
        layers.append(hidden0)
        
        for n in range(0, n_layers - 1):
            hidden_n = builder(layers[-1], n_nodes=hidden_layer_nodes[n + 1], name='hidden{}'.format(n+1),
                               keep_prob=keep_prob, training=training)
            layers.append(hidden_n)
    
        output = tf.layers.dropout(layers[-1], keep_prob, training=training)
        output = tf.layers.dense(output, units=1, kernel_initializer=init)
    
    return output

def adv_net_softmax_binning1(output_dense_net, latent_vab1, 
                             latent_vab1_labels,
                             n_classes,
                             n_layers, 
                             hidden_nodes, 
                             keep_prob, name,
                             builder=dense_layer_b,
                             actvf=tf.math.tanh,
                             training=True):

    init = tf.contrib.layers.xavier_initializer() 
    layers = [output_dense_net]
    
    with tf.variable_scope(name, reuse=False):
        hidden0 = builder(output_dense_net, n_nodes=hidden_nodes[0], 
                          name='advhidden0', 
                          keep_prob=keep_prob, 
                          training=training,
                          actvf=actvf)
        
        layers.append(hidden0)
        
        for n in range(0, n_layers - 1):
            hidden_n = builder(layers[-1], n_nodes=hidden_nodes[n + 1], name='advhidden{}'.format(n+1),
                               keep_prob=keep_prob, training=training, actvf=actvf)
            layers.append(hidden_n)
    
        output = tf.layers.dropout(layers[-1], keep_prob, training=training)
        output = tf.layers.dense(output, units=n_classes, kernel_initializer=init)

    
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=latent_vab1_labels, logits=output)
    
    return output

def adv_net_softmax_binning2(output_dense_net, latent_vab2, 
                             latent_vab2_labels,
                             n_classes,
                             n_layers, 
                             hidden_nodes, 
                             keep_prob, name,
                             builder=dense_layer_b,
                             actvf=tf.math.tanh,
                             training=True):

    init = tf.contrib.layers.xavier_initializer() 
    layers = [output_dense_net]
    
    with tf.variable_scope(name, reuse=False):
        hidden0 = builder(output_dense_net, n_nodes=hidden_nodes[0], 
                          name='advhidden0', 
                          keep_prob=keep_prob, 
                          training=training,
                          actvf=actvf)
        
        layers.append(hidden0)
        
        for n in range(0, n_layers - 1):
            hidden_n = builder(layers[-1], n_nodes=hidden_nodes[n + 1], name='advhidden{}'.format(n+1),
                               keep_prob=keep_prob, training=training, actvf=actvf)
            layers.append(hidden_n)
    
        output = tf.layers.dropout(layers[-1], keep_prob, training=training)
        output = tf.layers.dense(output, units=n_classes, kernel_initializer=init)

    
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=latent_vab1_labels, logits=output)
    
    return output






        
def scope_variables(name):

    with tf.variable_scope(name):
        
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=tf.get_variable_scope().name)

def mean_subtraction(X, mean):
    X -= mean
    return X
    

def scaling_Mbc(X, Mbc_min, Mbc_max):
    epsilon = 1e-15
    t_max = 1.0
    t_min = -1.0
    r_min = Mbc_min
    r_max = Mbc_max
    x = ((X - r_min)/((r_max - r_min) + epsilon))*(t_max - t_min) + t_min
    return x

def scaling_DeltaE(X, deltaE_min, deltaE_max):
    epsilon = 1e-15
    t_max = 1.0
    t_min = -1.0
    r_min = deltaE_min
    r_max = deltaE_max
    x = ((X - r_min)/((r_max - r_min) + epsilon))*(t_max - t_min) + t_min
    return x
    


def whitening(X, U, S):
    epsilon = 1e-20
    Xrot = np.dot(X,U)
    return Xrot/np.sqrt(S + epsilon)


def dataset_placeholder(features_placeholder, 
                        labels_placeholder, 
                        aux_placeholder, 
                        deltaElabels_placeholder, 
                        Mbclabels_placeholder,
                        batch_size, num_epochs, 
                        training):
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, 
                                                  labels_placeholder, 
                                                  aux_placeholder,
                                                  deltaElabels_placeholder, 
                                                  Mbclabels_placeholder))
    if training:
        dataset = dataset.shuffle(buffer_size=399000)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def datasetVal_placeholder(features_placeholder, labels_placeholder,
                           aux_placeholder,
                           deltaElabels_placeholder, 
                           Mbclabels_placeholder,
                           batchSize):
    dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, 
                                                  labels_placeholder, 
                                                  aux_placeholder, 
                                                  deltaElabels_placeholder, 
                                                  Mbclabels_placeholder))
    dataset = dataset.batch(batchSize) # 200
    dataset = dataset.repeat(2)
    return dataset

class vanillaDNN(object):
    
    def __init__(self, config,
                 featuresTrain, labelsTrain, auxTrain,
                 featuresTest, labelsTest, auxTest,
                 featuresVal, labelsVal, auxVal,
                 deltaElabelsTrain, MbclabelsTrain,
                 deltaElabelsTest, MbclabelsTest,
                 deltaElabelsVal, MbclabelsVal):

        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        
        self.featuresTrain_placeholder = tf.placeholder(np.float32, 
                                                        featuresTrain.shape)
        self.labelsTrain_placeholder = tf.placeholder(np.int32, 
                                                      labelsTrain.shape)
    
        self.auxTrain_placeholder = tf.placeholder(np.float32, 
                                                   auxTrain.shape)
        
        self.featuresTest_placeholder = tf.placeholder(np.float32,
                                                      featuresTest.shape)
    
        self.labelsTest_placeholder = tf.placeholder(np.int32,
                                                     labelsTest.shape)
        self.auxTest_placeholder = tf.placeholder(np.float32,
                                                  auxTest.shape)
        
        self.featuresVal_placeholder = tf.placeholder(np.float32,
                                                      featuresVal.shape)
    
        self.labelsVal_placeholder = tf.placeholder(np.int32, labelsVal.shape)
        self.auxVal_placeholder = tf.placeholder(np.float32,
                                                 auxVal.shape)
    
        self.deltaElabelsTrain_placeholder = tf.placeholder(np.int32, deltaElabelsTrain.shape)
        self.MbclabelsTrain_placeholder = tf.placeholder(np.int32, MbclabelsTrain.shape)
        
        self.deltaElabelsTest_placeholder = tf.placeholder(np.int32, deltaElabelsTest.shape)
        self.MbclabelsTest_placeholder = tf.placeholder(np.int32, MbclabelsTest.shape)
        
        self.deltaElabelsVal_placeholder = tf.placeholder(np.int32, deltaElabelsVal.shape)
        self.MbclabelsVal_placeholder = tf.placeholder(np.int32, MbclabelsVal.shape)
        

        trainDataset = dataset_placeholder(self.featuresTrain_placeholder, 
                                                self.labelsTrain_placeholder, 
                                                self.auxTrain_placeholder,
                                                self.deltaElabelsTrain_placeholder,
                                                self.MbclabelsTrain_placeholder,
                                                config.batch_size, 
                                                config.num_epochs, 
                                                training=True)
        
        testDataset = dataset_placeholder(self.featuresTest_placeholder, 
                                          self.labelsTest_placeholder, 
                                          self.auxTest_placeholder, 
                                          self.deltaElabelsTest_placeholder,
                                          self.MbclabelsTest_placeholder,
                                          config.batch_size, 
                                          config.num_epochs, 
                                          training=False)
        
        valDataset = datasetVal_placeholder(self.featuresVal_placeholder, 
                                            self.labelsVal_placeholder, 
                                            self.auxVal_placeholder,
                                            self.deltaElabelsVal_placeholder,
                                            self.MbclabelsVal_placeholder,
                                            config.val_batch_size)
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, trainDataset.output_types, trainDataset.output_shapes)
        self.train_iterator = trainDataset.make_initializable_iterator()
        self.test_iterator = testDataset.make_initializable_iterator()
        self.val_iterator = valDataset.make_initializable_iterator()

        self.features, self.label, self.auxiliary, self.deltaElabels, self.Mbclabels = self.iterator.get_next()
        self.dense_net_logits = residual_net(x=self.features, 
                                             n_layers=config.dense_n_layers,
                                             hidden_layer_nodes=config.dense_hidden_layers,
                                             keep_prob=config.dense_keep_prob,
                                             name=config.scope_dense_name,
                                             training=self.training_phase)
        # self.gaussian_log = MDN(config=config, x=tf.nn.softmax(self.dense_net_logits), 
        #                         latent_vab=self.auxiliary[:,1], 
        #                         n_layers=config.adv_n_layers, 
        #                         hidden_nodes=config.adv_hidden_layers,
        #                         keep_prob=config.adv_keep_prob,
        #                         name=config.scope_adv_name,
        #                         actvf=tf.nn.elu,
        #                         training=self.training_phase)
        

        
        #self.adv_loss = -tf.reduce_mean(tf.cast(self.label,tf.float32)*self.gaussian_log)
        self.adv_net_output1 = adv_net_softmax_binning1(output_dense_net=tf.nn.softmax(self.dense_net_logits),
                                                        latent_vab1=self.auxiliary[:,0],
                                                        latent_vab1_labels=self.deltaElabels,
                                                        n_classes=config.num_adv_classes,
                                                        n_layers=config.adv_n_layers,
                                                        hidden_nodes=config.adv_hidden_layers,
                                                        keep_prob=config.adv_keep_prob,
                                                        name=config.scope_adv_name1,
                                                        actvf=tf.math.tanh,
                                                        training=self.training_phase)
        self.adv_loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.deltaElabels, logits=self.adv_net_output1)
        self.adv_loss1 = tf.reduce_mean(self.adv_loss1)


        self.adv_net_output2 = adv_net_softmax_binning2(output_dense_net=tf.nn.softmax(self.dense_net_logits),
                                                        latent_vab2=self.auxiliary[:,1],
                                                        latent_vab2_labels=self.Mbclabels,
                                                        n_classes=config.num_adv_classes,
                                                        n_layers=config.adv_n_layers,
                                                        hidden_nodes=config.adv_hidden_layers,
                                                        keep_prob=config.adv_keep_prob,
                                                        name=config.scope_adv_name2,
                                                        actvf=tf.math.tanh,
                                                        training=self.training_phase)
        self.adv_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.deltaElabels, logits=self.adv_net_output2)
        self.adv_loss2 = tf.reduce_mean(self.adv_loss2)


        self.adv_loss = self.adv_loss1 + self.adv_loss2

        self.predictor_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dense_net_logits, labels=tf.expand_dims(tf.cast(self.label, tf.float32),1)))
        self.total_loss = self.predictor_loss - config.adv_lambda*self.adv_loss
        
        
        self.l2_loss = tf.losses.get_regularization_loss()
        self.predictor_loss += self.l2_loss
        

        theta_r = scope_variables(config.scope_adv_name)
        theta_f = scope_variables(config.scope_dense_name)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            
            predictor_optimizer = tf.train.AdamOptimizer(config.learning_rate)
            predictor_gs = tf.train.get_global_step()
            self.predictor_train_op = predictor_optimizer.minimize(self.predictor_loss, name='predictor_opt', 
                                                                   global_step=predictor_gs, var_list=theta_f)

    

            adversary_optimizer = tf.train.AdamOptimizer(config.adv_learning_rate)
            adversary_gs = tf.train.get_global_step()
            self.adversary_train_op = adversary_optimizer.minimize(self.adv_loss, name='adversary_opt', 
                                                                   global_step=adversary_gs, var_list=theta_r)



        self.cross_entropy = self.predictor_loss
        self.p = tf.math.sigmoid(self.dense_net_logits)
        self.transform = tf.log(self.p[:,0]/(1-self.p[:,0]+config.epsilon)+config.epsilon)
        predicted_class = tf.math.greater(self.p[:,0], 0.5)
        correct_prediction = tf.equal(predicted_class, tf.math.equal(self.label, np.int32(1.0)))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        _, self.auc_op = tf.metrics.auc(predictions = self.p, labels = tf.expand_dims(self.label,1), num_thresholds = 1024)
        self.pearson_dE, self.pearson_dE_op =  tf.contrib.metrics.streaming_pearson_correlation(predictions=self.transform,
                                                                                                labels=self.auxiliary[:,0], name='pearson_dE')
        self.pearson_mbc, self.pearson_mbc_op =  tf.contrib.metrics.streaming_pearson_correlation(predictions=self.transform,
                                                                                                  labels=self.auxiliary[:,1], name='pearson_mb\
c')
        # Add summaries                                                                                                                        
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('predictor_loss', self.predictor_loss)
        tf.summary.scalar('adversary_loss', self.adv_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('pearson_dE', self.pearson_dE_op)
        tf.summary.scalar('pearson_mbc', self.pearson_mbc_op)

        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'train_{}'.format(time.strftime('%d-%m_%I:%M'))), graph = tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, 'test_{}'.format(time.strftime('%d-%m_%I:%M'))))
        
        


        
                



def train(config, 
          featuresTrain, labelsTrain, auxTrain, 
          featuresTest, labelsTest, auxTest,
          featuresVal, labelsVal, auxVal,
          deltaElabelsTrain, MbclabelsTrain,
          deltaElabelsTest, MbclabelsTest,
          deltaElabelsVal, MbclabelsVal):
    
    VDNN = vanillaDNN(config,
                      featuresTrain, labelsTrain, auxTrain, 
                      featuresTest, labelsTest, auxTest,
                      featuresVal, labelsVal, auxVal,
                      deltaElabelsTrain, MbclabelsTrain,
                      deltaElabelsTest, MbclabelsTest,
                      deltaElabelsVal, MbclabelsVal)
    start_time = time.time()
    global_step, epochs = 0, 0
    saver = tf.train.Saver()
    #ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    
    with tf.Session() as sess:
        init_op_glob = tf.global_variables_initializer()
        sess.run(init_op_glob)
        init_op_loc = tf.local_variables_initializer()
        sess.run(init_op_loc)
        
        

        train_handle = sess.run(VDNN.train_iterator.string_handle())
        test_handle = sess.run(VDNN.test_iterator.string_handle())
        
        

        sess.run(VDNN.train_iterator.initializer, feed_dict={VDNN.featuresTrain_placeholder: featuresTrain,
                                                             VDNN.labelsTrain_placeholder: labelsTrain,
                                                             VDNN.auxTrain_placeholder: auxTrain,
                                                             VDNN.deltaElabelsTrain_placeholder: deltaElabelsTrain,
                                                             VDNN.MbclabelsTrain_placeholder: MbclabelsTrain})
        sess.run(VDNN.test_iterator.initializer, feed_dict={VDNN.featuresTest_placeholder: featuresTest,
                                                            VDNN.labelsTest_placeholder: labelsTest,
                                                            VDNN.auxTest_placeholder: auxTest,
                                                            VDNN.deltaElabelsTest_placeholder: deltaElabelsTest,
                                                            VDNN.MbclabelsTest_placeholder: MbclabelsTest})
        

        # feed_dict_train = {VDNN.training_phase:True, VDNN.handle:train_handle}

        # feed_dict_test = {VDNN.training_phase:False, VDNN.handle:test_handle}

        # feed_dict_val = {VDNN.training_phase:False, VDNN.handle:val_handle}


        while True:
            try:
                if config.adversary:
                    if global_step % config.K == 0:
                        sess.run(VDNN.predictor_train_op, feed_dict={VDNN.training_phase:True, VDNN.handle:train_handle})
                    else:
                        sess.run(VDNN.adversary_train_op, feed_dict={VDNN.training_phase:True, VDNN.handle:train_handle})
                        
                    global_step +=1

                    if global_step % (config.steps_per_epoch) == 0:
        

                        tensorboard_summary(model=VDNN, sess=sess, 
                                            train_handle=train_handle, 
                                            test_handle=test_handle, 
                                            epochs=epochs)
                        
                else:
                    # Run X steps on training dataset                                                                                          
                    sess.run(VDNN.predictor_train_op, feed_dict={VDNN.training_phase: True, 
                                                                 VDNN.handle: train_handle})
                    global_step +=1

                    if global_step % (config.steps_per_epoch) == 0:
          
                        tensorboard_summary(model=VDNN, sess=sess, 
                                            train_handle=train_handle, 
                                            test_handle=test_handle, 
                                            epochs=epochs)


                        epochs += 1
            except tf.errors.OutOfRangeError:
                break

        VDNN.test_writer.flush()
        VDNN.train_writer.flush()
        VDNN.test_writer.close()
        VDNN.train_writer.close()

        # final_validation(model=VDNN, config=config, 
        #                  sess=sess, saver=saver,
        #                  val_handle,
        #                  directories=directories,
        #                  global_step=global_step,
        #                  start_time=start_time)
        
        final_validation2(model=VDNN, config=config, 
                          sess=sess, saver=saver,
                          directories=directories,
                          global_step=global_step,
                          start_time=start_time,
                          featuresTrain=featuresTrain,
                          labelsTrain=labelsTrain, auxTrain=auxTrain,
                          featuresTest=featuresTest,
                          labelsTest=labelsTest,
                          auxTest=auxTest,
                          featuresVal=featuresVal, 
                          labelsVal=labelsVal, auxVal=auxVal,
                          deltaElabelsTrain=deltaElabelsTrain, MbclabelsTrain=MbclabelsTrain,
                          deltaElabelsTest=deltaElabelsTest, MbclabelsTest=MbclabelsTest,
                          deltaElabelsVal=deltaElabelsVal, MbclabelsVal=MbclabelsVal)
        
    
        # gaussian, term1, term2 = sess.run([VDNN.gaussian_log, VDNN.term1, VDNN.term2],
        #                                   feed_dict= {VDNN.training_phase:False, VDNN.handle:val_handle})
        

        # print("gaussian_log \n")
        # print(gaussian)
        # print("\n")
        # print("coefficients \n")
        # print(term1)
        # print("\n")
        # print("variances \n")
        # print(term2)


def predict_ckpFile(config, ckp_dir,
                    featuresTrain, labelsTrain, auxTrain,
                    featuresTest, labelsTest, auxTest,
                    featuresVal, labelsVal, auxVal,
                    train_deltaElabels, train_Mbclabels,
                    test_deltaElabels, test_Mbclabels,
                    val_deltaElabels, val_Mbclabels):
    
    VDNN = vanillaDNN(config,
                      featuresTrain, labelsTrain, auxTrain, 
                      featuresTest, labelsTest, auxTest,
                      featuresVal, labelsVal, auxVal,
                      deltaElabelsTrain, MbclabelsTrain,
                      deltaElabelsTest, MbclabelsTest,
                      deltaElabelsVal, MbclabelsVal)
    
    ckpt = tf.train.get_checkpoint_state(ckp_dir)
    saver = tf.train.Saver()
    direct = directories()
    start_time = time.time()

    with tf.Session() as sess:
        init_op_glob = tf.global_variables_initializer()
        sess.run(init_op_glob)
        init_op_loc = tf.local_variables_initializer()
        sess.run(init_op_loc)
        assert (ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('{} restored.'.format(ckpt.model_checkpoint_path))


        train_handle = sess.run(VDNN.train_iterator.string_handle())
        test_handle = sess.run(VDNN.test_iterator.string_handle())
        val_handle = sess.run(VDNN.val_iterator.string_handle())
        
        
        
        sess.run(VDNN.train_iterator.initializer, 
                 feed_dict={VDNN.featuresTrain_placeholder: featuresTrain,
                            VDNN.labelsTrain_placeholder: labelsTrain, 
                            VDNN.auxTrain_placeholder: auxTrain,
                            VDNN.deltaElabelsTrain_placeholder: train_deltaElabels,
                            VDNN.MbclabelsTrain_placeholder: train_Mbclabels})
        sess.run(VDNN.test_iterator.initializer, 
                 feed_dict={VDNN.featuresTest_placeholder: featuresTest,
                            VDNN.labelsTest_placeholder: labelsTest,
                            VDNN.auxTest_placeholder: auxTest,
                            VDNN.deltaElabelsTest_placeholder: test_deltaElabels,
                            VDNN.MbclabelsTest_placeholder: test_Mbclabels})
        sess.run(VDNN.val_iterator.initializer, 
                 feed_dict={VDNN.featuresVal_placeholder: featuresVal,
                            VDNN.labelsVal_placeholder: labelsVal,
                            VDNN.auxVal_placeholder: auxVal,
                            VDNN.deltaElabelsVal_placeholder: val_deltaElabels,
                            VDNN.MbclabelsVal_placeholder: val_Mbclabels})
    

        final_validation3(VDNN, config, sess, saver, 
                          direct, start_time,
                          train_handle, test_handle, val_handle)
        
    
    
        print(tf.VERSION)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict', help = 'Run inference', action = 'store_true')
    args = parser.parse_args()




    
    [featuresTrain, featuresTest, featuresVal, auxTrain, auxTest, auxVal,labelsTrain, labelsTest, labelsVal, train_deltaElabels, train_Mbclabels, test_deltaElabels, test_Mbclabels, val_deltaElabels, val_Mbclabels] = read_parquet_data()
    

    #performing preprocessing steps
    

    # finding std
    
    # featuresTrain = featuresTrain[:2,:2]
    
    
    # print(featuresTrain)
    # featuresTrain_mean = np.mean(featuresTrain, axis=0)
    # featuresTrain_std = np.std(featuresTrain, axis=0)
    
    # print(featuresTrain_mean)
    

    # # # # mean subtraction
    # featuresTrain -= np.mean(featuresTrain, axis=0)

    # print(featuresTrain)

    # # # # normalization
    # epsilon = 1e-20
     
    # # # featuresTrain = featuresTrain/(featuresTrain_std + epsilon)
    
    # # # # decorrelation
    # training_cov = np.dot(featuresTrain.T, featuresTrain)/featuresTrain.shape[0]
    
    # train_U, train_S, train_V = np.linalg.svd(training_cov)
    
    # featuresTrain_rot = np.dot(featuresTrain, train_U)
    
    # print(featuresTrain)

    # # # # whitening
    # featuresTrain = featuresTrain_rot/np.sqrt(train_S + epsilon)

    # norm = StandardScaler().fit(featuresTrain)
    # featuresTrain = norm.transform(featuresTrain)
    # pca = PCA(n_components=20, whiten=True)

    # pca.fit(featuresTrain)
    
    # X_pca = pca.transform(featuresTrain)

    # featuresTrain = X_pca

    


    config = configuration()

    training = True

    checkpoint_dir = 'check_points'

    print("Start training densenet")


    if args.predict:
        
        predict_ckpFile(config, checkpoint_dir,
                        featuresTrain, labelsTrain, auxTrain,
                        featuresTest, labelsTest, auxTest, 
                        featuresVal, labelsVal, auxVal,
                        train_deltaElabels, train_Mbclabels,
                        test_deltaElabels, test_Mbclabels,
                        val_deltaElabels, val_Mbclabels)
    
    else:
        
        train(config, 
              featuresTrain, labelsTrain, auxTrain,
              featuresTest, labelsTest, auxTest, 
              featuresVal, labelsVal, auxVal,
              train_deltaElabels, train_Mbclabels,
              test_deltaElabels, test_Mbclabels,
              val_deltaElabels, val_Mbclabels)
    
