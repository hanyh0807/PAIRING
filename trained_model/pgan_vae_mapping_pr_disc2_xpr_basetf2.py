# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 21:30:01 2021

@author: Younghyun Han

#base code from https://github.com/welch-lab/MichiGAN/blob/ee28a6ad930c601e83cc123d00a3e64d9f2944a0/models/gan.py#L20
"""

import os
import sys
import gc

import numpy as np
import numba as nb
from scipy.spatial.distance import cdist
import pandas as pd
from time import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

from Adam_prediction import Adam_Prediction_Optimizer
from metrics.GenerationMetrics import *
import progeny

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

style = True

class PGAN:
    def __init__(self, x_dim, **kwargs):
        tf.compat.v1.reset_default_graph()
        self.x_dim = x_dim
        self.z_dim = kwargs.get("z_dim", 512)
        self.learning_rate = kwargs.get("learning_rate", 5e-4)
        self.dropout_rate = kwargs.get("dropout_rate", 0.2)
        self.lamb_gp = kwargs.get("lamb_gp", 10.0)
        self.lamb_recon = kwargs.get("lamb_recon", 1.0)
        self.lamb_triple = kwargs.get("lamb_triple", 1.0)
        self.lamb_delta = kwargs.get("lamb_delta", 1.0)
        self.Diters = kwargs.get('Diters', 5)
        self.enc_unit1 = kwargs.get("enc_unit1", 512)
        self.enc_unit2 = kwargs.get("enc_unit2", 512)
        self.enc_unit3 = kwargs.get("enc_unit3", 512)
        self.gen_unit1 = kwargs.get("gen_unit1", 256)
        self.gen_unit2 = kwargs.get("gen_unit2", 512)
        self.gen_unit3 = kwargs.get("gen_unit3", 1024)
        self.disc_unit1 = kwargs.get("disc_unit1", 1024)
        self.disc_unit2 = kwargs.get("disc_unit2", 512)
        self.disc_unit3 = kwargs.get("disc_unit3", 10)
        self.init_w = tf.compat.v1.keras.initializers.glorot_normal()
        self.regu_w = tf.compat.v1.keras.regularizers.l2(5e-4)

        with tf.device('/gpu:0'):
            self.is_training = tf.compat.v1.placeholder(tf.bool, name = "training_flag")
            self.batch_size = tf.compat.v1.placeholder(tf.int32, shape = [], name="batch_size")
            self.t_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="t_term1")
            self.t_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="t_term2")
            self.t_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="t_term3")
            self.t_term4 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="t_term4")

            self.d_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="d_term1")
            self.d_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="d_term2")
            self.d_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="d_term3")
            self.d_term4 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="d_term4")

            self.tc_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="tc_term1")
            self.tc_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="tc_term2")
            self.tc_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="tc_term3")
            self.tc_term4 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="tc_term4")

            self.dc_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dc_term1")
            self.dc_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dc_term2")
            self.dc_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dc_term3")
            self.dc_term4 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dc_term4")

            self.du_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="du_term1")
            self.du_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="du_term2")
            self.du_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="du_term3")
            self.du_term4 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="du_term4")
            
            self.dd_term1 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dd_term1")
            self.dd_term2 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dd_term2")
            self.dd_term3 = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="dd_term3")

            self.no_term = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="no_term")
            
            self.x = tf.concat([self.t_term1, self.t_term2, self.t_term3, self.t_term4,
                                self.d_term1, self.d_term3,
                                self.no_term], axis=0, name="data")
                                #self.d_term1, self.d_term2, self.d_term3, self.d_term4,
                                #self.no_term], axis=0, name="data")
            self.x_mmd = tf.concat([self.t_term1, self.t_term2, self.t_term3, self.t_term4,
                                    self.d_term1, self.d_term3], axis=0, name="data_mmd")
            '''
            self.x_aug = tf.concat([self.tc_term1, self.tc_term2, self.tc_term3, self.tc_term4,
                                self.dc_term1, self.dc_term2, self.dc_term3, self.dc_term4,
                                self.no_term], axis=0, name="data")
            '''
            self.z = tf.compat.v1.placeholder(tf.float32, shape = [None, self.z_dim], name="latent")
            
            self.x_bulk = tf.compat.v1.placeholder(tf.float32, shape = [None, self.x_dim], name="x_bulk")
            self.gennum = tf.compat.v1.placeholder(tf.int32, shape = [], name="gennum")

            self.layernorm1 = tf.keras.layers.LayerNormalization(name='LayerNorm')
            self.layernorm2 = tf.keras.layers.LayerNormalization(name='LayerNorm_1')

            self.create_network()
            self.loss_function()

    def encoder(self, x_input):
        """
        TODO : convert to variational?? (to force en_output ~ N(0,1))
        """
        with tf.compat.v1.variable_scope('encoder', reuse=tf.compat.v1.AUTO_REUSE):
            
            en_dense1 = tf.compat.v1.layers.dense(inputs=x_input, units=self.enc_unit1, activation=None, kernel_regularizer=self.regu_w)
            # en_dense1 = tf.layers.batch_normalization(en_dense1, training=self.is_training)
            en_dense1 = self.layernorm1(en_dense1)
            en_dense1 = tf.nn.leaky_relu(en_dense1)
            en_dense1 = tf.compat.v1.layers.dropout(en_dense1, self.dropout_rate, training=self.is_training)

            en_dense2 = tf.compat.v1.layers.dense(inputs=en_dense1, units=self.enc_unit2, activation=None, kernel_regularizer=self.regu_w)
            # en_dense2 = tf.layers.batch_normalization(en_dense2, training=self.is_training)
            en_dense2 = self.layernorm2(en_dense2)
            en_dense2 = tf.nn.leaky_relu(en_dense2)
            en_dense2 = tf.compat.v1.layers.dropout(en_dense2, self.dropout_rate, training=self.is_training)

            '''
            en_dense3 = tf.layers.dense(inputs=en_dense2, units=self.enc_unit3, activation=None)
            en_dense3 = tf.layers.batch_normalization(en_dense3, training=self.is_training)
            en_dense3 = tf.nn.leaky_relu(en_dense3)
            # en_dense3 = tf.layers.dropout(en_dense3, self.dropout_rate, training=self.is_training)
            '''
            
            en_output_mu = tf.compat.v1.layers.dense(inputs=en_dense2, units=self.z_dim, activation=None)
            en_output_sig = tf.compat.v1.layers.dense(inputs=en_dense2, units=self.z_dim, activation=tf.nn.softplus) + 1e-6
            
            encoded = en_output_mu + en_output_sig * tf.random.normal(shape=[tf.shape(input=en_output_mu)[0], self.z_dim])
            
            
            return en_output_mu, en_output_sig, encoded

    def mapper(self, x_input):
        with tf.compat.v1.variable_scope('mapper', reuse=tf.compat.v1.AUTO_REUSE):
            '''
            nlayers = 3
            mapping = x_input
            for _ in range(nlayers):
                with tf.compat.v1.variable_scope('layer_'+str(_), reuse=tf.compat.v1.AUTO_REUSE):
                    mapping = tf.compat.v1.layers.dense(inputs=mapping, units=self.z_dim, activation=None, kernel_regularizer=self.regu_w)
                    # mapping = tf.layers.batch_normalization(mapping, training=self.is_training)
                    # mapping = tf.contrib.layers.layer_norm(mapping)
                    mapping = tf.nn.leaky_relu(mapping)
                    # if _ != (nlayers-1):
                    #     mapping = tf.nn.leaky_relu(mapping)
                    #     mapping = tf.layers.dropout(mapping, self.dropout_rate, training=self.is_training)
            
            return mapping
            '''
            return tf.identity(x_input)
        
    def generatorDropOut(self, z_input, return_dlatents=False):
        """
        generator with dropout layers of WGAN-GP
        """
        with tf.compat.v1.variable_scope('generatorDropOut', reuse=tf.compat.v1.AUTO_REUSE):

            if style:
                def applyNoise(x):
                    with tf.compat.v1.variable_scope('noise'):
                        noise = tf.random.normal([tf.shape(input=x)[0],tf.shape(input=x)[1]])
                        weight = tf.compat.v1.get_variable('weight', shape=[1,1], initializer=tf.compat.v1.initializers.zeros(), trainable=True)
                        return x + noise * tf.cast(weight, x.dtype)
                    
                x = tf.compat.v1.get_variable('const', shape=[1, self.gen_unit1], initializer=tf.compat.v1.initializers.random_normal(), trainable=True)
                x_bias = tf.compat.v1.get_variable('const_bias', shape=[1, self.gen_unit1], initializer=tf.compat.v1.constant_initializer(0.0), trainable=True)
                #x = (x - tf.reduce_mean(input_tensor=x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-6)
                x = tf.nn.leaky_relu(x + x_bias)
                #x = (x - tf.reduce_mean(x, axis=-1, keepdims=True)) / (tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-6)
                x = tf.tile(x, [tf.shape(input=z_input)[0],1])
                
                def applyStyle(inputs, style, units, firstdense=True):
                    outputs = inputs
                    if firstdense:
                        outputs = tf.compat.v1.layers.dense(inputs=outputs, units=units, activation=None, kernel_regularizer=self.regu_w)
                        #outputs = applyNoise(outputs)
                        #outputs = (outputs - tf.reduce_mean(input_tensor=outputs, axis=-1, keepdims=True)) / (tf.math.reduce_std(outputs, axis=-1, keepdims=True) + 1e-6)
                        outputs = tf.nn.leaky_relu(outputs)
                        #outputs = (outputs - tf.reduce_mean(outputs, axis=-1, keepdims=True)) / (tf.math.reduce_std(outputs, axis=-1, keepdims=True) + 1e-6)
                    s = tf.compat.v1.layers.dense(inputs=style, units=units, activation=None, kernel_regularizer=self.regu_w)
                    sw = tf.compat.v1.layers.dense(inputs=style, units=units, activation=None, kernel_regularizer=self.regu_w)
                    outputs = sw * outputs + s
                    
                    # outputs = tf.layers.dense(inputs=outputs, units=units, activation=tf.nn.leaky_relu)
                    # outputs = (outputs - tf.reduce_mean(outputs, axis=-1, keepdims=True)) / (tf.math.reduce_std(outputs, axis=-1, keepdims=True) + 1e-6)
                    # s_ = tf.layers.dense(inputs=style, units=units, activation=None)
                    # outputs = outputs + s_
                    
                    return outputs
                        
                z_input_broadcast = tf.tile(z_input[:,tf.newaxis], [1,4,1])
                with tf.compat.v1.variable_scope('layer1', reuse=tf.compat.v1.AUTO_REUSE):
                    ge_dense1 = applyStyle(x, z_input_broadcast[:,0], self.gen_unit1, firstdense=False)
                
                with tf.compat.v1.variable_scope('layer2', reuse=tf.compat.v1.AUTO_REUSE):
                    ge_dense2 = applyStyle(ge_dense1, z_input_broadcast[:,1], self.gen_unit2)
                    
                with tf.compat.v1.variable_scope('layer3', reuse=tf.compat.v1.AUTO_REUSE):
                    ge_dense3 = applyStyle(ge_dense2, z_input_broadcast[:,2], self.gen_unit3)
                
                with tf.compat.v1.variable_scope('layer4', reuse=tf.compat.v1.AUTO_REUSE):
                    ge_dense3 = applyStyle(ge_dense3, z_input_broadcast[:,3], self.gen_unit3)

            else:
                ge_dense1 = tf.compat.v1.layers.dense(inputs=z_input, units=self.gen_unit1, activation=None)
                # ge_dense1 = tf.layers.batch_normalization(ge_dense1, training=self.is_training)
                ge_dense1 = tf.keras.layers.LayerNormalization()(ge_dense1)
                ge_dense1 = tf.nn.leaky_relu(ge_dense1)
                ge_dense1 = tf.compat.v1.layers.dropout(ge_dense1, self.dropout_rate, training=self.is_training)
    
                ge_dense2 = tf.compat.v1.layers.dense(inputs=ge_dense1, units=self.gen_unit2, activation=None)
                # ge_dense2 = tf.layers.batch_normalization(ge_dense2, training=self.is_training)
                ge_dense2 = tf.keras.layers.LayerNormalization()(ge_dense2)
                ge_dense2 = tf.nn.leaky_relu(ge_dense2)
                ge_dense2 = tf.compat.v1.layers.dropout(ge_dense2, self.dropout_rate, training=self.is_training)
                
                ge_res1 = tf.compat.v1.layers.dense(inputs=ge_dense1, units=self.gen_unit2, activation=None)
                
                ge_dense3 = tf.compat.v1.layers.dense(inputs=ge_dense2+ge_res1, units=self.gen_unit3, activation=None)
                # ge_dense3 = tf.layers.batch_normalization(ge_dense3, training=self.is_training)
                ge_dense3 = tf.keras.layers.LayerNormalization()(ge_dense3)
                ge_dense3 = tf.nn.leaky_relu(ge_dense3)
                ge_dense3 = tf.compat.v1.layers.dropout(ge_dense3, self.dropout_rate, training=self.is_training)
                
                # ge_res2 = tf.layers.dense(inputs=ge_dense1, units=self.gen_unit3, activation=None)
                ge_res3 = tf.compat.v1.layers.dense(inputs=ge_dense2, units=self.gen_unit3, activation=None)
                ge_dense3 = ge_dense3 + ge_res3

            ge_output = tf.compat.v1.layers.dense(inputs=ge_dense3, units=self.x_dim, activation=None)

            if return_dlatents:
                return ge_output, z_input_broadcast
            return ge_output

    def discriminator(self, x_input):
        """
        discriminator of WGAN-GP
        """
        with tf.compat.v1.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):
            disc_dense1 = tf.compat.v1.layers.dense(inputs=x_input, units=self.disc_unit1, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense1 = tf.layers.batch_normalization(disc_dense1, training=self.is_training)
            disc_dense1 = tf.nn.leaky_relu(disc_dense1)

            disc_dense2 = tf.compat.v1.layers.dense(inputs=disc_dense1, units=self.disc_unit2, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense2 = tf.layers.batch_normalization(disc_dense2, training=self.is_training)
            disc_dense2 = tf.nn.leaky_relu(disc_dense2)

            disc_dense3_ = tf.compat.v1.layers.dense(inputs=disc_dense2, units=self.disc_unit3, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense3 = tf.layers.batch_normalization(disc_dense3, training=self.is_training)
            disc_dense3 = tf.nn.relu(disc_dense3_)

            disc_output = tf.compat.v1.layers.dense(inputs=disc_dense3, units=1, activation=None)
            return disc_output, disc_dense3_

    def discriminator2(self, x_input_list):
        """
        discriminator of WGAN-GP
        """
        with tf.compat.v1.variable_scope('discriminator2', reuse=tf.compat.v1.AUTO_REUSE):
            x_input = tf.concat(x_input_list, 0)
            disc_dense1 = tf.compat.v1.layers.dense(inputs=x_input, units=self.disc_unit1, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense1 = tf.layers.batch_normalization(disc_dense1, training=self.is_training)
            disc_dense1 = tf.nn.leaky_relu(disc_dense1)

            disc_dense2 = tf.compat.v1.layers.dense(inputs=disc_dense1, units=self.disc_unit2, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense2 = tf.layers.batch_normalization(disc_dense2, training=self.is_training)
            disc_dense2 = tf.nn.leaky_relu(disc_dense2)

            disc_concat = tf.concat([disc_dense2[:tf.shape(input=x_input_list[0])[0]], disc_dense2[tf.shape(input=x_input_list[0])[0]:]], 1)

            disc_dense3_ = tf.compat.v1.layers.dense(inputs=disc_concat, units=self.disc_unit3, activation=None, kernel_regularizer=self.regu_w)
            # disc_dense3 = tf.layers.batch_normalization(disc_dense3, training=self.is_training)
            disc_dense3 = tf.nn.relu(disc_dense3_)
            
            disc_output = tf.compat.v1.layers.dense(inputs=disc_dense3, units=1, activation=None)
            return disc_output

    def create_network(self):
        """
        construct the WGAN-GP networks
        """
        # reconstruction
        self.z_encode_mu, self.z_encode_sig, self.z_encode = self.encoder(self.x)
        self.z_mapping = self.mapper(self.z_encode)
        self.x_recon_data = self.generatorDropOut(self.z_mapping)
        self.Dex_real, self.Dex_real_hidden = self.discriminator(self.x_recon_data)

        #consistency regularization
        #self.x_aug_noise = self.x_aug + (10**tf.random.uniform((), minval=-2, maxval=-0.3)) * tf.random.normal(shape=tf.shape(input=self.x_aug))
        #self.z_encode_mu_aug, self.z_encode_sig_aug, self.z_encode_aug = self.encoder(self.x_aug_noise)
        self.x_aug = self.x + (10**tf.random.uniform((), minval=-2, maxval=-0.3)) * tf.random.normal(shape=tf.shape(input=self.x))
        self.z_encode_mu_aug, self.z_encode_sig_aug, self.z_encode_aug = self.encoder(self.x_aug)
        #_, self.Dx_real_aug_hidden = self.discriminator(self.x_aug_noise)
        
        # triple operation
        t1_mu, t1_sig, self.t_term1_encode = self.encoder(self.t_term1)
        self.t_term1_mapping = self.mapper(self.t_term1_encode)
        t2_mu, t2_sig, self.t_term2_encode = self.encoder(self.t_term2)
        self.t_term2_mapping = self.mapper(self.t_term2_encode)
        t3_mu, t3_sig, self.t_term3_encode = self.encoder(self.t_term3)
        self.t_term3_mapping = self.mapper(self.t_term3_encode)
        t4_mu, t4_sig, self.t_term4_encode = self.encoder(self.t_term4)
        self.t_term4_mapping = self.mapper(self.t_term4_encode)
        
        self.arithmetic_triple = self.t_term1_mapping - self.t_term2_mapping + self.t_term3_mapping
        self.gen_arith = self.generatorDropOut(self.arithmetic_triple)
        self.gen_target = self.generatorDropOut(self.t_term4_mapping)
        _, self.tri_real_hidden = self.discriminator(self.t_term4)
        self.tri_fake, self.tri_fake_hidden = self.discriminator(self.gen_arith)

        # delta operation
        d1_mu, d1_sig, self.d_term1_encode = self.encoder(self.d_term1)
        self.d_term1_mapping = self.mapper(self.d_term1_encode)
        d2_mu, d2_sig, self.d_term2_encode = self.encoder(self.d_term2)
        self.d_term2_mapping = self.mapper(self.d_term2_encode)
        d3_mu, d3_sig, self.d_term3_encode = self.encoder(self.d_term3)
        self.d_term3_mapping = self.mapper(self.d_term3_encode)
        d4_mu, d4_sig, self.d_term4_encode = self.encoder(self.d_term4)
        self.d_term4_mapping = self.mapper(self.d_term4_encode)
        
        self.left_arithmetic = self.d_term1_mapping - self.d_term2_mapping
        self.right_arithmetic = self.d_term3_mapping - self.d_term4_mapping

        self.left_arithmetic_c = self.d_term2_mapping - self.d_term4_mapping
        self.right_arithmetic_c = self.d_term1_mapping - self.d_term3_mapping

        d1u_mu, d1u_sig, self.du_term1_encode = self.encoder(self.du_term1)
        self.du_term1_mapping = self.mapper(self.du_term1_encode)
        d2u_mu, d2u_sig, self.du_term2_encode = self.encoder(self.du_term2)
        self.du_term2_mapping = self.mapper(self.du_term2_encode)
        d3u_mu, d3u_sig, self.du_term3_encode = self.encoder(self.du_term3)
        self.du_term3_mapping = self.mapper(self.du_term3_encode)
        d4u_mu, d4u_sig, self.du_term4_encode = self.encoder(self.du_term4)
        self.du_term4_mapping = self.mapper(self.du_term4_encode)
        self.left_arithmetic_u = self.du_term1_mapping - self.du_term2_mapping
        self.right_arithmetic_u = self.du_term3_mapping - self.du_term4_mapping

        # tri-delta swap
        self.arithmetic_triple_fromdel = self.d_term1_mapping - self.d_term2_mapping + self.d_term4_mapping
        self.gen_arith_fromdel = self.generatorDropOut(self.arithmetic_triple_fromdel)
        self.gen_target_fromdel = self.generatorDropOut(self.d_term3_mapping)
        _, self.tri_fromdel_real_hidden = self.discriminator(self.d_term3)
        self.tri_fromdel_fake, self.tri_fromdel_fake_hidden = self.discriminator(self.gen_arith_fromdel)

        self.left_arithmetic_fromtri = self.t_term1_mapping - self.t_term2_mapping
        self.right_arithmetic_fromtri = self.t_term4_mapping - self.t_term3_mapping
        
        # second discriminator
        self.D2_real = self.discriminator2([self.dd_term1, self.dd_term2])#same cell line
        self.D2_real_dif = self.discriminator2([self.dd_term1, self.dd_term3])#different cell line
        self.D2_fake = self.discriminator2([tf.concat([self.t_term4, self.d_term3], 0),
                                            tf.concat([self.gen_arith, self.gen_arith_fromdel], 0)])#same cell line (one is generated)

        # generate fake data
        '''
        z_from_xz = self.z_encode_mu + tf.random_normal(shape=[tf.shape(self.x)[0], self.z_dim], stddev=self.z_encode_sig)
        #z_from_xz = self.z_encode + tf.random_normal(shape=[tf.shape(self.x)[0], self.z_dim])
        z_from_xz = tf.stop_gradient(z_from_xz)
        self.x_gen_data = self.generatorDropOut(z_from_xz)
        '''
        noisemapping = self.mapper(self.z)
        # noisemapping = tf.stop_gradient(noisemapping)
        self.x_gen_data = self.generatorDropOut(noisemapping)

        self.xx_encode_mu, self.xx_encode_sig, self.xx_encode = self.encoder(self.x_bulk)
        self.xx_mapping = self.mapper(self.xx_encode)
        self.x_gen_bulk = self.generatorDropOut(self.xx_mapping)
        self.x_gen_bulk = tf.stop_gradient(self.x_gen_bulk)

        # score for real/fake data
        self.Dx_real, self.Dx_real_hidden = self.discriminator(self.x)
        self.Dx_fake, self.Dx_fake_hidden = self.discriminator(self.x_gen_data)

        #pathreg https://github.com/NVlabs/stylegan2/blob/master/training/loss.py
        pl_latents = tf.random.normal(tf.shape(input=self.z))
        pl_latents_mapping = self.mapper(pl_latents)
        # pl_latents_mapping = tf.stop_gradient(self.z_mapping)
        fake_out, fake_dlatents_out = self.generatorDropOut(pl_latents_mapping, return_dlatents=True)

        pl_noise = tf.random.normal(tf.shape(input=fake_out)) / np.sqrt(self.x_dim)
        pl_grads = tf.gradients(ys=tf.reduce_sum(input_tensor=fake_out * pl_noise), xs=[fake_dlatents_out])[0]
        self.pl_lengths = pl_lengths = tf.sqrt(tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(pl_grads), axis=2), axis=1))

        with tf.control_dependencies(None):
            pl_mean_var = tf.Variable(name='pl_mean', trainable=False, initial_value=0.0, dtype=tf.float32)
        pl_mean = pl_mean_var + 0.01 * (tf.reduce_mean(input_tensor=pl_lengths) - pl_mean_var)
        pl_update = tf.compat.v1.assign(pl_mean_var, pl_mean)

        # Calculate (|J*y|-a)^2.
        with tf.control_dependencies([pl_update]):
            self.pl_penalty = tf.square(pl_lengths - pl_mean)
            
    def compute_gp(self, x, x_gen_data):
        """
        gradient penalty of discriminator
        """
        epsilon_x = tf.random.uniform([tf.shape(input=x)[0],1], 0.0, 1.0)
        x_hat = x * epsilon_x + (1 - epsilon_x) * x_gen_data

        d_hat, _ = self.discriminator(x_hat)
        gradients = tf.gradients(ys=d_hat, xs=x_hat)[0]

        slopes = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients), axis=[1]))
        gradient_penalty =  tf.reduce_mean(input_tensor=(slopes - 1.0) ** 2)

        return gradient_penalty

    def loss_function(self):
        """
        loss function of WGAN-GP
        """

        # reconstruction MSE + discriminator's feature space MSE  -- no gradient to disc.
        # self.recon_loss = tf.reduce_mean(tf.squared_difference(self.x, self.x_recon_data)) +\
            # 1.0 * tf.reduce_mean(tf.squared_difference(self.Dx_real_hidden, self.Dex_real_hidden))

        oriprob = tfp.distributions.Normal(self.x_recon_data, 1)
        ori_logp_loss = -tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=oriprob.log_prob(self.x), axis=1) / self.x_dim)
        # discprob = tfp.distributions.Normal(self.Dex_real_hidden, 1)
        # logp_loss = -tf.reduce_mean(tf.reduce_sum(discprob.log_prob(self.Dx_real_hidden), 1) / self.disc_unit3)
        self.recon_loss = ori_logp_loss# + 0.1 * logp_loss

        prior = tfp.distributions.Normal(tf.zeros([tf.shape(self.x)[0], self.z_dim]), tf.ones([tf.shape(self.x)[0], self.z_dim]))
        #prior = tfp.distributions.Normal(tf.zeros([self.batch_size, self.z_dim]), tf.ones([self.batch_size, self.z_dim]))
        latent_prior = prior.log_prob(self.z_encode)
        z_norm = (self.z_encode - self.z_encode_mu) / self.z_encode_sig
        z_var = tf.square(self.z_encode_sig)
        latent_posterior = -0.5 * (z_norm * z_norm + tf.compat.v1.log(z_var) + np.log(2 * np.pi))
        
        latent_prior_joint = tf.reduce_sum(latent_prior, 1) / self.z_dim
        latent_posterior_joint = tf.reduce_sum(latent_posterior, 1) / self.z_dim
        kl_latent = -tf.reduce_mean(latent_prior_joint) + tf.reduce_mean(latent_posterior_joint)
        '''
        def compute_kernel(x, y):
            x_size = tf.shape(input=x)[0]
            y_size = tf.shape(input=y)[0]
            dim = tf.shape(input=x)[1]
            tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
            tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
            return tf.exp(-tf.reduce_mean(input_tensor=tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))
        
        def compute_mmd(x, y):
            x_kernel = compute_kernel(x, x)
            y_kernel = compute_kernel(y, y)
            xy_kernel = compute_kernel(x, y)
            return tf.reduce_mean(input_tensor=x_kernel) + tf.reduce_mean(input_tensor=y_kernel) - 2 * tf.reduce_mean(input_tensor=xy_kernel)
        
        true_samples = tf.random.normal(tf.stack([200, self.z_dim]))
        loss_mmd = compute_mmd(true_samples, self.z_mapping) +\
        '''
        loss_mmd = 0.1 * tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    tf.norm(tensor=tf.random.normal(tf.shape(input=self.z_mapping)), axis=-1),
                    tf.norm(tensor=self.z_mapping, axis=-1))) +\
                0.1 * tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    tf.norm(tensor=tf.random.normal(tf.shape(input=self.arithmetic_triple)), axis=-1),
                    tf.norm(tensor=self.arithmetic_triple, axis=-1))) +\
                0.1 * tf.reduce_mean(input_tensor=tf.math.squared_difference(
                    tf.norm(tensor=tf.random.normal(tf.shape(input=self.arithmetic_triple_fromdel)), axis=-1),
                    tf.norm(tensor=self.arithmetic_triple_fromdel, axis=-1)))

        # consistency regularization
        std_orig = tf.exp(self.z_encode_sig)
        std_aug = tf.exp(self.z_encode_sig_aug)
        self.cr_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(2 * tf.compat.v1.log(std_orig / std_aug) - 1\
                + (std_aug**2 + (self.z_encode_mu_aug - self.z_encode_mu) ** 2) / std_orig**2, axis=1))
        #self.gan_cr_loss = tf.reduce_mean(input_tensor=tf.compat.v1.squared_difference(self.Dx_real_hidden, self.Dx_real_aug_hidden))

        # triple_target = tf.stop_gradient(self.gen_target)
        # self.triple_loss = tf.reduce_mean(tf.squared_difference(self.gen_target, self.gen_arith))
        #self.triple_loss = tf.reduce_mean(tf.losses.absolute_difference(self.t_term4_mapping, self.arithmetic_triple)) +\
        #    tf.reduce_mean(tf.losses.absolute_difference(self.d_term3_mapping, self.arithmetic_triple_fromdel)) +\
        #    tf.reduce_mean(tf.losses.absolute_difference(self.t_term4, self.gen_arith)) +\
        #    tf.reduce_mean(tf.losses.absolute_difference(self.d_term3, self.gen_arith_fromdel))# +\
            #0.1 * tf.reduce_mean(tf.squared_difference(self.tri_real_hidden, self.tri_fake_hidden)) +\
            #0.1 * tf.reduce_mean(tf.squared_difference(self.tri_fromdel_real_hidden, self.tri_fromdel_fake_hidden))
        #self.triple_loss = tf.reduce_mean(tf.losses.absolute_difference(self.t_term4, self.gen_arith)) +\
        #    tf.reduce_mean(tf.losses.absolute_difference(self.d_term3, self.gen_arith_fromdel))
        self.triple_loss = tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(self.t_term4, self.gen_arith)) +\
            tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(self.d_term3, self.gen_arith_fromdel))

        # delta_target = tf.stop_gradient(self.right_arithmetic)
        self.delta_loss = (tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(self.right_arithmetic, self.left_arithmetic)) +\
                tf.reduce_mean(input_tensor=tf.compat.v1.losses.absolute_difference(self.right_arithmetic_c, self.left_arithmetic_c))) / 2


        '''
        delmask = tf.ones([tf.shape(self.right_arithmetic_u)[0], tf.shape(self.right_arithmetic_u)[0]], dtype=tf.float32)
        delmask = delmask - tf.linalg.band_part(delmask, 0, 0)
        delmask = tf.cast(delmask, tf.bool)
        delta_dissimilarity_1 = tf.matmul(self.right_arithmetic_u, tf.transpose(self.right_arithmetic_u))
        delta_dissimilarity_2 = tf.matmul(self.left_arithmetic_u, tf.transpose(self.left_arithmetic_u))
        delta_dissimilarity = tf.reduce_mean(tf.abs(tf.boolean_mask(delta_dissimilarity_1, delmask))) + tf.reduce_mean(tf.abs(tf.boolean_mask(delta_dissimilarity_2, delmask)))
        self.delta_loss = self.delta_loss + 0.01 * delta_dissimilarity
        '''

        '''
        delta_dissimilarity = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.abs(tf.expand_dims(self.right_arithmetic_u, 1) - tf.expand_dims(self.right_arithmetic_u, 0)), axis=2) / self.z_dim) +\
                              tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.abs(tf.expand_dims(self.left_arithmetic_u, 1) - tf.expand_dims(self.left_arithmetic_u, 0)), axis=2) / self.z_dim)
        self.delta_loss = self.delta_loss + tf.maximum(2.0 - delta_dissimilarity, 0.)
        '''

        norm_right = tf.nn.l2_normalize(self.right_arithmetic_u, axis=1)
        norm_left = tf.nn.l2_normalize(self.left_arithmetic_u, axis=1)
        logits_supcon = tf.matmul(norm_right, norm_left, transpose_b=True)

        temperature = 0.1
        logits_supcon = logits_supcon / temperature
        logits_supcon = (logits_supcon - tf.reduce_max(tf.stop_gradient(logits_supcon), axis=1, keepdims=True))
        exp_logits_supcon = tf.exp(logits_supcon)

        denominator = tf.reduce_sum(exp_logits_supcon, axis=1)
        delmask = tf.ones([tf.shape(self.right_arithmetic_u)[0], tf.shape(self.right_arithmetic_u)[0]], dtype=tf.float32)
        delmask = tf.linalg.band_part(delmask, 0, 0)
        delmask = tf.cast(delmask, tf.bool)
        numerator = tf.boolean_mask(exp_logits_supcon, delmask)
        loss_supcon = -tf.math.log(numerator / denominator)
        self.delta_loss = self.delta_loss + (10/self.lamb_delta) * tf.reduce_mean(loss_supcon * temperature)
        
        # self.D_raw_loss = (tf.reduce_mean(tf.nn.softplus(self.Dx_real)) + tf.reduce_mean(tf.nn.softplus(self.D2_real)) +\
        #                    tf.reduce_mean(tf.nn.softplus(-self.D2_real_dif))) / 3 +\
        #     (tf.reduce_mean(tf.nn.softplus(-self.Dx_fake)) + tf.reduce_mean(tf.nn.softplus(-self.Dex_real)) +\
        #      tf.reduce_mean(tf.nn.softplus(-self.tri_fake)) + tf.reduce_mean(tf.nn.softplus(-self.tri_fromdel_fake)) +\
        #      tf.reduce_mean(tf.nn.softplus(-self.D2_fake))) / 5# - tf.reduce_mean(self.Dex_real)    #according to VAE/GAN
        self.D_raw_loss = (tf.reduce_mean(input_tensor=tf.nn.softplus(self.Dx_real)) + tf.reduce_mean(input_tensor=tf.nn.softplus(self.D2_real))) / 2 +\
            (tf.reduce_mean(input_tensor=tf.nn.softplus(-self.Dx_fake)) + tf.reduce_mean(input_tensor=tf.nn.softplus(-self.Dex_real)) +\
             tf.reduce_mean(input_tensor=tf.nn.softplus(-self.tri_fake)) + tf.reduce_mean(input_tensor=tf.nn.softplus(-self.tri_fromdel_fake)) +\
             tf.reduce_mean(input_tensor=tf.nn.softplus(-self.D2_fake)) + tf.reduce_mean(input_tensor=tf.nn.softplus(-self.D2_real_dif))) / 6
        self.G_raw_loss = (tf.reduce_mean(input_tensor=tf.nn.softplus(self.Dx_fake)) + tf.reduce_mean(input_tensor=tf.nn.softplus(self.Dex_real)) +\
                           tf.reduce_mean(input_tensor=tf.nn.softplus(self.tri_fake)) + tf.reduce_mean(input_tensor=tf.nn.softplus(self.tri_fromdel_fake)) +\
                           tf.reduce_mean(input_tensor=tf.nn.softplus(self.D2_fake))) / 5
        self.G_loss = self.G_raw_loss +\
            self.lamb_recon * self.recon_loss + self.lamb_triple * self.triple_loss + self.lamb_delta * self.delta_loss + kl_latent + loss_mmd + self.cr_loss + 2 * self.pl_penalty +\
                tf.compat.v1.losses.get_regularization_loss("generatorDropOut")
        # self.gradient_penalty = self.compute_gp(self.x, self.x_gen_data)
        with tf.compat.v1.name_scope('GradientPenalty'):
            real_grads = tf.gradients(ys=tf.reduce_sum(input_tensor=self.Dx_real), xs=[self.x])[0]
            #real_grads_2 = tf.concat(tf.gradients(ys=tf.reduce_sum(input_tensor=self.D2_real), xs=[self.dd_term1, self.dd_term2]), axis=0)
            self.gradient_penalty = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(real_grads), axis=1))# + tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.square(real_grads_2), axis=1))
        self.D_loss = self.D_raw_loss + self.lamb_gp * 0.5 * self.gradient_penalty + tf.compat.v1.losses.get_regularization_loss("discriminator") +\
            tf.compat.v1.losses.get_regularization_loss("discriminator2")

        tf_vars_all = tf.compat.v1.trainable_variables()

        evars  = [var for var in tf_vars_all if var.name.startswith("encoder")]
        mvars  = [var for var in tf_vars_all if var.name.startswith("mapper")]
        dvars  = [var for var in tf_vars_all if var.name.startswith("discriminator")]
        d2vars  = [var for var in tf_vars_all if var.name.startswith("discriminator2")]
        gvars  = [var for var in tf_vars_all if var.name.startswith("generatorDropOut")]

        self.parameter_count = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in evars + mvars + dvars + gvars])

        with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
            self.g_train_op = Adam_Prediction_Optimizer(learning_rate = self.learning_rate,
                    beta1=0.9, beta2=0.999, prediction=True).minimize(self.G_loss, var_list = evars + mvars + gvars)
            self.d_train_op = Adam_Prediction_Optimizer(learning_rate = self.learning_rate,
                    beta1=0.9, beta2=0.999, prediction=False).minimize(self.D_loss, var_list = dvars + d2vars)


@nb.njit(fastmath=True,parallel=True)
def calc_distance(vec_1,vec_2):
    res=np.empty((vec_1.shape[0],vec_2.shape[0]),dtype='float32')
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            #res[i,j]=np.sqrt((vec_1[i,0]-vec_2[j,0])**2+(vec_1[i,1]-vec_2[j,1])**2+(vec_1[i,2]-vec_2[j,2])**2)
            res[i,j]=np.sqrt(np.sum((vec_1[i]-vec_2[j])**2))

    return res

class Batch_maker:
    def __init__(self, data, data_info, triweight=False):
        print('Initialize dataset...')
        self.data = data
        self.data_len = self.data.shape[0]
        self.data_info = data_info
        self.triweight = triweight
        
        #self.cell = np.unique(self.data_info[:,0])
        self.cell = np.unique(self.data_info[self.data_info[:,1] != 'control',0])
        self.pert = np.unique(self.data_info[:,1])
        
        self.idx_control = np.where(self.data_info[:,1] == 'control')[0]
        self.idx_dict = {}
        
        print('cell & perturbation indexing...')
        for c in self.cell:
            print('cell', c)
            self.idx_dict[c] = {}
            cidx = self.data_info[:,0] == c
            for _, p in enumerate(self.pert):
                if _ % 1000 == 0:
                    print(_,'/',self.pert.size)
                pidx = self.data_info[:,1] == p
                cpidx = cidx & pidx
                if np.sum(cpidx) > 0:
                    self.idx_dict[c][p] = np.where(cpidx)[0]
                
        # self.cell - sorted in alphabetical order
        self.common_pert = {}
        for a in range(self.cell.size-1):
            self.common_pert[self.cell[a]] = {}
            for b in range(a+1,self.cell.size):
                pertlist = np.intersect1d(list(self.idx_dict[self.cell[a]].keys()), list(self.idx_dict[self.cell[b]].keys()))
                # pertlist = list(set(pertlist) - set(['control']))
                self.common_pert[self.cell[a]][self.cell[b]] = pertlist

        if self.triweight:
            self.dist_contpert = {}
            for a in range(self.cell.size-1):
                self.dist_contpert[self.cell[a]] = {}
                for b in range(a+1,self.cell.size):
                    print('triweight cell', self.cell[a], '-', self.cell[b])
                    pertlist = np.intersect1d(list(self.idx_dict[self.cell[a]].keys()), list(self.idx_dict[self.cell[b]].keys()))

                    if pertlist.size > 1:
                        apert = self.idx_dict[self.cell[a]]
                        acont = self.idx_dict[self.cell[a]]['control']
                        bpert = self.idx_dict[self.cell[b]]
                        bcont = self.idx_dict[self.cell[b]]['control']

                        dist_weight = []
                        for _, p in enumerate(pertlist):
                            if _ % 1000 == 0:
                                print(_,'/',pertlist.size)
                            #dist_weight.append(np.mean([np.mean(cdist(self.data[acont], self.data[apert[p]], metric='minkowski', p=2)),
                            #np.mean(cdist(self.data[bcont], self.data[bpert[p]], metric='minkowski', p=2))]))
                            dist_weight.append(np.mean([np.mean(calc_distance(self.data[acont], self.data[apert[p]])),
                            np.mean(calc_distance(self.data[bcont], self.data[bpert[p]]))]))
                        dist_weight = np.array(dist_weight)
                        dist_weight[dist_weight<1e-4] = np.mean(dist_weight[dist_weight>1e-4])
                        e_x = np.exp(-dist_weight/5) #high temper --> flat dist.
                        pw = e_x / e_x.sum()
                        # pertlist = list(set(pertlist) - set(['control']))
                        self.dist_contpert[self.cell[a]][self.cell[b]] = pw

        print('done!')
        

    def get(self, shuffle, batch_size, start=0):
        if shuffle:
            if len(self.idx_control)>100:
                rand_idx = np.random.choice(self.data_len, int(batch_size/2), replace=False)
                rand_idx_c = np.random.choice(self.idx_control, int(batch_size/2), replace=False)
                rand_idx = np.concatenate((rand_idx, rand_idx_c))
            else:
                rand_idx = np.random.choice(self.data_len, batch_size, replace=False)
        else:
            rand_idx = np.arange(start,start+batch_size)

        return self.data[rand_idx,:]
    
    def get_triple_l_sample(self, batch_size):
        '''
        select samples for triple loss
        '''
        term1, term2, term3, term4 = [], [], [], [] # term1 - term2 + term3 = term4
        term1_, term2_, term3_, term4_ = [], [], [], [] # term1 - term2 + term3 = term4
        sampled_count = 0
        while sampled_count < batch_size:
            A, B = np.sort(np.random.choice(self.cell, 2, replace=False)) # self.cell - sorted in alphabetical order
            compert = self.common_pert[A][B]
            if len(compert) > 2:
                if self.triweight:
                    compert_p = self.dist_contpert[A][B]
                if np.random.rand() < 0.5:
                    A, B = B, A
                if self.triweight:
                    a, b = np.random.choice(compert, 2, replace=False, p=compert_p)
                else:
                    a, b = np.random.choice(compert, 2, replace=False)
                # if (a != 'control') & (b != 'control'):
                #     if np.random.rand() < 0.1:
                #         b = 'control'
                if (a == 'control') or (b == 'control'):
                    if a == 'control':
                        a, b = b, a
                    rn = np.random.rand()
                    if rn < 1/4:
                        # Aa - A + B = Ba
                        first = self.idx_dict[A][a]   #A-a
                        second = self.idx_dict[A][b]  #A-control
                        third = self.idx_dict[B][b]   #B-control
                        fourth = self.idx_dict[B][a]  #B-a
                    elif rn < 2/4:
                        # A - Aa + Ba = B
                        first = self.idx_dict[A][b]   #A-control
                        second = self.idx_dict[A][a]  #A-a
                        third = self.idx_dict[B][a]   #B-a
                        fourth = self.idx_dict[B][b]  #B-control
                    elif rn < 3/4:
                        # A - B + Ba = Aa
                        first = self.idx_dict[A][b]   #A-control
                        second = self.idx_dict[B][b]  #B-control
                        third = self.idx_dict[B][a]   #B-a
                        fourth = self.idx_dict[A][a]  #A-a
                    else:
                        # Aa - Ba + B = A
                        first = self.idx_dict[A][a]   #A-a
                        second = self.idx_dict[B][a]  #B-a
                        third = self.idx_dict[B][b]   #B-control
                        fourth = self.idx_dict[A][b]  #A-control
                else:
                    if np.random.rand() < 0.5:
                        # Aa - Ba + Bb = Ab
                        first = self.idx_dict[A][a]   #A-a
                        second = self.idx_dict[B][a]  #B-a
                        third = self.idx_dict[B][b]   #B-b
                        fourth = self.idx_dict[A][b]  #A-b
                    else:
                        # Aa - Ab + Bb = Ba
                        first = self.idx_dict[A][a]   #A-a
                        second = self.idx_dict[A][b]  #A-b
                        third = self.idx_dict[B][b]   #B-b
                        fourth = self.idx_dict[B][a]  #B-a

                term1.append(np.random.choice(first))
                term2.append(np.random.choice(second))
                term3.append(np.random.choice(third))
                term4.append(np.random.choice(fourth))
                term1_.append(np.random.choice(first))
                term2_.append(np.random.choice(second))
                term3_.append(np.random.choice(third))
                term4_.append(np.random.choice(fourth))
                
                sampled_count += 1
            
        return self.data[term1,:], self.data[term2,:], self.data[term3,:], self.data[term4,:], self.data[term1_,:], self.data[term2_,:], self.data[term3_,:], self.data[term4_,:]

    def get_delta_l_sample(self, batch_size):
        '''
        select samples for delta loss
        '''
        term1, term2, term3, term4 = [], [], [], [] # term1 - term2 = term3 - term4
        term1_, term2_, term3_, term4_ = [], [], [], [] # term1 - term2 = term3 - term4
        selalphas = []
        sampled_count = 0
        while sampled_count < batch_size:
            A, B = np.sort(np.random.choice(self.cell, 2, replace=False)) # self.cell - sorted in alphabetical order
            compert = self.common_pert[A][B]
            if len(compert) > 2:
                if self.triweight:
                    compert_p = self.dist_contpert[A][B]
                if np.random.rand() < 0.5:
                    A, B = B, A
                if self.triweight:
                    a, b = np.random.choice(compert, 2, replace=False, p=compert_p)
                else:
                    a, b = np.random.choice(compert, 2, replace=False)
                if (a != 'control') & (b != 'control'):
                    # if np.random.rand() < 0.1:
                    b = 'control'
                elif a == 'control':
                # if a == 'control':
                    a, b = b, a
                first = self.idx_dict[A][a]   #A-a
                second = self.idx_dict[A][b]  #A-b/control
                third = self.idx_dict[B][a]   #B-a
                fourth = self.idx_dict[B][b]  #B-b/control

                #if np.random.rand() < 0.5:
                # Aa-Ab == Ba-Bb
                term1.append(np.random.choice(first))
                term2.append(np.random.choice(second, np.max([int(second.size*0.2), 1]), replace=False))
                term3.append(np.random.choice(third))
                term4.append(np.random.choice(fourth, np.max([int(fourth.size*0.2), 1]), replace=False))
                term1_.append(np.random.choice(first))
                term2_.append(np.random.choice(second, np.max([int(second.size*0.2), 1]), replace=False))
                term3_.append(np.random.choice(third))
                term4_.append(np.random.choice(fourth, np.max([int(fourth.size*0.2), 1]), replace=False))
                '''
                else:
                    # Aa-Ba == Ab-Bb
                    term1.append(first)
                    term2.append(third)
                    term3.append(second)
                    term4.append(fourth)
                '''
                
                sampled_count += 1
                selalphas.append(a)

        _, uidx = np.unique(selalphas, return_index=True)

        selalphas = np.array(selalphas)
        mask = np.zeros([batch_size, batch_size])
        for b in range(batch_size):
            mask[b, selalphas==selalphas[b]] = 1
            
        d2data = np.concatenate([self.data[_,:].mean(axis=0, keepdims=True) for _ in term2], axis=0)
        d4data = np.concatenate([self.data[_,:].mean(axis=0, keepdims=True) for _ in term4], axis=0)
        d2_data = np.concatenate([self.data[_,:].mean(axis=0, keepdims=True) for _ in term2_], axis=0)
        d4_data = np.concatenate([self.data[_,:].mean(axis=0, keepdims=True) for _ in term4_], axis=0)
        #return self.data[term1,:], self.data[term2,:], self.data[term3,:], self.data[term4,:], self.data[term1_,:], self.data[term2_,:], self.data[term3_,:], self.data[term4_,:], self.data[term1,:][uidx], self.data[term2,:][uidx], self.data[term3,:][uidx], self.data[term4,:][uidx]
        return self.data[term1,:], d2data, self.data[term3,:], d4data, self.data[term1_,:], d2_data, self.data[term3_,:], d4_data, self.data[term1,:][uidx], d2data[uidx], self.data[term3,:][uidx], d4data[uidx]

    def get_disc_l_sample(self, batch_size):
        '''
        select samples for second discriminator
        '''
        term1, term2, term3 = [], [], []
        sampled_count = 0
        while sampled_count < batch_size:
            A, B = np.sort(np.random.choice(self.cell, 2, replace=False)) # self.cell - sorted in alphabetical order
            if np.random.rand() < 0.5:
                A, B = B, A
            # a, b = np.random.choice(list(self.idx_dict[A].keys()), 2, replace=True)
            # c = np.random.choice(list(self.idx_dict[B].keys()), 1, replace=True)[0]
            # first = np.random.choice(self.idx_dict[A][a])   #A-a
            # second = np.random.choice(self.idx_dict[A][b])  #A-b
            # third = np.random.choice(self.idx_dict[B][c])   #B-c
            
            if (len(list(self.idx_dict[A].keys())) > 1) | (len(list(self.idx_dict[B].keys())) > 1):
                if len(list(self.idx_dict[A].keys())) < 2:
                    a, b = np.random.choice(list(self.idx_dict[B].keys()), 2, replace=False)
                    first = np.random.choice(self.idx_dict[B][a])   #A-a
                    second = np.random.choice(self.idx_dict[B][a])  #A-a
                    third = np.random.choice(self.idx_dict[B][b])   #B-b
                else:
                    a, b = np.random.choice(list(self.idx_dict[A].keys()), 2, replace=False)
                    first = np.random.choice(self.idx_dict[A][a])   #A-a
                    second = np.random.choice(self.idx_dict[A][a])  #A-a
                    third = np.random.choice(self.idx_dict[A][b])   #B-b
            
                term1.append(first)
                term2.append(second)
                term3.append(third)
                
                sampled_count += 1
            
        return self.data[term1,:], self.data[term2,:], self.data[term3,:]

def run_train(sess, model, train_dataset, batch_size=32, genesym=None, pmodel=None):
    total_D_loss, total_G_loss, total_recon_loss, total_triple_loss, total_delta_loss, total_pl_lengths, total_pl_penalty, total_pact_cor = [], [], [], [], [], [], [], []
    for step in range(train_dataset.data_len // (batch_size*9)):
        for di in range(model.Diters):
            xt1_mb, xt2_mb, xt3_mb, xt4_mb, xt1c_mb, xt2c_mb, xt3c_mb, xt4c_mb = train_dataset.get_triple_l_sample(batch_size)
            xd1_mb, xd2_mb, xd3_mb, xd4_mb, xd1c_mb, xd2c_mb, xd3c_mb, xd4c_mb, xd1u_mb, xd2u_mb, xd3u_mb, xd4u_mb = train_dataset.get_delta_l_sample(batch_size)
            xdd1_mb, xdd2_mb, xdd3_mb = train_dataset.get_disc_l_sample(batch_size)
            x_mb = train_dataset.get(True, batch_size)
            z_mb = np.random.normal(0.0, scale = 1.0, size = (batch_size*9, model.z_dim))
            feed_dict = {model.t_term1:xt1_mb, model.t_term2:xt2_mb, model.t_term3:xt3_mb, model.t_term4:xt4_mb,
                         model.tc_term1:xt1c_mb, model.tc_term2:xt2c_mb, model.tc_term3:xt3c_mb, model.tc_term4:xt4c_mb,
                         model.d_term1:xd1_mb, model.d_term2:xd2_mb, model.d_term3:xd3_mb, model.d_term4:xd4_mb,
                         model.dc_term1:xd1c_mb, model.dc_term2:xd2c_mb, model.dc_term3:xd3c_mb, model.dc_term4:xd4c_mb,
                         model.du_term1:xd1u_mb, model.du_term2:xd2u_mb, model.du_term3:xd3u_mb, model.du_term4:xd4u_mb,
                         model.dd_term1:xdd1_mb, model.dd_term2:xdd2_mb, model.dd_term3:xdd3_mb,
                         model.no_term:x_mb, model.z:z_mb, model.batch_size:batch_size*9,
                         model.x_bulk:xt1_mb, model.gennum:xt1_mb.shape[0], model.is_training:True}
            sess.run(model.d_train_op, feed_dict=feed_dict)

        xt1_mb, xt2_mb, xt3_mb, xt4_mb, xt1c_mb, xt2c_mb, xt3c_mb, xt4c_mb = train_dataset.get_triple_l_sample(batch_size)
        xd1_mb, xd2_mb, xd3_mb, xd4_mb, xd1c_mb, xd2c_mb, xd3c_mb, xd4c_mb, xd1u_mb, xd2u_mb, xd3u_mb, xd4u_mb = train_dataset.get_delta_l_sample(batch_size)
        xdd1_mb, xdd2_mb, xdd3_mb = train_dataset.get_disc_l_sample(batch_size)
        x_mb = train_dataset.get(True, batch_size)
        z_mb = np.random.normal(0.0, scale = 1.0, size = (batch_size*9, model.z_dim))
        feed_dict = {model.t_term1:xt1_mb, model.t_term2:xt2_mb, model.t_term3:xt3_mb, model.t_term4:xt4_mb,
                     model.tc_term1:xt1c_mb, model.tc_term2:xt2c_mb, model.tc_term3:xt3c_mb, model.tc_term4:xt4c_mb,
                     model.d_term1:xd1_mb, model.d_term2:xd2_mb, model.d_term3:xd3_mb, model.d_term4:xd4_mb,
                     model.dc_term1:xd1c_mb, model.dc_term2:xd2c_mb, model.dc_term3:xd3c_mb, model.dc_term4:xd4c_mb,
                     model.du_term1:xd1u_mb, model.du_term2:xd2u_mb, model.du_term3:xd3u_mb, model.du_term4:xd4u_mb,
                     model.dd_term1:xdd1_mb, model.dd_term2:xdd2_mb, model.dd_term3:xdd3_mb,
                     model.no_term:x_mb, model.z:z_mb, model.batch_size:batch_size*9,
                     model.x_bulk:xt1_mb, model.gennum:xt1_mb.shape[0], model.is_training:True}
        _, D_loss, G_loss, recon_loss, triple_loss, delta_loss, pl_lengths, pl_penalty, gen1, gen2 = sess.run([model.g_train_op, model.D_raw_loss, model.G_raw_loss,
                                                                           model.recon_loss, model.triple_loss, model.delta_loss,
                                                                           model.pl_lengths, model.pl_penalty,
                                                                           model.gen_arith, model.gen_arith_fromdel],
                                                                          feed_dict=feed_dict)

        total_D_loss.append(D_loss)
        total_G_loss.append(G_loss)
        total_recon_loss.append(recon_loss)
        total_triple_loss.append(triple_loss)
        total_delta_loss.append(delta_loss)
        total_pl_lengths.append(pl_lengths)
        total_pl_penalty.append(pl_penalty)
        if genesym is not None:
            temp = pd.DataFrame(np.concatenate((xt4_mb, xd3_mb, gen1, gen2), axis=0))
            temp.columns = genesym
            with HiddenPrints():
                act = progeny.run(temp, pmodel, center=True, num_perm=0, norm=True, scale=False)
            trueexp = act[:int(act.shape[0]/2)].values
            simulexp = act[int(act.shape[0]/2):].values
            total_pact_cor.append(np.mean([np.corrcoef(trueexp[s,:], simulexp[s,:])[0,1] for s in range(trueexp.shape[0])]))

    total_D_loss = np.mean(total_D_loss)
    total_G_loss = np.mean(total_G_loss)
    total_recon_loss = np.mean(total_recon_loss)
    total_triple_loss = np.mean(total_triple_loss)
    total_delta_loss = np.mean(total_delta_loss)
    total_pl_lengths = np.mean(total_pl_lengths)
    total_pl_penalty = np.mean(total_pl_penalty)
    total_pact_cor = np.mean(total_pact_cor) if genesym is not None else 0

    return total_D_loss, total_G_loss, total_recon_loss, total_triple_loss, total_delta_loss, total_pl_lengths, total_pl_penalty, total_pact_cor

def run_val(sess, model, val_dataset, batch_size=32, RFE=None, genesym=None, pmodel=None):
    total_D_loss, total_G_loss, total_recon_loss, total_triple_loss, total_delta_loss, total_pl_lengths, total_pl_penalty, total_pact_cor = [], [], [], [], [], [], [], []
    for step in range(val_dataset.data_len // (batch_size*9)):
        xt1_mb, xt2_mb, xt3_mb, xt4_mb, xt1c_mb, xt2c_mb, xt3c_mb, xt4c_mb = val_dataset.get_triple_l_sample(batch_size)
        xd1_mb, xd2_mb, xd3_mb, xd4_mb, xd1c_mb, xd2c_mb, xd3c_mb, xd4c_mb, xd1u_mb, xd2u_mb, xd3u_mb, xd4u_mb = val_dataset.get_delta_l_sample(batch_size)
        xdd1_mb, xdd2_mb, xdd3_mb = val_dataset.get_disc_l_sample(batch_size)
        x_mb = val_dataset.get(True, batch_size)
        z_mb = np.random.normal(0.0, scale = 1.0, size = (batch_size*9, model.z_dim))
        feed_dict = {model.t_term1:xt1_mb, model.t_term2:xt2_mb, model.t_term3:xt3_mb, model.t_term4:xt4_mb,
                     model.tc_term1:xt1c_mb, model.tc_term2:xt2c_mb, model.tc_term3:xt3c_mb, model.tc_term4:xt4c_mb,
                     model.d_term1:xd1_mb, model.d_term2:xd2_mb, model.d_term3:xd3_mb, model.d_term4:xd4_mb,
                     model.dc_term1:xd1c_mb, model.dc_term2:xd2c_mb, model.dc_term3:xd3c_mb, model.dc_term4:xd4c_mb,
                     model.du_term1:xd1u_mb, model.du_term2:xd2u_mb, model.du_term3:xd3u_mb, model.du_term4:xd4u_mb,
                     model.dd_term1:xdd1_mb, model.dd_term2:xdd2_mb, model.dd_term3:xdd3_mb,
                     model.no_term:x_mb, model.z:z_mb, model.batch_size:batch_size*9, model.is_training:False}
        D_loss, G_loss, recon_loss, triple_loss, delta_loss, pl_lengths, pl_penalty, gen1, gen2 = sess.run([model.D_raw_loss, model.G_raw_loss,
                                                                        model.recon_loss, model.triple_loss, model.delta_loss,
                                                                        model.pl_lengths, model.pl_penalty,
                                                                        model.gen_arith, model.gen_arith_fromdel],
                                                                       feed_dict=feed_dict)

        total_D_loss.append(D_loss)
        total_G_loss.append(G_loss)
        total_recon_loss.append(recon_loss)
        total_triple_loss.append(triple_loss)
        total_delta_loss.append(delta_loss)
        total_pl_lengths.append(pl_lengths)
        total_pl_penalty.append(pl_penalty)
        if genesym is not None:
            temp = pd.DataFrame(np.concatenate((xt4_mb, xd3_mb, gen1, gen2), axis=0))
            temp.columns = genesym
            with HiddenPrints():
                act = progeny.run(temp, pmodel, center=True, num_perm=0, norm=True, scale=False)
            trueexp = act[:int(act.shape[0]/2)].values
            simulexp = act[int(act.shape[0]/2):].values
            total_pact_cor.append(np.mean([np.corrcoef(trueexp[s,:], simulexp[s,:])[0,1] for s in range(trueexp.shape[0])]))

    total_D_loss = np.mean(total_D_loss)
    total_G_loss = np.mean(total_G_loss)
    total_recon_loss = np.mean(total_recon_loss)
    total_triple_loss = np.mean(total_triple_loss)
    total_delta_loss = np.mean(total_delta_loss)
    total_pl_lengths = np.mean(total_pl_lengths)
    total_pl_penalty = np.mean(total_pl_penalty)
    total_pact_cor = np.mean(total_pact_cor) if genesym is not None else 0

    if RFE is not None:
        z_mb = np.random.normal(0.0, scale = 1.0, size = (val_dataset.data_len, model.z_dim))
        feed_dict = {model.x_bulk:val_dataset.data, model.gennum:val_dataset.data_len, model.z:z_mb, model.is_training:False}
        recon_data, gen_data = sess.run([model.x_gen_bulk, model.x_gen_data], feed_dict=feed_dict)

        errors_d = list(RFE.fit(val_dataset.data, recon_data, output_AUC=False)['avg'])[0]
        errors_d_z = list(RFE.fit(val_dataset.data, gen_data, output_AUC=False)['avg'])[0]

    else:
        errors_d, errors_d_z = 0, 0

    return total_D_loss, total_G_loss, total_recon_loss, total_triple_loss, total_delta_loss, total_pl_lengths, total_pl_penalty, errors_d, errors_d_z, total_pact_cor

def main():
    # tb_path = "model"
    tb_path = "pgan_house/model_z1536_klnormcrvae_shxpr_vecont02mean_trideltemper5_nonoise_supcon10_delcell_delcontnoloss"
    if not os.path.exists(tb_path):
        os.mkdir(tb_path)
    # load data
    train_data = pd.read_csv('l1000_data/p1shp2xpr/training_data_p1.csv', index_col=None)
    train_data_info = train_data.iloc[:,:2].values
    train_data = train_data.iloc[:,2:].values
    train_data_sh = pd.read_csv('l1000_data/p1shp2xpr/training_data_sh_p1.csv', index_col=None)
    train_data_sh_info = train_data_sh.iloc[:,:2].values
    train_data_sh = train_data_sh.iloc[:,2:].values
    train_data2 = pd.read_csv('l1000_data/p1shp2xpr/training_data_p2.csv', index_col=None)
    train_data2_info = train_data2.iloc[:,:2].values
    train_data2 = train_data2.iloc[:,2:].values
    train_data2_xpr = pd.read_csv('l1000_data/p1shp2xpr/training_data_xpr_p2.csv', index_col=None)
    train_data2_xpr_info = train_data2_xpr.iloc[:,:2].values
    train_data2_xpr = train_data2_xpr.iloc[:,2:].values
    train_data_ve = pd.read_csv('l1000_data/p1shp2xpr/training_data_vehicle.csv', index_col=None)
    train_data_ve_info = train_data_ve.iloc[:,:2].values
    train_data_ve = train_data_ve.iloc[:,2:].values
    dmso = train_data_ve_info[:,1] == 'DMSO'
    train_data_ve = train_data_ve[dmso]
    train_data_ve_info = train_data_ve_info[dmso]
    train_data_ve_info[:,1] = 'control'
    veccls = train_data_ve_info[:,0].copy()
    #train_data_ve_info[:,0] = 'basal'

    # basal_data = pd.read_csv('l1000_data/basal_expression_total_qn.csv', index_col=None)
    # basal_data = basal_data.T.values
    # basal_info = np.array([['basal','control']]*basal_data.shape[0])

    val_data = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p1.csv', index_col=None)
    geneid = np.array(val_data.columns[2:]).astype(np.int)
    val_data_info = val_data.iloc[:,:2].values
    val_data = val_data.iloc[:,2:].values
    val_data2 = pd.read_csv('l1000_data/p1shp2xpr/validation_data_p2.csv', index_col=None)
    val_data2_info = val_data2.iloc[:,:2].values
    val_data2 = val_data2.iloc[:,2:].values
    val_data2_xpr = pd.read_csv('l1000_data/p1shp2xpr/validation_data_xpr_p2.csv', index_col=None)
    val_data2_xpr_info = val_data2_xpr.iloc[:,:2].values
    val_data2_xpr = val_data2_xpr.iloc[:,2:].values
    val_data_ve = pd.read_csv('l1000_data/p1shp2xpr/validation_data_vehicle.csv', index_col=None)
    val_data_ve_info = val_data_ve.iloc[:,:2].values
    val_data_ve = val_data_ve.iloc[:,2:].values
    dmso = val_data_ve_info[:,1] == 'DMSO'
    val_data_ve = val_data_ve[dmso]
    val_data_ve_info = val_data_ve_info[dmso]
    val_data_ve_info[:,1] = 'control'


    train_data_info = np.concatenate((train_data_info, train_data_sh_info, train_data2_info, train_data2_xpr_info, train_data_ve_info), axis=0)
    train_data = np.concatenate((train_data, train_data_sh, train_data2, train_data2_xpr, train_data_ve), axis=0)

    val_data_info = np.concatenate((val_data_info, val_data2_info, val_data2_xpr_info), axis=0)
    val_data = np.concatenate((val_data, val_data2, val_data2_xpr), axis=0)
    filtercells = np.intersect1d(np.unique(val_data_info[:,0]), np.unique(val_data_ve_info[:,0]))
    idxfilt = np.full(val_data.shape[0], False)
    for vc in filtercells:
        tcc = val_data_info[:,0] == vc
        idxfilt = idxfilt | tcc
    val_data_info = np.concatenate((val_data_info[idxfilt], val_data_ve_info), axis=0)
    val_data = np.concatenate((val_data[idxfilt], val_data_ve), axis=0)


    np.random.seed(0)
    train_subsample = np.random.choice(train_data.shape[0], 4000, replace=False)
    train_controlsample = np.where(train_data_info[:,1]=='control')[0]
    train_subsample = np.unique(np.concatenate((train_subsample, train_controlsample)))
    tdi = train_data_info[train_subsample,:]
    td = train_data[train_subsample,:]

    val_subsample = np.random.choice(val_data.shape[0], 6000, replace=False)
    val_controlsample = np.where(val_data_info[:,1]=='control')[0]
    val_subsample = np.unique(np.concatenate((val_subsample, val_controlsample)))
    val_data_info = val_data_info[val_subsample,:]
    val_data = val_data[val_subsample,:]

    geneinfo = pd.read_csv('l1000_data/GSE92742_Broad_LINCS_gene_info.txt', sep='\t')
    genemapper = pd.Series(data=geneinfo['pr_gene_symbol'].values, index=geneinfo['pr_gene_id'].values)
    genesym = pd.Series(geneid).map(genemapper).values
    pmodel = progeny.load_model(organism='Human', top=2300)

    train_dataset = Batch_maker(train_data, train_data_info, triweight=True)
    tdd = Batch_maker(td, tdi)
    val_dataset = Batch_maker(val_data, val_data_info)

    # training params
    batch_size = 32 
    n_epoch = 4000
    val_freq= 50
    save_freq= 50
    
    # model params
    mp = {
        'z_dim':1536,
        'learning_rate':1e-4,
        'dropout_rate':0.2,
        'lamb_gp':10,
        'lamb_recon':100,
        'lamb_triple':100,
        'lamb_delta':100,
        'Diters':5,
        'enc_unit1':1024,
        'enc_unit2':512,
        'enc_unit3':256,
        'gen_unit1':512,
        'gen_unit2':512,
        'gen_unit3':1024,
        'disc_unit1':1024,
        'disc_unit2':512,
        'disc_unit3':256
        }

    model = PGAN(x_dim=train_data.shape[1], **mp)

    RFE = RandomForestError()

    saver = tf.compat.v1.train.Saver(max_to_keep=5)

    seconfig = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))#allow_soft_placement = True)
    seconfig.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=seconfig)

    sess.run(tf.compat.v1.global_variables_initializer())

    summary_writer = tf.compat.v1.summary.FileWriter(tb_path,sess.graph)

    for epoch in range(n_epoch+1):
        st = time()
        d_loss, g_loss, recon_loss, triple_loss, delta_loss, pl_lengths, pl_penalty, pact = run_train(sess, model, train_dataset, batch_size, genesym, pmodel)

        summ = tf.compat.v1.Summary()
        summ.value.add(tag='train/train_D_loss', simple_value=d_loss)
        summ.value.add(tag='train/train_G_loss', simple_value=g_loss)
        summ.value.add(tag='train/train_recon_loss', simple_value=recon_loss)
        summ.value.add(tag='train/train_triple_loss', simple_value=triple_loss)
        summ.value.add(tag='train/train_delta_loss', simple_value=delta_loss)
        summ.value.add(tag='train/train_pl_lengths', simple_value=pl_lengths)
        summ.value.add(tag='train/train_pl_penalty', simple_value=pl_penalty)
        summ.value.add(tag='train/train_pact_cor', simple_value=pact)
        summary_writer.add_summary(summ, epoch)

        if epoch % val_freq == 0:
            d_loss_trainsub, g_loss_train_sub, recon_loss_train_sub, triple_loss_train_sub, delta_loss_train_sub, pl_lengths_train, pl_penalty_train, errors_d_train, errors_d_z_train, pact_train = run_val(sess, model, tdd, batch_size, RFE, genesym, pmodel)
            d_loss_val, g_loss_val, recon_loss_val, triple_loss_val, delta_loss_val, pl_lengths_val, pl_penalty_val, errors_d, errors_d_z, pact_val = run_val(sess, model, val_dataset, batch_size, RFE, genesym, pmodel)

            summ = tf.compat.v1.Summary()
            summ.value.add(tag='train_sub/train_sub_D_loss', simple_value=d_loss_trainsub)
            summ.value.add(tag='train_sub/train_sub_G_loss', simple_value=g_loss_train_sub)
            summ.value.add(tag='train_sub/train_sub_recon_loss', simple_value=recon_loss_train_sub)
            summ.value.add(tag='train_sub/train_sub_triple_loss', simple_value=triple_loss_train_sub)
            summ.value.add(tag='train_sub/train_sub_delta_loss', simple_value=delta_loss_train_sub)
            summ.value.add(tag='train_sub/train_sub_pl_lengths', simple_value=pl_lengths_train)
            summ.value.add(tag='train_sub/train_sub_pl_penalty', simple_value=pl_penalty_train)
            summ.value.add(tag='train_sub/train_sub_RFE', simple_value=errors_d_train)
            summ.value.add(tag='train_sub/train_sub_RFE_z', simple_value=errors_d_z_train)
            summ.value.add(tag='train_sub/train_sub_pact_cor', simple_value=pact_train)
            summ.value.add(tag='val/val_D_loss', simple_value=d_loss_val)
            summ.value.add(tag='val/val_G_loss', simple_value=g_loss_val)
            summ.value.add(tag='val/val_recon_loss', simple_value=recon_loss_val)
            summ.value.add(tag='val/val_triple_loss', simple_value=triple_loss_val)
            summ.value.add(tag='val/val_delta_loss', simple_value=delta_loss_val)
            summ.value.add(tag='val/val_pl_lengths', simple_value=pl_lengths_val)
            summ.value.add(tag='val/val_pl_penalty', simple_value=pl_penalty_val)
            summ.value.add(tag='val/val_RFE', simple_value=errors_d)
            summ.value.add(tag='val/val_RFE_z', simple_value=errors_d_z)
            summ.value.add(tag='val/val_pact_cor', simple_value=pact_val)
            summary_writer.add_summary(summ, epoch)
            
            print('epoch',epoch,'/',n_epoch,' - train sub D loss:', d_loss_trainsub,
                  ', train sub G loss:', g_loss_train_sub, ', train sub errors_d', errors_d_train, ', train sub errors_d_z', errors_d_z_train, ', train sub pact', pact_train)
            print('epoch',epoch,'/',n_epoch,' - val D loss:', d_loss_val, ', val G loss:', g_loss_val, ', val errors_d', errors_d, ', val errors_d_z', errors_d_z, ', val pact', pact_val)

        print('epoch',epoch,'/',n_epoch,' - D loss:', d_loss, ', G loss:', g_loss, 'Elapsed..', time()-st)
        sys.stdout.flush()
        if epoch % save_freq == 0:
            saver.save(sess, tb_path+'/model-'+str(epoch)+'.ckpt')


if __name__ == "__main__":
    main()
