# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow.contrib as tf_contrib
import cv2 as cv
import random
import os
import ops

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True, scope=scope)

def batch_norm(x, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True, scope=scope)

def flatten(x) :
    return tf.layers.flatten(x)

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha) 

def relu(x):
    return tf.nn.relu(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def resblock(x_init, c, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = slim.conv2d(x_init, c, kernel_size=[3,3], stride=1, activation_fn = None)
            x = batch_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = slim.conv2d(x, c, kernel_size=[3,3], stride=1, activation_fn = None)
            x = batch_norm(x)

        return x + x_init
    
def conv(x, c):
    x1 = slim.conv2d(x, c, kernel_size=[5,5], stride=2, padding = 'SAME', activation_fn=relu)
#    print(x1.shape)
    x2 = slim.conv2d(x, c, kernel_size=[3,3], stride=2, padding = 'SAME', activation_fn=relu)
#    print(x2.shape)
    x3 = slim.conv2d(x, c, kernel_size=[1,1], stride=2, padding = 'SAME', activation_fn=relu)
#    print(x3.shape)
    out = tf.concat([x1, x2, x3],axis = 3)
    out = slim.conv2d(out, c, kernel_size=[1,1], stride=1, padding = 'SAME', activation_fn=None)
#    print(out.shape)
    return out

def mixgenerator(x_init, c, org_pose, trg_pose):    
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.variable_scope('generator', reuse = reuse):
        org_pose = tf.cast(tf.reshape(org_pose, shape=[-1, 1, 1, org_pose.shape[-1]]), tf.float32)
        print(org_pose.shape)
        org_pose = tf.tile(org_pose, [1, x_init.shape[1], x_init.shape[2], 1])
        print(org_pose.shape)
        x = tf.concat([x_init, org_pose], axis=-1)
        print(x.shape)
                        
        x = conv(x, c)
        x = batch_norm(x, scope='bat_norm_1')
        x = relu(x)#64
        print('----------------')
        print(x.shape)
        
        x = conv(x, c*2)
        x = batch_norm(x, scope='bat_norm_2')
        x = relu(x)#32
        print(x.shape)
        
        x = conv(x, c*4)
        x = batch_norm(x, scope='bat_norm_3')
        x = relu(x)#16
        print(x.shape)
        
        f_org = x
        
        x = conv(x, c*8)
        x = batch_norm(x, scope='bat_norm_4')
        x = relu(x)#8
        print(x.shape)

        x = conv(x, c*8)
        x = batch_norm(x, scope='bat_norm_5')
        x = relu(x)#4
        print(x.shape)

        
        for i in range(6):
            x = resblock(x, c*8, scope = str(i)+"_resblock")
        
        trg_pose = tf.cast(tf.reshape(trg_pose, shape=[-1, 1, 1, trg_pose.shape[-1]]), tf.float32)
        print(trg_pose.shape)
        trg_pose = tf.tile(trg_pose, [1, x.shape[1], x.shape[2], 1])
        print(trg_pose.shape)
        x = tf.concat([x, trg_pose], axis=-1)
        print(x.shape)

        x = slim.conv2d_transpose(x, c*8, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_8')
        x = relu(x)#8
        print(x.shape)
        
        x = slim.conv2d_transpose(x, c*4, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_9')
        x = relu(x)#16
        print(x.shape)
        
        f_trg =x

        x = slim.conv2d_transpose(x, c*2, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_10')
        x = relu(x)#32
        print(x.shape)

        x = slim.conv2d_transpose(x, c, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_11')
        x = relu(x)#64
        print(x.shape)

        z = slim.conv2d_transpose(x, 3 , kernel_size=[3,3], stride=2, activation_fn = tf.nn.tanh)
        f = tf.concat([f_org, f_trg], axis=-1)
        print(f.shape)
        return z, f  

def snpixdiscriminator(x_init):

    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0

    with tf.variable_scope('discriminator', reuse=reuse):

        x = ops.snconv(x_init, 32, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_7')

        x = batch_norm(x, scope='bat_norm_21')

        x = lrelu(x, 0.2)

        print('D---------------- ------------------D')

        print(x.shape)#64

        

        x = ops.snconv(x, 64, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_8')

        x = batch_norm(x, scope='bat_norm_22')

        x = lrelu(x, 0.2)

        print(x.shape)#32

        

        x = ops.snconv(x, 64, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_9')

        x = batch_norm(x, scope='bat_norm_23')

        x = lrelu(x, 0.2)

        print(x.shape)#16

        

        x = ops.snconv(x, 128, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_10')

        x = batch_norm(x, scope='bat_norm_24')

        x = lrelu(x, 0.2)

        print(x.shape)#8



        x = ops.snconv(x, 128, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_11')

        x = batch_norm(x, scope='bat_norm_25')

        x = lrelu(x, 0.2)

        print(x.shape)  #4

        

        x = ops.snconv(x, 256, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_12')

        x = batch_norm(x, scope='bat_norm_26')

        x = lrelu(x, 0.2)

        print(x.shape)  #2 

        

        x = ops.snconv(x, 256, kernel=3, stride=2, pad=1, pad_type='zero', use_bias=True, sn=True, scope='conv_13')

        x = batch_norm(x, scope='bat_norm_27')

        x = lrelu(x, 0.2)#1

        print(x.shape)

        

        logit = ops.snconv(x, 1, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=True, sn=True, scope='conv_14')

        print(logit.shape)



        pose_logit = ops.snconv(x, 9, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_15')        

        pose_logit = tf.reshape(pose_logit, [-1,9])
        print(pose_logit.shape)
        

        return tf.nn.sigmoid(logit), logit, pose_logit 

def generator(x, c, trg_pose):    
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator_1')]) > 0
    with tf.variable_scope('generator_1', reuse = reuse):
        trg_pose = tf.cast(tf.reshape(trg_pose, shape=[-1, 1, 1, trg_pose.shape[-1]]), tf.float32)
        print(trg_pose.shape)
        trg_pose = tf.tile(trg_pose, [1, x.shape[1], x.shape[2], 1])
        print(trg_pose.shape)
        x = tf.concat([x, trg_pose], axis=-1)
        print(x.shape)
        
        x = slim.conv2d(x, c*8, kernel_size=[3,3], stride=1, padding = 'SAME', activation_fn=None)
        x = batch_norm(x, scope='bat_norm_28')
        x = relu(x)#16
        
        for i in range(6):
            x = resblock(x, c*8, scope = str(i)+"_gresblock")
        
        x = slim.conv2d(x, c*8, kernel_size=[3,3], stride=2, padding = 'SAME', activation_fn=None)
        x = batch_norm(x, scope='bat_norm_29')
        x = relu(x)#8
        
        x = slim.conv2d_transpose(x, c*4, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_30')
        x = relu(x)#16
        print(x.shape)
        
        x = slim.conv2d_transpose(x, c*2, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_31')
        x = relu(x)#32
        print(x.shape)

        x = slim.conv2d_transpose(x, c, kernel_size=[3, 3], stride=2, activation_fn=None)
        x = batch_norm(x, scope='bat_norm_32')
        x = relu(x)#64
        print(x.shape)

        z = slim.conv2d_transpose(x, 3 , kernel_size=[3,3], stride=2, activation_fn = tf.nn.tanh)
        
        return z  
        
        