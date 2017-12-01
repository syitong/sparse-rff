import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def tfRF2L(n_old_features,n_components,Lambda,Gamma,n_classes):
    d = n_old_features
    N = n_components

    x = tf.placeholder(dtype=tf.float32,
        shape=[None,d],name='features')
    y = tf.placeholder(dtype=tf.int,
        shape=[None,1],name='labels')

    initializer = tf.random_normal_initializer(
        stddev=tf.sqrt(Gamma.astype(np.float32)))
    with tf.name_scope('first'):
        cos_layer = tf.layers.dense(inputs=x,
            units=2*N,activation=tf.cos,use_bias=False,
            kernel_initializer=initializer)
        sin_layer = tf.layers.dense(inputs=x,
            units=2*N,activation=tf.sin,use_bias=False,
            kernel_initializer=initializer)
        RF_layer = tf.div(tf.concat([cos_layer,sin_layer],axis=1),
            tf.sqrt(N*1.0))
        in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            'RF_Layer')
    logits = tf.layers.dense(inputs=RF_layer,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
        units=n_classes,name='logits')
    out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        'logits')
