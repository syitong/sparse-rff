import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class tfRF2L:
    """
    This is class constructs a 2-layer net with cos and sin nodes
    in the hidden layer. The weights in the first layer is
    initialized using random Gaussian features.
    Layerwise training can be applied.
    """
    def __init__(self,n_old_features,
        n_components,Lambda,Gamma,n_classes,
        loss_fn):
        self._d = n_old_features
        self._N = n_components
        self._Lambda = Lambda
        self._Gamma = Gamma
        self._n_classes = n_classes
        self._loss_fn = loss_fn
        self._predict = None
        self._train_op_1 = None
        self._train_op_2 = None

    def _model_fn(self):
        d = self._d
        N = self._N
        Lambda = self._Lambda
        Gamma = self._Gamma
        n_classes = self._n_classes
        loss_fn = self._loss_fn

        with name_scope('inputs'):
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.int,
                shape=[None,1],name='labels')

        with name_scope('Layer1'):
            initializer = tf.random_normal_initializer(
                stddev=tf.sqrt(Gamma.astype(np.float32)))

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

        with name_scope('Layer2'):
            logits = tf.layers.dense(inputs=RF_layer,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                units=n_classes,name='logits')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'logits')

        if loss_fn == 'hinge':
            probab = None
        elif loss_fn == 'log':
            probab = tf.nn.softmax(logits, name="softmax_tensor")
        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": probab}

        if mode == 'predict':
            return predictions[pred_keys]

        # Calculate Loss (for both TRAIN and EVAL modes)
        if loss_fn == 'hinge':
            loss = tf.losses.hinge_loss(
                labels=labels,
                logits=logits
            )
        elif loss_fn == 'log':
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.uint8),
                depth=n_classes)
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == 'train':
            return {'loss': loss,
                'inner weights': in_weights,
                'outer weights': out_weights}

        return tf.metrics.accuracy(labels=labels,
            predictions=predictions["classes"])

            global_step = tf.Variable(0,trainable=False)
            learning_rate = tf.train.inverse_time_decay(
                learning_rate=1.,
                decay_steps=1,
                global_step=global_step,
                decay_rate=1.)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
             #   l2_regularization_strength=0.)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step(),
                var_list=out_weights
            )
            return train_op
