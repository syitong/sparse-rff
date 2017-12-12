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
        loss_fn='log loss'):
        self._d = n_old_features
        self._N = n_components
        self._Lambda = np.float32(Lambda)
        self._Gamma = np.float32(Gamma)
        self._n_classes = n_classes
        self._loss_fn = loss_fn
        self._total_iter = 0
        self._sess = tf.Session()
        self._model_fn()
        self._sess.run(tf.global_variables_initializer())

    @property
    def d(self):
        return self._d
    @property
    def N(self):
        return self._N
    @property
    def Lambda(self):
        return self._Lambda
    @property
    def Gamma(self):
        return self._Gamma
    @property
    def n_classes(self):
        return self._n_classes
    @property
    def loss_fn(self):
        return self._loss_fn
    @property
    def global_step_layer_1(self):
        return self._sess.run(_global_step_layer_1)
    @property
    def global_step_layer_2(self):
        return self._sess.run(_global_step_layer_2)

    def _model_fn(self):
        d = self._d
        N = self._N
        Lambda = self._Lambda
        Gamma = self._Gamma
        n_classes = self._n_classes
        loss_fn = self._loss_fn

        with self._sess.graph.as_default():
            g = self._sess.graph
            global_step_1 = tf.Variable(0,trainable=False,name='global1')
            global_step_2 = tf.Variable(0,trainable=False,name='global2')
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')

            with tf.name_scope('RF_layer'):
                initializer = tf.random_normal_initializer(
                    stddev=tf.sqrt(Gamma))

                trans_layer = tf.layers.dense(inputs=x,units=N,
                    use_bias=False,
                    kernel_initializer=initializer,
                    name='Gaussian')
                self._sess.run(tf.global_variables_initializer())

                cos_layer = tf.cos(trans_layer)
                sin_layer = tf.sin(trans_layer)
                concated = tf.concat([cos_layer,sin_layer],axis=1)
                RF_layer = tf.div(concated,tf.sqrt(N*1.0))
                tf.summary.histogram('inner weights',
                    g.get_tensor_by_name('Gaussian/kernel:0'))

            logits = tf.layers.dense(inputs=RF_layer,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                units=n_classes,name='Logits')
            tf.add_to_collection("Probab",logits)
            tf.summary.histogram('outer weights',
                g.get_tensor_by_name('Logits/kernel:0'))

            probab = tf.nn.softmax(logits, name="softmax")
            tf.add_to_collection("Probab",probab)

            # hinge loss only works for binary classification.
            regularizer = tf.losses.get_regularization_loss(scope='Logits')
            # if self._n_classes == 2:
            #     loss_hinge = tf.losses.hinge_loss(labels=y,
            #         logits=logits,
            #         loss_collection="loss") + regularizer
            #     loss_ramp = (tf.max(loss_hinge,1)
            #         + regularizer)
            #     tf.add_to_collection("loss",loss_ramp)
            onehot_labels = tf.one_hot(indices=tf.cast(y, tf.uint8),
                depth=n_classes)
            loss_log = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)
            reg_log_loss = tf.add(loss_log,regularizer)
            tf.add_to_collection('Loss',reg_log_loss)

            merged = tf.summary.merge_all()
            tf.add_to_collection('Summary',merged)
            self._train_writer = tf.summary.FileWriter('tmp',
                tf.get_default_graph())
            self._sess.run(tf.global_variables_initializer())
            summary = self._sess.run(merged)
            self._train_writer.add_summary(summary)

    def predict(self,data):
        with self._sess.graph.as_default():
            g = self._sess.graph
            feed_dict = {'features:0':data}
            logits,probab = tf.get_collection('Probab')
            predictions = {
                # Generate predictions (for PREDICT and EVAL mode)
                "classes": tf.argmax(input=logits,axis=1),
                # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
                # `logging_hook`.
                "probabilities": probab}
            return self._sess.run(predictions,feed_dict=feed_dict)

    def fit(self,data,labels,mode,n_iter):
        with self._sess.graph.as_default():
            g = self._sess.graph
            in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Gaussian')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Logits')
            # if self._n_classes == 2:
            #     h_loss,r_loss,l_loss = tf.get_collection('loss')
            # else:
            #     l_loss = tf.get_collection('loss')
            # if self._loss_fn == 'hinge loss':
            #     loss = h_loss
            # elif self._loss_fn == 'log loss':
            #     loss = l_loss
            loss = tf.get_collection('Loss')[0]
            global_step_1 = g.get_tensor_by_name('global1:0')
            global_step_2 = g.get_tensor_by_name('global2:0')
            merged = tf.get_collection('Summary')[0]
            if mode == 'layer 2':
                learning_rate = tf.train.inverse_time_decay(
                    learning_rate=1.,
                    decay_steps=1,
                    global_step=global_step_2,
                    decay_rate=1.)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                 #   l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_2,
                    var_list=out_weights
                )
            if mode == 'layer 1':
                learning_rate = tf.train.inverse_time_decay(
                    learning_rate=1.,
                    decay_steps=1,
                    global_step=global_step_1,
                    decay_rate=1.)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    var_list=in_weights
                )
            if mode == 'over all':
                learning_rate = tf.train.inverse_time_decay(
                    learning_rate=1.,
                    decay_steps=1,
                    global_step=global_step_1,
                    decay_rate=1.)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                # optimizer = tf.train.FtrlOptimizer(learning_rate=50,
                #     l2_regularization_strength=0.)
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    # var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    )

            for idx in range(n_iter):
                rand_i = np.random.randint(len(data))
                feed_dict = {'features:0':data[None,rand_i,:],
                             'labels:0':labels[None,rand_i]}
                if idx % 10 == 1:
                    print('loss: {0:.4f}'.format(self._sess.run(loss,feed_dict)))
                    summary = self._sess.run(merged)
                    self._train_writer.add_summary(summary,self._total_iter)
                self._sess.run(train_op,feed_dict)
                self._total_iter += 1
            print('loss: {0:.4f}'.format(self._sess.run(loss,feed_dict)))
            summary = self._sess.run(merged)
            self._train_writer.add_summary(summary,idx)

    def close(self):
        self._sess.close()

    def get_params(self):
        params = {
            'n_old_features': self._d,
            'n_components': self._N,
            'Lambda': self._Lambda,
            'Gamma': self._Gamma,
            'n_classes': self._n_classes,
            'loss_fn': self._loss_fn
        }
        return params
