import numpy as np
import tensorflow as tf

class optRBFSampler:
    """
    The random nodes have the form
    (1/sqrt(q(w)))cos(sqrt(gamma)*w dot x),
    (1/sqrt(q(w)))sin(sqrt(gamma)*w dot x).
    q(w) is the optimized density of features with respect to the initial
    feature distribution determined by the RBF kernel and data distribution.
    Without applying reweight method, this class provides a uniform sampling
    of random features.
    """
    def __init__(self,
                 n_old_features,
                 feature_pool_size,
                 gamma=1,
                 n_new_features=20):
        self.name = 'opt_rbf'
        self.pool = (np.random.randn(n_old_features,
                                     feature_pool_size)
                    * np.sqrt(gamma))
        self.feature_pool_size = feature_pool_size
        self.gamma = gamma
        self.n_new_features = n_new_features
        Weight = np.ones(feature_pool_size)
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(feature_pool_size,
                                size=n_new_features,
                                p=self.Prob)
        self.sampler = self.pool[:,self.feature_list]

    def reweight(self, X, X_pool_fraction, Lambda=1):
        ### calculate weight and resample the features from pool
        m = len(X)
        feature_pool_size = self.feature_pool_size
        X_pool_size = min(int(m * X_pool_fraction),500)
        T = np.empty((X_pool_size,feature_pool_size*2))
        k = np.random.randint(m,size=X_pool_size)
        X_pool = X[k,:]
        A = X_pool.dot(self.pool)
        T[:,:feature_pool_size] = np.cos(A)
        T[:,feature_pool_size:] = np.sin(A)
        U,s,V = np.linalg.svd(T, full_matrices=False)
        Trace = s**2 / (s**2 + Lambda * X_pool_size * feature_pool_size)
        Weight = np.empty(feature_pool_size*2)
        for idx in range(feature_pool_size*2):
        # V is actually V.T in standard notation
            Weight[idx] = V[:,idx].dot(Trace * V[:,idx])
        Weight = Weight[:feature_pool_size] + Weight[feature_pool_size:]
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(feature_pool_size,
                                             size=self.n_new_features,
                                             p=self.Prob)
        self.sampler = self.pool[:,self.feature_list]

    def update(self, idx):
        n = np.random.choice(self.pool.shape[1],p=self.Prob)
        self.sampler[:,idx] = self.pool[:,n]
        return 1

    def fit_transform(self, X):
        X_tilc = np.cos(X.dot(self.sampler))
        X_tils = np.sin(X.dot(self.sampler))
        X_til = np.concatenate((X_tilc,X_tils),axis=-1)
        return X_til / np.sqrt(self.n_new_features)

class optReLUSampler:
    """
    The random nodes have the form
    (1/sqrt(q(w)))ReLU(w dot x).
    q(w) is the optimized density of features with respect to the initial
    feature distribution determined by data distribution.
    Without applying reweight method, this class provides a uniform sampling
    of random features.
    """
    def __init__(self, n_old, n_pool, n_new=20):
        self.name = 'opt_relu'
        self.pool = np.random.randn(n_old + 1, n_pool)
        self.pool = self.pool / np.linalg.norm(self.pool,axis=0)
        self.n_pool = n_pool
        self.n_new = n_new
        self.Weight = np.ones(n_pool)
        self.Prob = self.Weight / np.sum(self.Weight)
        self.feature_list = np.random.choice(n_pool,
                                size=n_new,
                                p=self.Prob,
                                replace=False)
        self.sampler = self.pool[:,self.feature_list]

    def reweight(self, X, n_Xpool=500):
        ### calculate weight and resample the features from pool
        m = len(X)
        T = np.empty((n_Xpool,self.n_pool))
        k = np.random.randint(m,size=n_Xpool)
        Xpool = X[k,:]
        Xpool = np.concatenate((Xpool,np.ones((n_Xpool,1))),axis=1)
        T = np.maximum(0,Xpool.dot(self.pool))
        U,s,V = np.linalg.svd(T, full_matrices=False)
        # Trace = s**2 / (s**2 + s[int(len(s)/2)]**2)
        Weight = np.empty(self.n_pool)
        for idx in range(self.n_pool):
            Weight[idx] = np.argmax(V[:,idx])
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(self.n_pool,
                                             size=self.n_new,
                                             p=self.Prob,
                                             replace=False)
        self.sampler = self.pool[:,self.feature_list]

    def update(self, idx):
        n = np.random.choice(self.pool.shape[1],p=self.Prob)
        self.sampler[:,idx] = self.pool[:,n]
        return 1

    def fit_transform(self, X):
        X = np.concatenate((X,np.ones((len(X),1))),axis=1)
        X_til = np.maximum(0,X.dot(self.sampler))
        # X_til = X_til / np.sqrt(self.Prob[self.feature_list])
        return X_til / np.sqrt(self.n_new)

class RF:
    """
    This is a class constructing a 2-layer net with Fourier or
    ReLU nodes in the hidden layer. The weights in the first layer is
    initialized using random Gaussian or random uniform features,
    respectively. Layerwise training can be applied.
    """
    def __init__(self,feature,n_old_features,
        n_new_features,classes,Lambda=0.,Gamma=1.,
        loss_fn='log',log=False,initializer=None):
        self._initializer = initializer
        self._feature = feature
        self._d = n_old_features
        self._N = n_new_features
        self._Lambda = Lambda
        self._Gamma = Gamma
        self._classes = classes
        self._loss_fn = loss_fn
        self.log = log
        self._total_iter = 0
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        if self._model_fn() == 0:
            raise ValueError

    @property
    def params(self):
        return self.get_params()

    def _feature_layer(self,x):
        N = self._N
        d = self._d
        if self._feature == 'Gaussian':
            if self._initializer == None:
                # Gamma is only required by random Fourier for Gaussian kernel
                k_initializer = tf.random_normal_initializer(stddev =
                    np.sqrt(self._Gamma))
            else:
                k_initializer = tf.constant_initializer(self._initializer,
                    dtype=tf.float32)
            # random fourier features requires bias to be uniform in [0,pi]
            b_initializer = tf.random_uniform_initializer(minval=0.,maxval=np.pi)
            activation_node = tf.cos
        elif self._feature == 'ReLU':
            k_initializer = np.random.randn(d+1,N)
            # initialize (w_i,b_i) by vectors uniform over unit sphere
            k_initializer = k_initializer / np.linalg.norm(k_initializer,axis=0)
            b_initializer = tf.constant_initializer(k_initializer[-1,:],
                dtype=tf.float32)
            k_initializer = tf.constant_initializer(k_initializer[:-1,:],
                dtype=tf.float32)
            activation_node = tf.nn.relu
        trans_layer = tf.layers.dense(inputs=x,units=N,
            use_bias=True,
            kernel_initializer=k_initializer,
            bias_initializer=b_initializer,
            activation=activation_node,
            name='RF')
        RF_layer = tf.div(trans_layer,tf.sqrt(N*1.0))
        tf.add_to_collection('Hidden',RF_layer)
        tf.summary.histogram('inner weights',
            self._graph.get_tensor_by_name('RF/kernel:0'))
        return RF_layer

    def _output_layer(self,x,n_outputs):
        N = self._N
        np_init = np.random.choice([-1,1],size=(N,n_outputs))
        logits_init = tf.constant_initializer(np_init,dtype=tf.float32)
        logits = tf.layers.dense(inputs=x,
            use_bias = False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=
                self._Lambda),
            kernel_initializer=logits_init,
            units=n_outputs,name='Logits')
        tf.summary.histogram('outer weights',
            self._graph.get_tensor_by_name('Logits/kernel:0'))
        return logits

    def _model_fn(self):
        d = self._d
        N = self._N
        Lambda = self._Lambda
        Gamma = self._Gamma
        n_classes = len(self._classes)
        loss_fn = self._loss_fn
        with self._graph.as_default():
            global_step_1 = tf.Variable(0,trainable=False,name='global1')
            global_step_2 = tf.Variable(0,trainable=False,name='global2')
            # note that the shape of x and y are not aligned,
            # this will be addressed before computing loss by
            # one_hot for log loss or reshape for hinge and squared loss
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')
            RF_layer = self._feature_layer(x)
            if self._loss_fn in ('hinge','squared'):
                if n_classes == 2:
                    logits = self._output_layer(RF_layer,1)
                    logits = tf.reshape(logits,shape=[-1])
                else:
                    print("hinge or squared loss only works for binary classificaiton.")
                    return 0
            elif self._loss_fn == 'log':
                logits = self._output_layer(RF_layer,n_classes)
                probab = tf.nn.softmax(logits, name="softmax")
                tf.add_to_collection("Output",probab)
            tf.add_to_collection("Output",logits)
            regularizer = tf.losses.get_regularization_loss(scope='Logits')
            if self._loss_fn == 'hinge':
                reg_loss = tf.losses.hinge_loss(labels=y,
                    logits=logits) + regularizer
            elif self._loss_fn == 'squared':
                reg_loss = tf.losses.mean_squared_error(labels=y,
                    predictions=logits) + regularizer
            elif self._loss_fn == 'log':
                onehot_labels = tf.one_hot(indices=y, depth=n_classes)
                loss_log = tf.losses.softmax_cross_entropy(
                    onehot_labels=onehot_labels, logits=logits)
                reg_loss = loss_log + regularizer
            tf.add_to_collection('Output',reg_loss)

            merged = tf.summary.merge_all()
            tf.add_to_collection('Summary',merged)
            self._sess.run(tf.global_variables_initializer())
        return 1

    def predict(self,data,batch_size=50):
        with self._graph.as_default():
            f_vec = tf.get_collection('Hidden')[0]
            if self._loss_fn in ('hinge','squared'):
                logits = tf.get_collection('Output')[0]
                predictions = {"indices": logits,
                    "feature_vec": f_vec}
            elif self._loss_fn == 'log':
                probab,logits,_ = tf.get_collection('Output')
                predictions = {
                    "indices": tf.argmax(input=logits,axis=1),
                    "probabilities": probab,
                    "feature_vec": f_vec}
        classes = []
        probabilities = []
        sparsity = 0
        idx = 0
        while idx < len(data):
            t = idx + batch_size
            batch = data[idx:t,:]
            batch.reshape(len(batch),-1)
            idx = t
            feed_dict = {'features:0':batch}
            results = self._sess.run(predictions,feed_dict=feed_dict)
            if self._loss_fn == 'log':
                classes.extend([self._classes[index] for index in results['indices']])
                probabilities.extend(results['probabilities'])
            elif self._loss_fn == 'hinge':
                classes.extend([self._classes[index>0] for index in results['indices']])
            elif self._loss_fn == 'squared':
                classes.extend([self._classes[index>.5] for index in results['indices']])
            feature_vec = results['feature_vec']
            sparsity += np.count_nonzero(feature_vec)/feature_vec.shape[1]
        sparsity = sparsity / len(data)
        return classes,probabilities,sparsity

    def score(self,data,labels):
        predictions,_,_ = self.predict(data)
        accuracy = sum(predictions==labels) / len(data)
        return accuracy

    def fit(self,data,labels,mode='layer 2',
        opt_method='sgd',opt_rate=10.,
        batch_size=200,n_epoch=5,bd=100):
        indices = [self._classes.index(label) for label in labels]
        indices = np.array(indices)
        with self._graph.as_default():
            in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'RF')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Logits')
            loss = tf.get_collection('Output')[-1]
            global_step_1 = self._graph.get_tensor_by_name('global1:0')
            global_step_2 = self._graph.get_tensor_by_name('global2:0')
            merged = tf.get_collection('Summary')[0]
            if opt_method == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=opt_rate)
            elif opt_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=
                    opt_rate)
            if mode == 'layer 2':
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_2,
                    var_list=out_weights
                )
                if self._Lambda == 0:
                    clip_op = clip_by_maxnorm(out_weights[0],bd)
            # elif mode == 'layer 1':
            #     train_op = optimizer.minimize(
            #         loss=loss,
            #         global_step=global_step_1,
            #         var_list=in_weights
            #     )
            elif mode == 'over all':
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    )

            # initialize global variables in optimizer
            self._sess.run(tf.global_variables_initializer())
        if self.log:
            self._train_writer = tf.summary.FileWriter('tmp',
                tf.get_default_graph())
        for idx in range(n_epoch):
            rand_indices = np.random.permutation(len(data)) - 1
            for jdx in range(len(data)//batch_size):
                batch_indices = rand_indices[jdx*batch_size:(jdx+1)*batch_size]
                feed_dict = {'features:0':data[batch_indices,:],
                             'labels:0':indices[batch_indices]}
                if jdx % 10 == 1:
                    print('epoch: {2:d}, iter: {0:d}, loss: {1:.4f}'.format(
                        jdx, self._sess.run(loss,feed_dict),idx))
                    if self.log:
                        summary = self._sess.run(merged)
                        self._train_writer.add_summary(summary,self._total_iter)
                self._sess.run(train_op,feed_dict)
                if mode == 'layer 2' and self._Lambda == 0:
                    self._sess.run(clip_op)
                self._total_iter += 1

    def get_params(self,deep=False):
        params = {
            'feature': self._feature,
            'n_old_features': self._d,
            'n_new_features': self._N,
            'Lambda': self._Lambda,
            'Gamma': self._Gamma,
            'classes': self._classes,
            'loss_fn': self._loss_fn
        }
        return params

    def __del__(self):
        self._sess.close()
        print('Session is closed.')

def gamma_est(X,portion = 0.3):
    """
    returns 1/average squared distance among data points.
    """
    s = 0
    n = int(X.shape[0]*portion)
    if n > 200:
        n = 200
    for idx in range(n):
        for jdx in range(n):
            s = s+np.linalg.norm(X[idx,:]-X[jdx,:])**2
    return n**2/s

def clip_by_maxnorm(t,c):
    cc = c*tf.ones(shape=tf.shape(t))
    nt = tf.norm(t,ord=np.inf,axis=0)
    factor = tf.divide(tf.minimum(nt,cc),nt)
    t1 = tf.multiply(t,factor)
    return tf.assign(t,t1)
