import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import tensorflow as tf

class optRBFSampler:
    """
    The random nodes have the form
    (1/sqrt(q(w)))cos(sqrt(gamma)*w dot x), (1/sqrt(q(w)))sin(sqrt(gamma)*w dot x).
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

# class myReLUSampler:
#     """
#     The random nodes have the form
#     sqrt(gamma)*max(w dot x + b, 0). w and b are random Gaussian.
#     """
#     def __init__(self,n_old_features,gamma=1,n_new_features=20):
#         self.name = 'ReLU'
#         self.sampler = np.random.randn(n_old_features + 1,n_new_features)*np.sqrt(gamma)
#         self.gamma = gamma
#         self.n_new_features = n_new_features
#
#     def fit_transform(self, X):
#         """
#         It transforms one data vector a time
#         """
#         X_til = np.empty(self.n_new_features)
#         for idx in range(self.n_new_features):
#                 X_til[idx] = max(X.dot(self.sampler[:-1,idx])+self.sampler[-1,idx],0)
#         return X_til / np.sqrt(self.n_new_features)

class tfRF2L:
    """
    This is a class constructing a 2-layer net with Fourier or
    ReLU nodes in the hidden layer. The weights in the first layer is
    initialized using random Gaussian or random uniform features,
    respectively. Layerwise training can be applied.
    """
    def __init__(self,feature,n_old_features,
        n_new_features,Lambda,Gamma,classes,
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

    def _feature_layer(self,x,N):
        if self._feature == 'Gaussian':
            if self._initializer == None:
                k_initializer = tf.random_normal_initializer(stddev=np.sqrt(self._Gamma))
                b_initializer = tf.random_uniform_initializer(minval=0.,maxval=np.pi)
            else:
                k_initializer = tf.constant_initializer(self._initializer,dtype=tf.float32)
                b_initializer = tf.random_uniform_initializer(minval=0.,maxval=np.pi)
            trans_layer = tf.layers.dense(inputs=x,units=N,
                use_bias=True,
                kernel_initializer=k_initializer,
                # random fourier features requires bias to be uniform in [0,pi]
                bias_initializer=b_initializer,
                activation=tf.cos,
                name='Gaussian')
            RF_layer = tf.div(trans_layer,tf.sqrt(N*1.0))
            tf.add_to_collection('Hidden',RF_layer)

        elif self._feature == 'ReLU':
            k_initializer = np.random.randn(self._d+1,self._N)
            # initialize by unit vectors
            for idx in range(self._N):
                k_vec = k_initializer[:,idx].copy()
                k_initializer[:,idx] = k_vec / np.linalg.norm(k_vec)
            b_initializer = tf.constant_initializer(k_initializer[-1,:],
                dtype=tf.float32)
            k_initializer = tf.constant_initializer(k_initializer[:-1,:],
                dtype=tf.float32)
            trans_layer = tf.layers.dense(inputs=x,units=N,
                use_bias=True,
                kernel_initializer=k_initializer,
                bias_initializer=b_initializer,
                activation=tf.nn.relu,
                name='Gaussian')
            RF_layer = tf.div(trans_layer,tf.sqrt(N*1.0))
            tf.add_to_collection('Hidden',RF_layer)

        tf.summary.histogram('inner weights',
            self._graph.get_tensor_by_name('Gaussian/kernel:0'))
        return RF_layer

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
            x = tf.placeholder(dtype=tf.float32,
                shape=[None,d],name='features')
            y = tf.placeholder(dtype=tf.uint8,
                shape=[None],name='labels')

            RF_layer = self._feature_layer(x,N)

            if self._loss_fn in ('hinge','squared'):
                np_init = np.random.choice([-1,1],size=(N,1))
                logits_init = tf.constant_initializer(np_init,dtype=tf.float32)
                if n_classes == 2:
                    logits = tf.layers.dense(inputs=RF_layer,
                        use_bias = False,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                        kernel_initializer=logits_init,
                        units=1,name='Logits')
                    logits = tf.reshape(logits,shape=[-1])
                else:
                    print("hinge or squared loss only works for binary classificaiton.")
                    return 0
            elif self._loss_fn == 'log':
                np_init = np.random.choice([-1,1],size=(N,n_classes))
                logits_init = tf.constant_initializer(np_init,dtype=tf.float32)
                logits = tf.layers.dense(inputs=RF_layer,
                    use_bias = False,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=Lambda),
                    kernel_initializer=logits_init,
                    units=n_classes,name='Logits')
                probab = tf.nn.softmax(logits, name="softmax")
                tf.add_to_collection("Probab",probab)

            tf.add_to_collection("Probab",logits)
            tf.summary.histogram('outer weights',
                self._graph.get_tensor_by_name('Logits/kernel:0'))

            # hinge only works for binary classification.
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
            tf.add_to_collection('Loss',reg_loss)

            merged = tf.summary.merge_all()
            tf.add_to_collection('Summary',merged)
            self._sess.run(tf.global_variables_initializer())

        if self.log:
            summary = self._sess.run(merged)
            self._train_writer.add_summary(summary)
        return 1

    def predict(self,data,batch_size=50):
        with self._graph.as_default():
            f_vec = tf.get_collection('Hidden')[0]
            if self._loss_fn == 'hinge':
                logits = tf.get_collection('Probab')[0]
                predictions = {"indices": logits,
                    "feature_vec": f_vec}
            elif self._loss_fn == 'squared':
                logits = tf.get_collection('Probab')[0]
                predictions = {"indices": logits,
                    "feature_vec": f_vec}
            elif self._loss_fn == 'log':
                probab,logits = tf.get_collection('Probab')
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
        predictions,_ = self.predict(data)
        s = 0.
        for idx in range(len(data)):
            s += predictions[idx]==labels[idx]
        accuracy = s / len(data)
        return accuracy

    def fit(self,data,labels,mode='layer 2',opt_method='adam',opt_rate=10.,
        batch_size=1,n_iter=1000,bd=100):
        indices = [self._classes.index(label) for label in labels]
        indices = np.array(indices)
        with self._graph.as_default():
            in_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Gaussian')
            out_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                'Logits')
            loss = tf.get_collection('Loss')[0]
            global_step_1 = self._graph.get_tensor_by_name('global1:0')
            global_step_2 = self._graph.get_tensor_by_name('global2:0')
            merged = tf.get_collection('Summary')[0]
            if opt_method == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=opt_rate)
            if opt_method == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=opt_rate)
            if mode == 'layer 2':
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_2,
                    var_list=out_weights
                )
                if self._Lambda == 0:
                    clip_op = clip_by_maxnorm(out_weights[0],bd)
            if mode == 'layer 1':
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    var_list=in_weights
                )
            if mode == 'over all':
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=global_step_1,
                    )
                if self._Lambda == 0:
                    clip_op = clip_by_maxnorm(out_weights[0],bd)
            # initialize global variables in optimizer
            self._sess.run(tf.global_variables_initializer())
            if self.log:
                self._train_writer = tf.summary.FileWriter('tmp',
                    tf.get_default_graph())

        for idx in range(n_iter):
            rand_list = np.random.randint(len(data),size=batch_size)
            feed_dict = {'features:0':data[rand_list,:],
                         'labels:0':indices[rand_list]}
            if idx % 1000 == 1:
                if self.log:
                    print('iter: {0:d}, loss: {1:.4f}'.format(
                        idx, self._sess.run(loss,feed_dict)))
                    summary = self._sess.run(merged)
                    self._train_writer.add_summary(summary,self._total_iter)
            self._sess.run(train_op,feed_dict)
            if self._Lambda == 0:
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

def unit_interval(leftend,rightend,samplesize):
    if min(leftend,rightend)<0 or max(leftend,rightend)>1:
        print("The endpoints must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    for idx in range(samplesize):
        x = np.random.random()
        X.append(x)
        if leftend>rightend:
            if x>rightend and x<leftend:
                Y.append(-1)
            else:
                Y.append(1)
        else:
            if x>leftend and x<rightend:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle(datarange,overlap,samplesize):
    if min(datarange,overlap)<0 or max(datarange,overlap)>1:
        print("The datarange and overlap values must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    rad1upper = 1+datarange*overlap/2
    rad1lower = rad1upper-datarange
    rad2lower = 1-datarange*overlap/2
    rad2upper = rad2lower+datarange
    for idx in range(samplesize):
        if np.random.random()<0.5:
            Y.append(-1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1lower,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
        else:
            Y.append(1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad2lower,rad2upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle_ideal(gap,label_prob,samplesize):
    X = list()
    Y = list()
    rad1upper = 1 - gap/2
    rad2lower = 1 + gap/2
    for idx in range(samplesize):
        p = np.random.random()
        if p < 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(0,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5*label_prob:
                Y.append(-1)
            else:
                Y.append(1)
        if p > 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1upper,2)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5 + 0.5*label_prob:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

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

def plot_interval(X,Y,ratio=1):
    m = int(len(X) * ratio)
    X = X[0:m]
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(X,Y,c=c)
    plt.savefig('image/interval.eps')
    plt.close(fig)
    return 1

def plot_circle(X,Y,ratio=1):
    m = int(len(X) * ratio)
    A = np.array(X[0:m])
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(A[:,0],A[:,1],c=c)
    circle = plt.Circle((0,0),1,fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.savefig('image/circle.eps')
    plt.close(fig)
    return 1

def clip_by_maxnorm(t,c):
    cc = c*tf.ones(shape=tf.shape(t))
    nt = tf.norm(t,ord=np.inf,axis=0)
    factor = tf.divide(tf.minimum(nt,cc),nt)
    t1 = tf.multiply(t,factor)
    return tf.assign(t,t1)
