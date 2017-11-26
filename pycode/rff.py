import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class myRBFSampler:
    """
    The random nodes have the form
    cos(sqrt(gamma)*w dot x), sin(sqrt(gamma)*w dot x)
    """
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.name = 'rbf'
        self.sampler = np.random.randn(n_old_features,n_components)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def update(self, idx):
        self.sampler[:,idx] = np.random.randn(self.sampler.shape[0])*np.sqrt(self.gamma)
        return 1

    def fit_transform(self, X):
        X_tilc = np.cos(X.dot(self.sampler))
        X_tils = np.sin(X.dot(self.sampler))
        X_til = np.concatenate((X_tilc,X_tils),axis=-1)
        return X_til / np.sqrt(self.n_components)

    def weight_estimate(self, X, X_pool_fraction, Lambda):
        m = len(X)
        X_pool_size = min(int(m * X_pool_fraction),500)
        n_components = self.n_components
        T = np.empty((X_pool_size,n_components*2))
        k = np.random.randint(m,size=X_pool_size)
        X_pool = X[k,:]
        A = X_pool.dot(self.sampler)
        T[:,:n_components] = np.cos(A)
        T[:,n_components:] = np.sin(A)
        U,s,V = np.linalg.svd(T, full_matrices=False)
        Trace = s**2 / (s**2 + Lambda * X_pool_size * n_components)
        Weight = np.empty(n_components*2)
        for idx in range(n_components*2):
            Weight[idx] = V[:,idx].dot(Trace * V[:,idx])
        Weight = Weight[:n_components] + Weight[n_components:]
        return Weight

class optRBFSampler:
    """
    The random nodes have the form
    (1/sqrt(q(w)))cos(sqrt(gamma)*w dot x), (1/sqrt(q(w)))sin(sqrt(gamma)*w dot x).
    q(w) is the optimized density of features with respect to the initial
    feature distribution determined only by the RBF kernel.
    """
    def __init__(self,
                 n_old_features,
                 feature_pool_size,
                 gamma=1,
                 n_components=20):
        self.name = 'opt_rbf'
        self.pool = (np.random.randn(n_old_features,
                                     feature_pool_size)
                    * np.sqrt(gamma))
        self.feature_pool_size = feature_pool_size
        self.gamma = gamma
        self.n_components = n_components
        Weight = np.ones(feature_pool_size)
        self.Weight = Weight
        self.Prob = Weight / np.sum(Weight)
        self.feature_list = np.random.choice(feature_pool_size,
                                size=n_components,
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
                                             size=self.n_components,
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
        return X_til / np.sqrt(self.n_components)

class myReLUSampler:
    """
    The random nodes have the form
    max(sqrt(gamma)*w dot x, 0)
    """
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.name = 'ReLU'
        self.sampler = np.random.randn(n_old_features,n_components)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def update(self, idx):
        self.sampler[:,idx] = np.random.randn(self.sampler.shape[0])*np.sqrt(self.gamma)
        return 1

    def fit_transform(self, X):
        """
        It transforms one data vector a time
        """
        X_til = np.empty(self.n_components)
        for idx in range(self.n_components):
                X_til[idx] = max(X.dot(self.sampler[:,idx]),0)
        return X_til / np.sqrt(self.n_components)

class HRFSVM_binary:
    """
    This class implements the RFSVM with random drop out for each round
    of subgrad descent.
    """
    def __init__(self,n_old_features,n_components=20,gamma=1,p=0,
        alpha=0,max_iter=5,tol=10**(-3)):
        self.sampler = myRBFSampler(n_old_features=n_old_features,
            n_components=n_components,gamma=gamma)
        # self.feature_type = self.sampler.name
        self.classes_ = [1,-1]
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.w = np.zeros(2 * self.sampler.n_components)
        # if self.feature_type == 'rbf':
        #     self.w = np.zeros(2*self.sampler.n_components)
        # else:
        #     self.w = np.zeros(self.sampler.n_components)

    def fit(self,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        self.classes_ = list(set(Y))
        if self.classes_ != [-1,1] and self.classes_ != [1,-1]:
            for idx in range(len(Y)):
                if Y[idx] == self.classes_[0]:
                    Y[idx] = 1
                elif Y[idx] == self.classes_[1]:
                    Y[idx] = -1
        n = len(Y)
        T = 0
        score = [1000]
        for idx in range(self.max_iter):
            jlist = np.random.permutation(n)
            for jdx in range(n):
                T = jdx+idx*n+1
                score.append(self.partial_fit(X[jlist[jdx]],Y[jlist[jdx]],T))
                if len(score) > 1:
                    if score[-2] - score[-1] < self.tol:
                        break
            if len(score) > 1:
                if score[-2] - score[-1] < self.tol:
                    break
        return score

    def partial_fit(self,Xrow,y,T):
        if np.random.rand() < self.p:
            n_components = self.sampler.n_components
            w_norm = np.empty(n_components)
            if self.sampler.name == 'rbf':
                for idx in range(n_components):
                    w_norm[idx] = self.w[idx]**2+self.w[idx+n_components]**2
                update_idx = np.argmin(w_norm)
                self.sampler.update(update_idx)
                self.w[update_idx] = 0
                self.w[update_idx+n_components] = 0
            else:
                for idx in range(n_components):
                    w_norm[idx] = np.abs(self.w[idx])
                update_idx = np.argmin(w_norm)
                self.sampler.update(update_idx)
                self.w[update_idx] = 0
        Xrow_til = self.sampler.fit_transform(Xrow)
        score = max(1 - np.dot(Xrow_til,self.w.T)*y,0)
        if score > 0:
            if self.alpha == 0:
                self.w = self.w + y*Xrow_til/np.sqrt(T)
            else:
                self.w = (1-1/T)*self.w + y*Xrow_til/T/self.alpha
        else:
            if self.alpha == 0:
                self.w = self.w
            else:
                self.w = (1-1/T)*self.w
        score = max(1 - np.dot(Xrow_til,self.w.T)*y,0)
        return score

    def predict(self,X):
        output = []
        X_til = self.sampler.fit_transform(X)
        if X_til.dot(self.w.T) > 0:
            output.append(self.classes_[0])
        else:
            output.append(self.classes_[1])
        return np.array(output)

class HRFSVM:
    def __init__(self,n_components=20,gamma=1,p=0,
        alpha=0,max_iter=5,tol=10**(-3),n_jobs=-1):
        self.n_old_features = 0
        self.n_components = 20
        self.gamma = 1
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.estimator = []
        self.classes_ = []

    def _fit_binary(self,X,Y):
        estimator = HRFSVM_binary(n_old_features=self.n_old_features,
            n_components=self.n_components,gamma=self.gamma,
            p=self.p,alpha=self.alpha,max_iter=self.max_iter,
            tol=self.tol)
        estimator.fit(X,Y)
        return estimator

    def fit(self,X,Y):
        self.n_old_features = len(X[0])
        self.classes_ = list(set(Y))
        if len(self.classes_) > 2:
            Ycopy = np.empty((len(self.classes_),len(Y)))
            for idx,val in enumerate(self.classes_):
                for jdx,label in enumerate(Y):
                    if label == val:
                        Ycopy[idx,jdx] = 1
                    else:
                        Ycopy[idx,jdx] = -1
            self.estimator = Parallel(n_jobs=self.n_jobs,backend="threading")(
                delayed(self._fit_binary)(X,Ycopy[idx])
                for idx in range(len(self.classes_)))
            return 1

        elif len(self.classes_) == 2:
            for idx in range(len(Y)):
                if Y[idx] == self.classes_[0]:
                    Y[idx] = 1
                elif Y[idx] == self.classes_[1]:
                    Y[idx] = -1
            self.estimator = self._fit_binary(X,Y)
            return 1

    def predict(self,X):
        if len(self.classes_) > 2:
            output = []
            for idx in range(len(X)):
                score = 0
                label = self.classes_[0]
                for jdx,val in enumerate(self.classes_):
                    X_til = self.estimator[jdx].sampler.fit_transform(X[idx])
                    s = X_til.dot(self.estimator[jdx].w.T)
                    if score < s:
                        score = s
                        label = val
                output.append(label)
            return output
        elif len(self.classes_) == 2:
            X_til = self.estimator.sampler.fit_transform(X)
            score = X_til.dot(self.estimator.w.T)
            output = []
            for idx in range(len(X)):
                if score[idx] > 0:
                    output.append(self.classes_[0])
                else:
                    output.append(self.classes_[1])
            return output

    def get_params(self,deep=False):
        return {'n_components': self.n_components,
                'gamma': self.gamma,
                'p': self.p,
                'alpha': self.alpha,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'n_jobs': self.n_jobs}

class tfRFSVM(tf.estimator.LinearClassifier):
    """
    This class implements the RFSVM with softmax + cross entropy
    as loss function by using Tensor Flow library
    to solve multi-class problems natively.
    """
    def __init__(self,)

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

def dim_modifier(X,dim,method='const'):
    if method == 'const':
        Tail = np.ones((X.shape[0],dim))
        Xtil = np.concatenate((X,Tail),axis=-1)
        return Xtil
    else:
        Tail = np.random.randn(X.shape[0],dim)
        Xtil = np.concatenate((X,Tail),axis=-1)
        return Xtil

def gamma_est(X,portion = 0.3):
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
