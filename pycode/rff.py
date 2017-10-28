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
        m = len(X)
        X_til = np.empty((m,self.n_components*2))
        X_til[:,:self.n_components] = np.cos(X.dot(self.sampler))
        X_til[:,self.n_components:] = np.sin(X.dot(self.sampler))
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
        m = len(X)
        X_til = np.empty((m,self.n_components*2))
        factor = (np.sqrt(self.Prob[self.feature_list]
                  * self.feature_pool_size))
        X_til[:,:self.n_components] = np.cos(X.dot(self.sampler)) / factor
        X_til[:,self.n_components:] = np.sin(X.dot(self.sampler)) / factor
        return X_til / np.sqrt(self.n_components)

class myReLUSampler:
    """
    The random nodes have the form
    max(sqrt(gamma)*w dot x, 0)
    """
    def __init__(self,n_oldkfeatures,gamma=1,n_components=20):
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

class HyperRFSVM:
    """
    This class implements the RFSVM with random drop out for each round
    of subgrad descent.
    """
    def __init__(self,sampler,p=0,reg=0,max_iter=5,tol=10**(-3),n_jobs=n_jobs):
        self.sampler = sampler
        self.type = self.sampler.name
        self.classes_ = [1,-1]
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg
        self.n_jobs = n_jobs
        if self.type == 'rbf':
            self.w = np.zeros(2*sampler.n_components)
        else:
            self.w = np.zeros(sampler.n_components)

    def _fit_binary(self,X,Y):
        n = len(Y)
        T = 0
        score = list()
        cycle = self.max_iter
        for idx in range(cycle):
            jlist = np.random.permutation(n)
            for jdx in range(n):
                T = jdx+idx*n+1
                score.append(self.partial_fit(X[jlist[jdx]],Y[jlist[jdx]],T))
                if score[-1] - score[-2] < self.tol:
                    break
            if score[-1] - score[-2] < self.tol:
                break
        return score

    def fit(self,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        self.classes_ = list(set(Y))
        if len(self.classes_) > 2:
            Ycopy = np.empty((len(Y),self.classes_))
            for idx,val in enumerate(self.classes_):
                for jdx,label in enumerate(Y):
                    if label == val:
                        Ycopy[idx,jdx] = 1
                    else:
                        Ycopy[idx,jdx] = -1

            self.w = Parallel(n_jobs=self.n_jobs,backend="threading")(
                delayed(_fit_binary)(self,X,Y[idx])
                for idx in range(len(self.classes_)))

        elif len(self.classes_) == 2:
            for idx in range(len(Y)):
                if Y[idx] == self.classes_[0]:
                    Y[idx] = 1
                elif Y[idx] == self.classes_[1]:
                    Y[idx] = -1
            return self._fit_binary(X,Y)

    def partial_fit(self,Xrow,y,T):
        if y != 1 or y != -1:
            print('The label for partial_fit must be 1 or -1!')
            return 0
        if np.random.rand() < self.p:
            n_components = self.sampler.n_components
            w_norm = np.empty(n_components)
            if self.type == 'rbf':
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
            if self.reg == 0:
                self.w = self.w + y*Xrow_til/np.sqrt(T)
            else:
                self.w = (1-1/T)*self.w + y*Xrow_til/T/self.reg
        else:
            if self.reg == 0:
                self.w = self.w
            else:
                self.w = (1-1/T)*self.w
        score = max(1 - np.dot(Xrow_til,self.w.T)*y,0)
        return score

    def predict(self,X):
        output = []
        if len(self.classes_) > 2:
            for idx in range(len(X)):
                score = 0
                label = self.classes_[0]
                for jdx,val in enumerate(self.classes_):
                    s = X_til.dot(self.w[jdx].T)
                    if score < s:
                        score = s
                        label = val
                output.append(label)

        elif len(self.classes_) == 2:
            X_til = self.sampler.fit_transform(X)
            if X_til.dot(self.w.T) > 0:
                output.append(self.classes_[0])
            else:
                output.append(self.classes_[1])
        return np.array(output)

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
