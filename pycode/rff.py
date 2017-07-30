import numpy as np

class myRBFSampler:
    """
    The random nodes have the form
    cos(sqrt(gamma)*w dot x), sin(sqrt(gamma)*w dot x)
    """
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.sampler = np.random.randn(n_old_features,n_components)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def fit_transform(self, X, mask):
        """
        mask is a list of the same length with the feature
        vector but with only 1s at very sparse positions.
        """
        n = X.shape[0]
        X_cos = np.empty((n,self.n_components))
        X_sin = np.empty((n,self.n_components))
        for idx in range(self.n_components):
            if mask[idx] == 0:
                X_cos[:,idx] = 0
                X_sin[:,idx] = 0
            else:
                X_cos[:,idx] = np.cos(X.dot(self.sampler[:,idx]))
                X_sin[:,idx] = np.sin(X.dot(self.sampler[:,idx]))
        X_til = np.concatenate([X_cos,X_sin],1)
        return X_til / np.sqrt(self.n_components)

class HyperRFSVM:
    """
    This class implements the RFSVM with random drop out for each round
    of subgrad descent.
    """
    def __init__(self,sampler,eta=1,p=0):
        self.eta = eta
        self.sampler = sampler
        self.p = p
        self.w = np.zeros(2*sampler.n_components)

    def train(self,cycle,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        d = X.shape[1]
        n = len(Y)
        for idx in range(cycle):
            for jdx in range(n):
                self.partial_train(X[jdx,:].reshape(1,d),Y[jdx],jdx+idx*n+1)
        return 1

    def partial_train(self,Xrow,y,T):
        mask = np.random.binomial(1,self.p,self.sampler.n_components)
        Xrow_til = self.sampler.fit_transform(Xrow,mask)
        if np.dot(Xrow_til,self.w.T)*y < 1:
            self.w = self.w + y*Xrow_til*self.eta / T
        return 1

    def test(self,X):
        n = X.shape[0]
        mask = np.ones(self.w.shape[1]/2)
        X_til = self.sampler.fit_transform(X,mask)
        output = np.empty(n)
        for idx in range(n):
            if X_til[idx,:].dot(self.w.T) > 0:
                output[idx] = 1
            else:
                output[idx] = -1
        return output
