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
        X_til = np.empty((n,1))
        X_cos = np.empty((n,1))
        X_sin = np.empty((n,1))
        for idx in range(len(mask)):
            if mask[idx] == 0:
                v = np.zeros((n,1))
                X_cos = np.concatenate((X_cos,v),1)
                X_sin = np.concatenate((X_sin,v),1)
            else:
                X_cos = np.concatenate((X_cos,np.cos(X.dot(self.sampler[:,idx])).reshape(n,1)),1)
                X_sin = np.concatenate((X_sin,np.sin(X.dot(self.sampler[:,idx])).reshape(n,1)),1)
        X_til = np.concatenate((X_cos,X_sin),1)
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
        self.w = np.zeros(sampler.n_components)

    def train(self,cycle,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        for idx in range(cycle):
            n = len(Y)
            for jdx in range(n):
                self.partial_train(X[jdx,:],Y[jdx],jdx+idx*n)
        return 1

    def partial_train(self,Xrow,y,T):
        mask = np.random.binomial(1,self.p,self.sampler.n_components)
        Xrow_til = self.sampler.fit_transform(Xrow,mask)
        if np.dot(Xrow_til,self.w)*y < 1:
            self.w = self.w + y*Xrow*self.eta / T
        return 1

    def test(self,X):
        mask = np.ones(shape(X)[1])
        X_til = self.sampler.fit_transform(X,mask)
        return np.sign(X_til.dot(self.w.T))
