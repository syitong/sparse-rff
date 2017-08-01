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

    def update(self, idx):
        self.sampler[:,idx] = np.random.randn(self.sampler.shape[0])*np.sqrt(self.gamma)
        return 1

    def fit_transform(self, X):
        """
        It transform one data vector a time
        """
        X_til = np.empty(self.n_components*2)
        for idx in range(self.n_components*2):
            if idx < self.n_components:
                X_til[idx] = np.cos(X.dot(self.sampler[:,idx]))
            else:
                X_til[idx] = np.sin(X.dot(self.sampler[:,idx-self.n_components]))
        return X_til / np.sqrt(self.n_components)

class HyperRFSVM:
    """
    This class implements the RFSVM with random drop out for each round
    of subgrad descent.
    """
    def __init__(self,sampler,p=0,reg=0):
        self.sampler = sampler
        self.p = p
        self.w = np.zeros(2*sampler.n_components)
        self.reg = reg

    def train(self,cycle,X,Y):
        """
        We run the cyclic subgradient descent. cycle is the number of
        repeats of the cycles of the dataset.
        """
        n = len(Y)
        T = 0
        score = list()
        for idx in range(cycle):
            jlist = np.random.permutation(n)
            for jdx in range(n):
                T = jdx+idx*n+1
                score.append(self.partial_train(X[jlist[jdx]],Y[jlist[jdx]],T))
        return score

    def partial_train(self,Xrow,y,T):
        if np.random.rand() < self.p:
            n_components = self.sampler.n_components
            w_norm = np.empty(n_components)
            for idx in range(n_components):
                w_norm[idx] = self.w[idx]**2+self.w[idx+n_components]**2
            update_idx = np.argmin(w_norm)
            self.sampler.update(update_idx)
            self.w[update_idx] = 0
            self.w[update_idx+n_components] = 0
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
        return score

    def test(self,X):
        n = len(X)
        output = list()
        for idx in range(n):
            X_til = self.sampler.fit_transform(X[idx])
            if X_til.dot(self.w.T) > 0:
                output.append(1)
            else:
                output.append(-1)
        return output
