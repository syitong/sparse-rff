import numpy as np

class myRBFSampler:
    ### the random nodes have the form
    ### cos(sqrt(gamma)*w dot x), sin(sqrt(gamma)*w dot x)
    def __init__(self,n_old_features,gamma=1,n_components=20):
        self.sampler = np.random.randn(n_components,n_old_features)*np.sqrt(gamma)
        self.gamma = gamma
        self.n_components = n_components

    def fit_transform(self, X):
        n = X.shape[0]
        X_til = list()
        c = np.cos(np.dot(X, self.sampler))
        s = np.sin(np.dot(X, self.sampler))
        X_til = np.append(c, s, 1)
        return X_til / np.sqrt(self.n_components)
