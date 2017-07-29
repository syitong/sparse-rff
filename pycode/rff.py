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

    def fit_transform(self, X, mask=0):
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
