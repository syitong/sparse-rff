import numpy as np
import matplotlib.pyplot as plt
import datagen, dataplot
import rff

Sampler = rff.myRBFSampler(n_old_features=3,n_components=5)
X = np.array([[1,2,3]])
mask = [0,1,0,0,1]
X_til = Sampler.fit_transform(X,mask)
print X_til
