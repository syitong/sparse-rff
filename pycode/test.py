import numpy as np
import matplotlib.pyplot as plt
import rff
import tensorflow as tf

X,Y = rff.unit_circle_ideal(0.1,0.9,300)
Xtrain = X[:200]
Ytrain = Y[:200]
Xtest = X[200:]
Ytest = Y[200:]
n_old_features = 2
n_components = 20
Lambda = 0.0005
method = 'layer 2'
batch_size = 1
n_iter = 1000
Gamma = rff.gamma_est(Xtrain) / 1
classes = [-1,1]
clf = rff.tfRF2L(n_old_features,n_components,
    Lambda,Gamma,classes,method,batch_size,n_iter)
accuracy = clf.score(Xtest,Ytest)
print('accuracy={0:.3f}'.format(accuracy))
clf.fit(Xtrain,Ytrain)
clf.method = 'layer 1'
clf.fit(Xtrain,Ytrain)
clf.method = 'over all'
clf.fit(Xtrain,Ytrain)
accuracy = clf.score(Xtest,Ytest)
print('accuracy={0:.3f}'.format(accuracy))
clf.close()
