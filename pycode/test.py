import numpy as np
import matplotlib.pyplot as plt
import rff
import tensorflow as tf
import tfRF2L

X,Y = rff.unit_circle_ideal(0.1,0.9,300)
for idx in range(len(Y)):
    if Y[idx] == -1:
        Y[idx] = 0
Xtrain = X[:200]
Ytrain = Y[:200]
Xtest = X[200:]
Ytest = Y[200:]
n_old_features = 2
n_components = 20
Lambda = 0.0005
batch_size = 10
Gamma = rff.gamma_est(Xtrain) / 0.2
n_classes = 2
clf = tfRF2L.tfRF2L(n_old_features,n_components,
    Lambda,Gamma,n_classes)
Ypred = clf.predict(Xtest)['classes']
accuracy = np.sum(Ytest==Ypred) / 100
print('accuracy={0:.3f}'.format(accuracy))
clf.fit(Xtrain,Ytrain,'layer 2',batch_size,1000)
clf.fit(Xtrain,Ytrain,'layer 1',batch_size,1000)
# clf.fit(Xtrain,Ytrain,'over all',batch_size,2000)
Ypred = clf.predict(Xtest)['classes']
accuracy = np.sum(Ytest==Ypred) / 100
print('accuracy={0:.3f}'.format(accuracy))
clf.close()
