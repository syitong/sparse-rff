import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# my module
import datagen, dataplot
import rff

### set up parameters
datarange = 0.5
overlap = 0.3
samplesize = 1500
trials = 1

### generate train and test dataset
X,Y = datagen.unit_circle(datarange,overlap,samplesize)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,random_state=0)
gamma = datagen.gamma_est(X_train,portion=0.1)
sampler = rff.myRBFSampler(X.shape[1],gamma,20)
HyperModel = rff.HyperRFSVM(sampler,1,0.3)

for idx in range(trials):
    HyperModel.train(5,X_train,Y_train)

l = len(Y_test)
output = HyperModel.test(X_test)
score = 0
for idx in range(l):
    if output[idx] == Y_test[idx]:
        score = score + 1
print score, l
print float(score) / l 
