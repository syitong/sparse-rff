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

### set up parameters
datarange = 0.5
overlap = 0
samplesize = 500
logclist = np.arange(-3,4,0.5)
trials = 1

### generate train and test dataset
X,Y = datagen.unit_circle(datarange,overlap,samplesize)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.4,random_state=0)

### estimate gamma in the rbf kernel
gamma = datagen.gamma_est(X_train)

### rbf kernel support vector machine
kscore = list()
for idx in range(len(logclist)):
    C = 10**logclist[idx]
    clf = svm.SVC(C=C,gamma=gamma)
    clf.fit(X_train,Y_train)
    kscore.append(clf.score(X_test,Y_test))

### full and sparse random features method
l1score = list()
l2score = list()
l1sparsity = list()
l2sparsity = list()
for idx in range(trials):
    rbf_feature = RBFSampler(gamma=gamma,n_components=40)
    scaler = StandardScaler()
    X_train_til = rbf_feature.fit_transform(X_train)
    scaler.fit(X_train_til)
    X_train_til = scaler.transform(X_train_til)
    X_test_til = rbf_feature.fit_transform(X_test)
    X_test_til = scaler.transform(X_test_til)
    m = X_train_til.shape[0]
    for idx in range(len(logclist)):
        C = 10**logclist[idx]
        clfl1 = SGDClassifier(loss='hinge',penalty='l1',alpha=1/C/m)
        clfl1.fit(X_train_til,Y_train)
        l1score.append(clfl1.score(X_test_til,Y_test))
        l1sparsity.append(np.sum(clfl1.coef_!=0))
        clfl2 = SGDClassifier(loss='hinge',penalty='l2',alpha=1/C/m)
        clfl2.fit(X_train_til,Y_train)
        l2score.append(clfl2.score(X_test_til,Y_test))
        l2sparsity.append(np.sum(clfl2.coef_!=0))

plt.plot(logclist,kscore,'b-o',fillstyle='none')
plt.plot(logclist,l2score,'r--s',fillstyle='none')
plt.plot(logclist,l1score,'g:x',fillstyle='none')
plt.xlabel('$\log(C)$')
plt.ylabel('accuracy')
plt.savefig('image/results.eps')
with open('result/l1spasity.csv','wb') as csvfile:
    datawriter = csv.writer(csvfile,delimiter=' ')
    datawriter.writerow(l1sparsity)
with open('result/l2spasity.csv','wb') as csvfile:
    datawriter = csv.writer(csvfile,delimiter=' ')
    datawriter.writerow(l2sparsity)
