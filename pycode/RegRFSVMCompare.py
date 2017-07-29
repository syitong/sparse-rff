### This code is for comparing the performance of L2, L1 regularized
### RFSVM and KSVM.
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
overlap = 0
samplesize = 1500
logclist = np.arange(-3,4.5,0.5)
trials = 1

### generate train and test dataset
X,Y = datagen.unit_circle(datarange,overlap,samplesize)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state=0)

### estimate gamma in the rbf kernel
gamma = datagen.gamma_est(X_train)

### rbf kernel support vector machine
kscore = list()
ksparsity = list()
for idx in range(len(logclist)):
    C = 10**logclist[idx]
    clf = svm.SVC(C=C,gamma=gamma)
    clf.fit(X_train,Y_train)
    kscore.append(clf.score(X_test,Y_test))
    ksparsity.append(clf.n_support_)

### full and sparse random features method
l1score = list()
l2score = list()
l1sparsity = list()
l2sparsity = list()
for idx in range(trials):
    #rbf_feature = rff.myRBFSampler(gamma=gamma,n_old_features=X_train.shape[1])
    rbf_feature = RBFSampler(gamma=gamma,n_components=20)
    X_train_til = rbf_feature.fit_transform(X_train)
    X_test_til = rbf_feature.transform(X_test)
    m = X_train_til.shape[0]
    for idx in range(len(logclist)):
        C = 10**logclist[idx]
        clfl1 = SGDClassifier(loss='hinge',penalty='l1',alpha=1/C/m)
        #clfl1 = svm.LinearSVC(penalty='l1',C=C,dual=False)
        clfl1.fit(X_train_til,Y_train)
        l1score.append(clfl1.score(X_test_til,Y_test))
        l1sparsity.append(np.sum(clfl1.coef_!=0))
        clfl2 = SGDClassifier(loss='hinge',penalty='l2',alpha=1/C/m)
        #clfl2 = svm.LinearSVC(loss='hinge',penalty='l2',C=C)
        clfl2.fit(X_train_til,Y_train)
        l2score.append(clfl2.score(X_test_til,Y_test))
        l2sparsity.append(np.sum(clfl2.coef_!=0))
    l1score = np.array(l1score)
    l2score = np.array(l2score)
    np.set_printoptions(precision=2)
    print l1score,'\n',l2score

plt.plot(logclist,kscore,'r-o',fillstyle='none')
plt.plot(logclist,l2score,'b--s',fillstyle='none')
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
with open('result/kspasity.csv','wb') as csvfile:
    datawriter = csv.writer(csvfile,delimiter=' ')
    datawriter.writerow(ksparsity)
