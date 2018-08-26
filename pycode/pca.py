import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from uci_pre import read_data
from libmnist import get_train_test_data
from result_show import print_params

def pca(dataset,N,m,gamma):
    if dataset == 'mnist':
        X,_,_,_ = get_train_test_data()
        X = X[:m]
    else:
        X = read_data(dataset+'-test-data.npy')[:m]
    d = len(X[0])
    k_RFF = np.random.randn(d,N) * np.sqrt(gamma)
    b_RFF = np.random.rand(N) * np.pi
    X_RFF = np.cos(X.dot(k_RFF) + b_RFF)
    k_RRF = np.random.randn(d+1,N)
    k_RRF = k_RRF / np.linalg.norm(k_RRF,axis=0)
    X_ext = np.concatenate((X,np.ones((m,1))),axis=1)
    X_RRF = np.maximum(X_ext.dot(k_RRF),0)
    _,s_RFF,_ = np.linalg.svd(X_RFF)
    s_RFF = s_RFF / np.max(s_RFF)
    _,s_RRF,_ = np.linalg.svd(X_RRF)
    s_RRF = s_RRF / np.max(s_RRF)
    fig = plt.figure()
    plt.plot(s_RFF,label='Fourier')
    plt.plot(s_RRF,label='ReLU')
    plt.legend(loc=1)
    plt.savefig('image/pca-'+dataset+'.eps')
    plt.close(fig)

if __name__ == '__main__':
    dataset = 'covtype'
    F_gamma,F_rate,R_rate = print_params(dataset)
    pca(dataset,500,3000,10**F_gamma)
