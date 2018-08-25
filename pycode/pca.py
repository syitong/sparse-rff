import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from uci_pre import read_data

def pca(N,m,gamma):
    X = read_data('strips-test-data.npy')[:m]
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
    plt.legend(loc=4)
    plt.savefig('image/pca.eps')
    plt.close(fig)

if __name__ == '__main__':
    pca(20,100,1000)
