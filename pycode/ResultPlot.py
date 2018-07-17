import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

tags = ['accuracy','sparsity','traintime','testtime']
units = ['-','-','s','s']
for idx in range(4):
    tag = tags[idx]
    unit = units[idx]
    samplesize = np.arange(1000,60001,5000)
    orfsvm = np.zeros((10,len(samplesize)))
    urfsvm = np.zeros((10,len(samplesize)))
    for prefix in range(1,11,1):
        orfsvm[prefix-1,:] = np.loadtxt(
            'result/tfRF2L_{0:s}{2:s}{1:s}'.format(
            'Gaussian',str(prefix), 'layer 2'))[:,idx]
        urfsvm[prefix-1,:] = np.loadtxt(
            'result/tfRF2L_{0:s}{2:s}{1:s}'.format(
                'ReLU',str(prefix),'layer 2'))[:,idx]
    
    orfmean = np.mean(orfsvm,axis=0)
    urfmean = np.mean(urfsvm,axis=0)
    orfstd = np.std(orfsvm,axis=0)
    urfstd = np.std(urfsvm,axis=0)
    
    # opt vs unif feature selection
    # plt.title("opt vs unif feature selection on MNIST")
    # plt.xlabel('sample size (k)')
    # plt.ylabel('accuracy')
    # plt.xticks(samplesize/1000)
    # plt.errorbar(samplesize/1000,orfmean,yerr=orfstd,fmt='bs--',label='opt',fillstyle='none')
    # plt.errorbar(samplesize/1000,urfmean,yerr=urfstd,fmt='gx:',label='unif')
    # plt.legend(loc=4)
    # plt.savefig('image/opt_vs_unif.eps')
    
    # Gaussian vs ReLU random features
    fig = plt.figure()
    plt.title("Fourier vs ReLU Feature {} on MNIST".format(tag))
    plt.xlabel('sample size (k)')
    plt.ylabel('{0} ({1})'.format(tag,unit))
    plt.xticks(samplesize/1000)
    plt.errorbar(samplesize/1000,orfmean,yerr=orfstd,fmt='bs--',label='Fourier',fillstyle='none')
    plt.errorbar(samplesize/1000,urfmean,yerr=urfstd,fmt='gx:',label='ReLU')
    plt.legend(loc=4)
    plt.savefig('image/Fourier_vs_ReLU_{}.eps'.format(tag))
    plt.close(fig)
