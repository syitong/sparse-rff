import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

tags = ['accuracy','sparsity','traintime','testtime']
units = ['-','-','s','s']
for idx in range(4):
    tag = tags[idx]
    unit = units[idx]
    x_labels = np.arange(0.,3.,0.5)
    orfsvm = np.zeros((10,len(x_labels)))
    urfsvm = np.zeros((10,len(x_labels)))
    for prefix in range(1,11,1):
        orfsvm[prefix-1,:] = np.loadtxt(
            'result/adult_{0:s}{2:s}{1:s}'.format(
            'Gaussian',str(prefix), 'layer 2'))[:,idx]
        urfsvm[prefix-1,:] = np.loadtxt(
            'result/adult_{0:s}{2:s}{1:s}'.format(
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
    plt.title("Fourier vs ReLU Feature {} on Adult".format(tag))
    plt.xlabel('log opt rate (-)')
    plt.ylabel('{0} ({1})'.format(tag,unit))
    plt.xticks(x_labels)
    plt.errorbar(x_labels,orfmean,yerr=orfstd,fmt='bs--',label='Fourier',fillstyle='none')
    plt.errorbar(x_labels,urfmean,yerr=urfstd,fmt='gx:',label='ReLU')
    plt.legend(loc=4)
    plt.savefig('image/adult_Fourier_vs_ReLU_{}.eps'.format(tag))
    plt.close(fig)
