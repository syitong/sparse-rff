import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

def plot_learning_rate():
    tags = ['accuracy','sparsity','traintime','testtime']
    units = ['-','-','s','s']
    for idx in range(4):
        tag = tags[idx]
        unit = units[idx]
        x_labels = np.arange(-2.,3.,0.5)
        orfsvm = np.zeros((10,len(x_labels)))
        urfsvm = np.zeros((10,len(x_labels)))
        for prefix in range(1,11,1):
            orfsvm[prefix-1,:] = np.loadtxt(
                'result/covtype_{0:s}{2:s}{1:s}'.format(
                'Gaussian',str(prefix), 'layer 2'))[:,idx]
            urfsvm[prefix-1,:] = np.loadtxt(
                'result/covtype_{0:s}{2:s}{1:s}'.format(
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
        plt.title("Fourier vs ReLU Feature {} on Covtype".format(tag))
        plt.xlabel('log opt rate (-)')
        plt.ylabel('{0} ({1})'.format(tag,unit))
        plt.xticks(x_labels)
        plt.errorbar(x_labels,orfmean,yerr=orfstd,fmt='bs--',label='Fourier',fillstyle='none')
        plt.errorbar(x_labels,urfmean,yerr=urfstd,fmt='gx:',label='ReLU')
        plt.legend(loc=4)
        plt.savefig('image/covtype_Fourier_vs_ReLU_{}.eps'.format(tag))
        plt.close(fig)

def plot_params():
    dataset = 'covtype-refine'
    with open('result/'+dataset+'-alloc','r') as f:
        result = eval(f.read())
    result_trim = [row[1:] for row in result[1:]]
    result_trim = np.array(result_trim)
    F_result = np.max(result_trim[:,:-1],axis=0)
    x_labels = result[0][1:-1]
    R_result = [max(result_trim[:,-1])] * len(x_labels)
    fig = plt.figure()
    plt.title("Random Features Methods on "+dataset)
    plt.xlabel('log(Gamma)')
    plt.ylabel('accuracy')
    plt.xticks(x_labels)
    plt.ylim((0,1.01))
    plt.plot(x_labels,F_result,'x--',label='Fourier')
    plt.plot(x_labels,R_result,':',label='ReLU')
    plt.legend(loc=5)
    plt.savefig('image/{}-gamma.eps'.format(dataset))
    plt.close(fig)

if __name__ == '__main__':
    plot_params()
