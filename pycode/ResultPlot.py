import numpy as np
import matplotlib.pyplot as plt

samplesize = np.arange(1000,60001,5000)
orfsvm = np.zeros((10,len(samplesize)))
urfsvm = np.zeros((10,len(samplesize)))
# mode = 'over all'
for prefix in range(1,11,1):
    orfsvm[prefix-1,:] = np.loadtxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format('ReLU',str(prefix),'over all'))
    urfsvm[prefix-1,:] = np.loadtxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format('ReLU',str(prefix),'layer 1'))

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
plt.title("Overall vs Layer1 ReLU on MNIST")
plt.xlabel('sample size (k)')
plt.ylabel('accuracy')
plt.xticks(samplesize/1000)
plt.errorbar(samplesize/1000,orfmean,yerr=orfstd,fmt='bs--',label='over all',fillstyle='none')
plt.errorbar(samplesize/1000,urfmean,yerr=urfstd,fmt='gx:',label='layer 1')
plt.legend(loc=4)
plt.savefig('image/Overall_vs_Layer1_ReLU.eps')
