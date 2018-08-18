import numpy as np

tags = ['accuracy','sparsity','traintime','testtime']
with open('result/best_covtype_b_summary','w') as f:
    for idx in range(4):
        tag = tags[idx]
        orfsvm = np.zeros(10)
        urfsvm = np.zeros(10)
        for prefix in range(1,11,1):
            orfsvm[prefix-1] = np.loadtxt(
                'result/best_covtype_b_{0:s}{2:s}{1:s}'.format(
                'Gaussian',str(prefix), 'layer 2'))[idx]
            urfsvm[prefix-1] = np.loadtxt(
                'result/best_covtype_b_{0:s}{2:s}{1:s}'.format(
                    'ReLU',str(prefix),'layer 2'))[idx]

        orfmean = np.mean(orfsvm)
        urfmean = np.mean(urfsvm)
        orfstd = np.std(orfsvm)
        urfstd = np.std(urfsvm)
        f.write('Gaussian {0:s} mean: {1:.4f}\n'.format(tag,orfmean))
        f.write('Gaussian {0:s} std: {1:.4f}\n'.format(tag,orfstd))
        f.write('ReLU {0:s} mean: {1:.4f}\n'.format(tag,urfmean))
        f.write('ReLU {0:s} std: {1:.4f}\n'.format(tag,urfstd))
