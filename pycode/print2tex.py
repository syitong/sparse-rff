import numpy as np
from numpy import array

fnames = ['sine1','sine1-10','strips',
        'square','checkboard','adult',
        'mnist','covtype']
tags = ['traintime']
features = ['Gaussian','ReLU']

if __name__ == '__main__':
    fw = open('alloc4tex','w')
    for tag in tags:
        fw.write(' & '+tag)
    for tag in tags:
        fw.write(' & '+tag)
    fw.write(' \\\\')
    fw.write('\n')
    for fname in fnames:
        feature = 'Gaussian'
        with open('result/'+fname+'-'+feature+'-test-alloc','r') as f:
            result1 = eval(f.read())[0]
        feature = 'ReLU'
        with open('result/'+fname+'-'+feature+'-test-alloc','r') as f:
            result2 = eval(f.read())[0]
        fw.write(fname)
        for tag in tags:
            fw.write(' & {mean:.3f}({std:.3f})'.format(**result1[tag]))
        for tag in tags:
            fw.write(' & {mean:.3f}({std:.3f})'.format(**result2[tag]))
        fw.write(' \\\\')
        fw.write('\n')
    fw.close()

