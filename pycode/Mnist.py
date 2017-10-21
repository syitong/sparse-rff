"""
This code is used to solve the famous handwritten number
recognition problem via RFSVM.
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff

def read_MNIST_data(filepath):
    with open(filepath,'rb') as file:
        data = file.read()
        offset = 0
        if data[2] < 10:
            length = 1
        elif data[2] < 12:
            length = 2
        elif data[2] < 14:
            length = 4
        elif data[2] < 15:
            length = 8
        dim = list()
        offset = offset + 4
        for idx in range(data[3]):
            dim.append(int.from_bytes(data[3+idx*4+1:3+idx*4+5],'big'))
            offset = offset + 4
        l = length
        for idx in range(1,len(dim)):
            l = l * dim[idx]
        X = np.empty((dim[0],l))
        for idx in range(dim[0]):
            for jdx in range(l):
                index = offset + idx * l + jdx
                X[idx,jdx] = data[index]
        return X

def main():
    Xtrain = read_MNIST_data('data/train-images.idx3-ubyte')
    Ytrain = read_MNIST_data('data/train-labels.idx1-ubyte')
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte')
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte')

if __name__ == '__main__':
    main()
