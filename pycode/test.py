from experiments import _read_cifar
import matplotlib.pyplot as plt
import numpy as np

Xtr,Ytr,_,_ = _read_cifar()
graycoef = np.array([0.299, 0.587, 0.114])
img = graycoef.dot(Xtr[0].reshape((3,-1)))
img = img.reshape((32,32))
print(np.shape(img))
plt.imshow(img,cmap='gray')
plt.savefig('test.png')
