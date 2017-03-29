import numpy as np
import matplotlib.pyplot as plt
import datagen

[X,Y] = datagen.unit_circle(0.3,0.3,50)
c = list()
for idx in range(len(Y)):
    if Y[idx]==1:
        c.append('r')
    else:
        c.append('b')
plt.scatter(X[:,0],X[:,1],c=c)
plt.show()
