import numpy as np
import matplotlib.pyplot as plt

def plot_interval(X,Y):
    c = list()
    for idx in range(len(Y)):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    plt.scatter(X,Y,c=c)
    plt.savefig('image/interval.eps')

def plot_circle(X,Y):
    c = list()
    for idx in range(len(Y)):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    plt.scatter(X[:,0],X[:,1],c=c)
    circle = plt.Circle((0,0),1,fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.savefig('image/circle.eps') 
