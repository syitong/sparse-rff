import matplotlib.pyplot as plt
import numpy as np

def unit_interval(leftend,rightend,samplesize):
    if min(leftend,rightend)<0 or max(leftend,rightend)>1:
        print("The endpoints must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    for idx in range(samplesize):
        x = np.random.random()
        X.append(x)
        if leftend>rightend:
            if x>rightend and x<leftend:
                Y.append(-1)
            else:
                Y.append(1)
        else:
            if x>leftend and x<rightend:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle(datarange,overlap,samplesize):
    if min(datarange,overlap)<0 or max(datarange,overlap)>1:
        print("The datarange and overlap values must be between 0 and 1!")
        return False
    X = list()
    Y = list()
    rad1upper = 1+datarange*overlap/2
    rad1lower = rad1upper-datarange
    rad2lower = 1-datarange*overlap/2
    rad2upper = rad2lower+datarange
    for idx in range(samplesize):
        if np.random.random()<0.5:
            Y.append(-1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1lower,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
        else:
            Y.append(1)
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad2lower,rad2upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def unit_circle_ideal(gap,label_prob,samplesize):
    X = list()
    Y = list()
    rad1upper = 1 - gap/2
    rad2lower = 1 + gap/2
    for idx in range(samplesize):
        p = np.random.random()
        if p < 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(0,rad1upper)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5*label_prob:
                Y.append(-1)
            else:
                Y.append(1)
        if p > 0.5:
            theta = np.random.random()*2*np.pi
            radius = np.random.uniform(rad1upper,2)
            X.append(np.array([radius*np.cos(theta),radius*np.sin(theta)]))
            if p < 0.5 + 0.5*label_prob:
                Y.append(1)
            else:
                Y.append(-1)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def check_board(samplesize,size=4):
    data = np.random.rand(samplesize,2) * size - size / 2
    labels = np.sum(data//1,axis=1) % 2
    np.save('data/checkboard-train-data',data[:int(0.9*len(data))])
    np.save('data/checkboard-train-label',labels[:int(0.9*len(data))])
    np.save('data/checkboard-test-data',data[int(0.9*len(data)):])
    np.save('data/checkboard-test-label',labels[int(0.9*len(data)):])

def plot_check_board(samplesize=500,size=4):
    X = np.load('data/checkboard-train-data.npy')[:samplesize]
    Y = np.load('data/checkboard-train-label.npy')[:samplesize]
    c = []
    for idx in range(samplesize):
        if Y[idx] == 0:
            c.append('b')
        else:
            c.append('r')
    fig = plt.figure()
    plt.scatter(X[:,0],X[:,1],s=0.1,c=c)
    plt.savefig('image/checkboard.eps')
    plt.close(fig)
    return 1

def plot_interval(X,Y,ratio=1):
    m = int(len(X) * ratio)
    X = X[0:m]
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(X,Y,c=c)
    plt.savefig('image/interval.eps')
    plt.close(fig)
    return 1

def plot_circle(X,Y,ratio=1):
    m = int(len(X) * ratio)
    A = np.array(X[0:m])
    Y = Y[0:m]
    c = list()
    for idx in range(m):
        if Y[idx]==1:
            c.append('r')
        else:
            c.append('b')
    fig = plt.figure()
    plt.scatter(A[:,0],A[:,1],c=c)
    circle = plt.Circle((0,0),1,fill=False)
    plt.gcf().gca().add_artist(circle)
    plt.axis('equal')
    plt.savefig('image/circle.eps')
    plt.close(fig)
    return 1

if __name__ == '__main__':
    # check_board(samplesize=100000)
    plot_check_board()
