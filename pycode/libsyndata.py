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
    cut = int(0.9*len(data))
    np.save('data/checkboard-train-data',data[:cut])
    np.save('data/checkboard-train-label',labels[:cut])
    np.save('data/checkboard-test-data',data[cut:])
    np.save('data/checkboard-test-label',labels[cut:])

def strips(samplesize,n_strip = 4):
    data = np.random.rand(samplesize,2) * n_strip - n_strip / 2
    labels = data[:,0] // 1 % 2
    data[data[:,0] > 0,0] = data[data[:,0] > 0,0] / 10
    cut = int(0.9*len(data))
    np.save('data/strips-train-data',data[:cut])
    np.save('data/strips-train-label',labels[:cut])
    np.save('data/strips-test-data',data[cut:])
    np.save('data/strips-test-label',labels[cut:])

def plot_data(dataset,samplesize=500,size=4):
    X = np.load('data/'+dataset+'-train-data.npy')[:samplesize]
    Y = np.load('data/'+dataset+'-train-label.npy')[:samplesize]
    c = []
    for idx in range(samplesize):
        if Y[idx] == 0:
            c.append('b')
        else:
            c.append('r')
    fig = plt.figure()
    plt.scatter(X[:,0],X[:,1],s=0.1,c=c)
    plt.savefig('image/'+dataset+'.eps')
    plt.close(fig)
    return 1

if __name__ == '__main__':
    # check_board(samplesize=100000)
    strips(100000)
    plot_data('strips')
