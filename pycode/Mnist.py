"""
This code is used to solve the famous handwritten number
recognition problem via RFSVM.
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import time

def read_MNIST_data(filepath):
    """
    This method is sufficiently general to read in any
    data set of the same structure with MNIST
    """
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    # set up timer and progress tracker
    progress = {'task':['start'],'time':[time.process_time()]}
    print(progress['task'][-1],' ',progress['time'][-1])

    # read in MNIST data set
    Xtrain = read_MNIST_data('data/train-images.idx3-ubyte')
    Ytrain = read_MNIST_data('data/train-labels.idx1-ubyte')
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte')
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte')
    progress['task'].append('data read in complete')
    progress['time'].append(time.progress_time())
    print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])

    # set up parameters
    LogLambda = np.arange(-8.0,1.0,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-2,2,0.5)
    LogGamma = np.log10(gamma) * 10**LogGamma
    X_pool_fraction = 0.3
    feature_pool_size = n_components * 100
    n_components = 10

    # use the same pool for all config of parameters
    opt_feature = rff.optRBFSampler(Xtrain.shape[1],
        feature_pool_size,n_components=n_components)
    progress['task'].append('feature pool generated')
    progress['time'].append(time.progress_time())
    print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in len(LogGamma):
        Gamma = 10**LogGamma[idx]
        opt_feature.gamma = Gamma
        for jdx in len(LogLambda):
            Lambda = 10**LogLambda[jdx]
            opt_feature.reweight(Xtrain,X_pool_fraction,Lambda=Lambda)
            progress['task'].append('features generated for'
                + 'Gamma={0:.1e} and Lambda={1:d}'.format(Gamma,Lambda))
            progress['time'].append(time.progress_time())
            print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])
            Xtraintil = opt_feature.fit_transform(Xtrain)
            progress['task'].append('data transformed')
            progress['time'].append(time.progress_time())
            print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])
            # n_jobs is used for parallel computing 1 vs all;
            # -1 means all available cores
            clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
                tol=10**(-3),max_iter=10,n_jobs=-1,warm_start=True)
            score = cross_val_score(clf,Xtraintil,Ytrain,cv=5,n_jobs=-1)
            progress['task'].append('crossval done')
            progress['time'].append(time.progress_time())
            print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            crossval_result['score'].append(score)
            if score > best_score:
                best_score = score
                best_Gamma = Gamma
                best_Lambda = Lambda
                best_Sampler = opt_feature
                best_clf = clf
                best_Xtil = Xtraintil

    # performance test
    clf.fit(Xtraintil,Ytrain)
    progress['task'].append('best model trained')
    progress['time'].append(time.progress_time())
    print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])
    Xtesttil = best_Sampler.fit_transform(Xtest)
    Ypred = clf.predict(Xtesttil)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    progress['task'].append('test done')
    progress['time'].append(time.progress_time())
    print(progress['task'][-1],':',progress['time'][-1] - progress['time'][-2])

    # write results and log files
    classes = range(10)
    print('Classification Accuracy = ',score)
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/MNIST-cm.eps')
    plt.close(fig)
    with open('log/MNIST-log','w') as logfile:
        for idx in range(1,len(progress['task'])):
            logfile.write(progress['task'][idx]
                          + ': {:.2e}\n'.format(progress['time'][idx]
                                          - progress['time'][idx - 1]))
        logfile.write('Classification Accuracy = {:4e}'.format(score))



if __name__ == '__main__':
    main()
