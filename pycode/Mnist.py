"""
This code is used to solve the famous handwritten number
recognition problem via RFSVM.
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff
import log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import time

def read_MNIST_data(filepath,obs=1000):
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
        if obs > 0 and obs < dim[0]:
            dim[0] = obs
        X = np.empty((dim[0],l))
        for idx in range(dim[0]):
            for jdx in range(l):
                index = offset + idx * l + jdx
                X[idx,jdx] = data[index]
        if l == 1:
            X = X[:,0]
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

def ORFSVM_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/ORFSVM_MNIST_{}.log'.format(n_components),'MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller dataset for cross validation
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    # normalize features
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    Xtr = scaler.transform(Xtr)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma
    X_pool_fraction = 0.3
    feature_pool_size = n_components * 10

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in range(len(LogGamma)):
        Gamma = 10**LogGamma[idx]
        for jdx in range(len(LogLambda)):
            Lambda = 10**LogLambda[jdx]
            opt_feature = rff.optRBFSampler(Xtrain.shape[1],
                gamma=Gamma,
                feature_pool_size=feature_pool_size,
                n_components=n_components)
            mylog.time_event('feature pool generated')
            opt_feature.reweight(Xtrain,X_pool_fraction,Lambda=Lambda)
            mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                             +'features generated')
            Xtraintil = opt_feature.fit_transform(Xtrain)
            mylog.time_event('data transformed')
            # n_jobs is used for parallel computing 1 vs all;
            # -1 means all available cores
            clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
                tol=10**(-3),n_jobs=-1,warm_start=True)
            score = cross_val_score(clf,Xtraintil,Ytrain,cv=5,n_jobs=-1)
            mylog.time_event('crossval done')
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            avg_score = np.sum(score) / 5
            print('score = {:.4f}'.format(avg_score))
            crossval_result['score'].append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_Gamma = Gamma
                best_Lambda = Lambda
                best_Sampler = opt_feature
                best_clf = clf

    # performance test
    best_Xtil = best_Sampler.fit_transform(Xtr)
    Xtesttil = best_Sampler.fit_transform(Xtest)
    best_clf.fit(best_Xtil,Ytr)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtesttil)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/ORFSVM_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)

def URFSVM_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/URFSVM_MNIST_{}.log'.format(n_components),'MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in range(len(LogGamma)):
        Gamma = 10**LogGamma[idx]
        for jdx in range(len(LogLambda)):
            Lambda = 10**LogLambda[jdx]
            unif_feature = rff.myRBFSampler(Xtrain.shape[1],
                gamma=Gamma,n_components=n_components)
            mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                             +'features generated')
            Xtraintil = unif_feature.fit_transform(Xtrain)
            mylog.time_event('data transformed')
            # n_jobs is used for parallel computing 1 vs all;
            # -1 means all available cores
            clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
                tol=10**(-3),n_jobs=-1,warm_start=True)
            score = cross_val_score(clf,Xtraintil,Ytrain,cv=5,n_jobs=-1)
            mylog.time_event('crossval done')
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            avg_score = np.sum(score) / 5
            print('score = {:.4f}'.format(avg_score))
            crossval_result['score'].append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_Gamma = Gamma
                best_Lambda = Lambda
                best_Sampler = unif_feature
                best_clf = clf

    # performance test
    best_Xtil = best_Sampler.fit_transform(Xtr)
    Xtesttil = best_Sampler.fit_transform(Xtest)
    best_clf.fit(best_Xtil,Ytr)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtesttil)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/URFSVM_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)

def URFMLR_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/URFMLR_MNIST_{}.log'.format(n_components),'MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in range(len(LogGamma)):
        Gamma = 10**LogGamma[idx]
        for jdx in range(len(LogLambda)):
            Lambda = 10**LogLambda[jdx]
            unif_feature = rff.myRBFSampler(Xtrain.shape[1],
                gamma=Gamma,n_components=n_components)
            mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                             +'features generated')
            Xtraintil = unif_feature.fit_transform(Xtrain)
            mylog.time_event('data transformed')
            # n_jobs is used for parallel computing 1 vs all;
            # -1 means all available cores
            clf = LogisticRegression(penalty='l2',C=1 / m / Lambda,
                multi_class = 'multinomial',solver='sag',tol=10**(-3),
                warm_start=True)
            score = cross_val_score(clf,Xtraintil,Ytrain,cv=5,n_jobs=-1)
            mylog.time_event('crossval done')
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            avg_score = np.sum(score) / 5
            print('score = {:.4f}'.format(avg_score))
            crossval_result['score'].append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_Gamma = Gamma
                best_Lambda = Lambda
                best_Sampler = unif_feature

    # performance test
    best_Xtil = best_Sampler.fit_transform(Xtr)
    Xtesttil = best_Sampler.fit_transform(Xtest)
    m = len(best_Xtil)
    best_clf = LogisticRegression(penalty='l2',C=1 / m / best_Lambda,
                multi_class = 'multinomial',solver='sag',tol=10**(-3),
                warm_start=True)
    best_clf.fit(best_Xtil,Ytr)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtesttil)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/URFMLR_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)

def KSVM_MNIST(m=1000,trainsize=1000):
    # set up timer and progress tracker
    mylog = log.log('log/KSVM_MNIST_{}.log'.format(trainsize),'KSVM MNIST classfication starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',trainsize)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',trainsize)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma
    cv = 5 # cross validation folds

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in range(len(LogGamma)):
        Gamma = 10**LogGamma[idx]
        for jdx in range(len(LogLambda)):
            Lambda = 10**LogLambda[jdx]
            C = 1 / Lambda / ((cv - 1) * m / cv)
            clf = svm.SVC(C=C,gamma=Gamma)
            score = cross_val_score(clf,Xtrain,Ytrain,cv=cv,n_jobs=-1)
            mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                             +'crossval done')
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            avg_score = np.sum(score) / 5
            print('score = {:.4f}'.format(avg_score))
            crossval_result['score'].append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_Gamma = Gamma
                best_Lambda = Lambda

    # performance test
    C = 1 / best_Lambda / trainsize
    best_clf = svm.SVC(C=C,gamma=best_Gamma)
    best_clf.fit(Xtr,Ytr)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/KSVM_MNIST_{}-cm.eps'.format(trainsize))
    plt.close(fig)

def HRFSVM_MNIST():
    # set up timer and progress tracker
    mylog = log.log('log/HRFSVM_MNIST.log','MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xts = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Yts = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    # m = 1000
    Xtrain = Xtr
    Ytrain = Ytr
    Xtest = Xts
    Ytest = Yts
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-6.0,-1,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.1,0.4,0.1)
    LogGamma = np.log10(gamma) + LogGamma
    n_components = 5000

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    for idx in range(len(LogGamma)):
        Gamma = 10**LogGamma[idx]
        for jdx in range(len(LogLambda)):
            Lambda = 10**LogLambda[jdx]
            # n_jobs is used for parallel computing 1 vs all;
            # -1 means all available cores
            clf = rff.HRFSVM(n_components=n_components,
                gamma=Gamma,p=0,alpha=Lambda,n_jobs=-1)
            mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                             +'features generated')
            score = cross_val_score(clf,Xtrain,Ytrain,cv=5,n_jobs=-1,scoring='accuracy')
            mylog.time_event('crossval done')
            crossval_result['Gamma'].append(Gamma)
            crossval_result['Lambda'].append(Lambda)
            avg_score = np.sum(score) / 5
            print('score = {:.4f}'.format(avg_score))
            crossval_result['score'].append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                best_Gamma = Gamma
                best_Lambda = Lambda
                best_clf = clf

    # performance test
    best_clf.fit(Xtrain,Ytrain)
    mylog.time_event('best model trained')
    Ypred = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/HRFSVM_MNIST-cm.eps')
    plt.close(fig)

def tfRFLM_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/tfURFMLR_MNIST_{}.log'.format(n_components),'MNIST classification starts')

    # read in MNIST data set
    Xtr = read_MNIST_data('data/train-images.idx3-ubyte',-1)
    Ytr = read_MNIST_data('data/train-labels.idx1-ubyte',-1)
    Xtest = read_MNIST_data('data/t10k-images.idx3-ubyte',-1)
    Ytest = read_MNIST_data('data/t10k-labels.idx1-ubyte',-1)
    mylog.time_event('data read in complete')

    # extract a smaller data set
    Xtrain = Xtr[:m]
    Ytrain = Ytr[:m]
    scaler = StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtr = scaler.transform(Xtr)
    Xtest = scaler.transform(Xtest)

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,0.1)
    LogGamma = np.log10(gamma) + LogGamma
    params = {
        'n_old_features': len(Xtrain[0]),
        'n_components': n_components,
        'lambda': 10.**(-6),
        'gamma': 10.**LogGamma[2],
        'n_classes': 10,
        'method': 'lr'
    }

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    # for idx in range(len(LogGamma)):
    #     Gamma = 10**LogGamma[idx]
    #     for jdx in range(len(LogLambda)):
    #         Lambda = 10**LogLambda[jdx]
    #         params['lambda'] = Lambda
    #         params['gamma'] = gamma
    #         clf = rff.tfRFLM(params)
    #         score = cross_val_score(clf,Xtrain,Ytrain,cv=3
    #             # ,n_jobs=-1
    #             )
    #         mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
    #                          +'crossval done')
    #         crossval_result['Gamma'].append(Gamma)
    #         crossval_result['Lambda'].append(Lambda)
    #         avg_score = np.sum(score) / 5
    #         print('score = {:.4f}'.format(avg_score))
    #         crossval_result['score'].append(avg_score)
    #         if avg_score > best_score:
    #             best_score = avg_score
    #             best_Gamma = Gamma
    #             best_Lambda = Lambda
    #             best_clf = clf

    # performance test
    best_clf = rff.tfRFLM(params)
    best_clf.fit(Xtr,Ytr,n_iter=2000)
    mylog.time_event('best model trained')
    Ypred = [p['classes'] for _, p in enumerate(best_clf.infer(Xtest,predict_keys='classes'))]
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')

    # write results and log files
    classes = range(10)
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(crossval_result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(crossval_result['Gamma'][idx],
                                                         crossval_result['Lambda'][idx],
                                                         crossval_result['score'][idx]))
    mylog.record(results)
    mylog.save()

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/tfURFMLR_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)


def main():
    # URFSVM_MNIST(m=1000,n_components=500)
    # ORFSVM_MNIST(m=1000,n_components=500)
    # URFSVM_MNIST(m=1000,n_components=1000)
    # ORFSVM_MNIST(m=1000,n_components=1000)
    # KSVM_MNIST(m=1000,trainsize=2000)
    # URFMLR_MNIST(m=1000,n_components=2000)
    tfRFLM_MNIST(m=1000,n_components=2000)

if __name__ == '__main__':
    main()
