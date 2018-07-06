import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff
import log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sys import argv
from libmnist import plot_confusion_matrix, get_train_test_data

def tfURF2L_MNIST(m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/tfnnRF2Ldropout_MNIST_{}.log'.format(n_components),'MNIST classification starts')

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
    LogLambda = np.arange(-12.0,-2,1.)
    gamma = rff.gamma_est(Xtrain)
    LogGamma = np.arange(-0.2,0.8,.1)
    LogGamma = np.log10(gamma) + LogGamma
    params = {
        'feature': 'ReLU',
        'n_old_features': len(Xtrain[0]),
        'n_components': n_components,
        # 'Lambda': np.float32(10.**(-6)),
        'Lambda': np.float32(0.),
        'Gamma': np.float32(10.**LogGamma[2]),
        'classes': [0,1,2,3,4,5,6,7,8,9],
    }
    fit_params = {
        'mode': 'layer 2',
        'batch_size': 1,
        'n_iter': 5000
    }

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 10.**LogGamma[2]
    # best_Lambda = 10.**(-6)
    best_Lambda = 0.
    crossval_result = {'Gamma':[],'Lambda':[],'score':[]}
    # for idx in range(len(LogGamma)):
    #     Gamma = np.float32(10**LogGamma[idx])
    #     for jdx in range(len(LogLambda)):
    #         Lambda = np.float32(10**LogLambda[jdx])
    #         params['Lambda'] = Lambda
    #         params['Gamma'] = Gamma
    #         clf = rff.tfRF2L(**params)
    #         score = cross_val_score(clf,Xtrain,Ytrain,fit_params=fit_params,cv=5)
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
    best_clf = rff.tfRF2L(**params)
    best_clf.log = True
    best_clf.fit(Xtr,Ytr,mode='layer 2',batch_size=100,n_iter=7000)
    # best_clf.fit(Xtr,Ytr,mode='layer 1',batch_size=100,n_iter=7000)
    # best_clf.fit(Xtr,Ytr,mode='over all',batch_size=100,n_iter=7000)
    mylog.time_event('best model trained')
    Ypred,_ = best_clf.predict(Xtest)
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
    plt.savefig('image/tfURF2L_ReLU_MNIST_{}-cm.eps'.format(n_components))
    plt.close(fig)
