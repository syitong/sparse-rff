"""
This code is used to solve the famous handwritten number
recognition problem via RFSVM using svm and sgdclassifier module
in sklearn package.
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
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sys import argv
from libmnist import plot_confusion_matrix, get_train_test_data

# set up parameters
LogLambda = np.arange(-12.0,-2,20)
Gamma = 10.**(-3)

def RFSVM_MNIST(method='U',m=1000,n_components=1000):
    # set up timer and progress tracker
    mylog = log.log('log/{0:s}RFSVM_N_{1:.2e}_m_{2:.2e}.log'.format(method,
        n_components,m),
        '{:s}RFSVM MNIST classification starts'.format(method))
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,
        test_size=int(m/3))
    mylog.time_event('data read in complete')

    X_pool_fraction = 0.3
    feature_pool_size = n_components * 10

    # hyper-parameter selection
    best_score = 0
    best_Lambda = 1
    result = {'Gamma':[],'Lambda':[],'score':[]}
    for jdx in range(len(LogLambda)):
        Lambda = 10**LogLambda[jdx]
        feature = rff.optRBFSampler(Xtrain.shape[1],
            gamma=Gamma,
            feature_pool_size=feature_pool_size,
            n_components=n_components)
        # if the method requires optimized feature distribution
        if method == 'O':
            feature.reweight(Xtrain,X_pool_fraction,Lambda=Lambda)
        mylog.time_event('Gamma={0:.1e} and Lambda={1:.1e}\n'.format(Gamma,Lambda)
                         +'features generated')

        Xtraintil = feature.fit_transform(Xtrain)
        mylog.time_event('data transformed')

        # n_jobs is used for parallel computing 1 vs all;
        # -1 means all available cores
        clf = SGDClassifier(loss='hinge',penalty='l2',alpha=Lambda,
            tol=10**(-3),n_jobs=-1,warm_start=True)
        clf.fit(Xtraintil,Ytrain)
        mylog.time_event('training done')
        result['Gamma'].append(Gamma)
        result['Lambda'].append(Lambda)
        Xtesttil = feature.fit_transform(Xtest)
        Ypred = clf.predict(Xtesttil)
        score = np.sum(Ypred == Ytest) / len(Ytest)
        print('score = {:.4f}'.format(score))
        mylog.time_event('testing done')
        result['score'].append(score)
        if score > best_score:
            best_score = score
            best_Gamma = Gamma
            best_Lambda = Lambda
            best_Sampler = feature
            best_clf = clf

    # plot confusion matrix
    Xtesttil = best_Sampler.fit_transform(Xtest)
    Ypred = best_clf.predict(Xtesttil)
    C_matrix = confusion_matrix(Ytest,Ypred)
    fig = plt.figure()
    classes = range(10)
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/{0:s}RFSVM_N_{1:.2e}_m_{2:.2e}.eps'.format(method,
        n_components,m))
    plt.close(fig)

    # write results and log files
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(result['Gamma'][idx],
                                                         result['Lambda'][idx],
                                                         result['score'][idx]))
    mylog.record(results)
    mylog.save()
    return best_score

def KSVM_MNIST(m=1000):
    # set up timer and progress tracker
    mylog = log.log('log/KSVM_m_{:.2e}.log'.format(m),
        'KSVM MNIST classfication starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,
        test_size=int(m/3))
    mylog.time_event('data read in complete')

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
    best_Lambda = 1
    result = {'Gamma':[],'Lambda':[],'score':[]}
    for jdx in range(len(LogLambda)):
        Lambda = 10**LogLambda[jdx]
        C = 1 / Lambda
        clf = svm.SVC(C=C,gamma=Gamma)
        clf.fit(Xtrain,Ytrain)
        mylog.time_event('training done')
        result['Gamma'].append(Gamma)
        result['Lambda'].append(Lambda)
        Ypred = clf.predict(Xtest)
        score = np.sum(Ypred == Ytest) / len(Ytest)
        print('score = {:.4f}'.format(score))
        mylog.time_event('testing done')
        result['score'].append(score)
        if score > best_score:
            best_score = score
            best_Gamma = Gamma
            best_Lambda = Lambda
            best_clf = clf

    # plot confusion matrix
    Ypred = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    fig = plt.figure()
    classes = range(10)
    plot_confusion_matrix(C_matrix,classes=classes,normalize=True)
    plt.savefig('image/KSVM_m_{:.2e}.eps'.format(m))
    plt.close(fig)

    # write results and log files
    results = ('Best Gamma = {:.1e}\n'.format(best_Gamma)
               + 'Best Lambda = {:.1e}\n'.format(best_Lambda)
               + 'Classification Accuracy = {}\n'.format(score))
    print(results)
    results = results + 'Gamma    Lambda    score\n'
    for idx in range(len(result['Gamma'])):
        results = (results
                   + '{0:.1e}{1:9.1e}{2:10.4f}\n'.format(result['Gamma'][idx],
                                                         result['Lambda'][idx],
                                                         result['score'][idx]))
    mylog.record(results)
    mylog.save()
    return best_score

def main():
    prefix = argv[1]
    uscore_list = []
    oscore_list = []
    kscore_list = []
    for m in range(1000,60001,70000):
        score = RFSVM_MNIST(method='U',m=m,n_components=int(np.sqrt(m)))
        uscore_list.append(score)
        score = RFSVM_MNIST(method='O',m=m,n_components=int(np.sqrt(m)))
        oscore_list.append(score)
        score = KSVM_MNIST(m=m)
        kscore_list.append(score)
    np.savetxt('result/URFSVM'+str(prefix),np.array(uscore_list))
    np.savetxt('result/ORFSVM'+str(prefix),np.array(oscore_list))
    np.savetxt('result/KSVM'+str(prefix),np.array(kscore_list))

if __name__ == '__main__':
    main()
