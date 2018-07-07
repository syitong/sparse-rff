import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import rff
import log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sys import argv
from libmnist import plot_confusion_matrix, get_train_test_data

def tfRF2L_MNIST(m=1000,n_components=1000,feature='ReLU',mode='layer 2'):
    # set up timer and progress tracker
    mylog = log.log('log/tfRF2L_{2:s}{3:s}_m_{0:.2e}_N_{1:.2e}.log'.format(m,
        n_components,feature,mode),
        'MNIST classification starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,
        test_size=int(m/3))
    mylog.time_event('data read in complete')

    # set up parameters
    LogLambda = np.arange(-12.0,-2,1.)
    # gamma = rff.gamma_est(Xtrain)
    # LogGamma = np.arange(-0.2,0.8,.1)
    # LogGamma = np.log10(gamma) + LogGamma
    params = {
        'feature': feature,
        'n_old_features': len(Xtrain[0]),
        'n_components': n_components,
        # 'Lambda': np.float32(10.**(-6)),
        'Lambda': np.float32(0.),
        'Gamma': np.float32(10.**(-3)),
        'classes': [0,1,2,3,4,5,6,7,8,9],
    }
    fit_params = {
        'mode': mode,
        'opt_method': 'sgd',
        'opt_rate': 0.5,
        'batch_size': 10,
        'n_iter': m
    }

    # hyper-parameter selection
    best_score = 0
    best_Gamma = 1
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
    best_clf.fit(Xtrain,Ytrain,**fit_params)
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
    plt.savefig('image/tfRF2L_{2:s}{3:s}_m_{0:.2e}_N_{1:.2e}.eps'.format(m,
        n_components,feature,mode))
    plt.close(fig)

    return score

def main():
    prefix = argv[1]
    score_list = []
    feature = 'ReLU'
    mode = 'over all'
    increment = 5000
    for m in range(1000,60001,increment):
        score = tfRF2L_MNIST(m=m,n_components=int(np.sqrt(m)),feature=feature,mode=mode)
        score_list.append(score)
    np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,str(prefix),mode),np.array(score_list))
    score_list = []
    feature = 'Gaussian'
    for m in range(1000,60001,increment):
        score = tfRF2L_MNIST(m=m,n_components=int(np.sqrt(m)),feature=feature,mode=mode)
        score_list.append(score)
    np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,str(prefix),mode),np.array(score_list))

if __name__ == '__main__':
    main()
