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

def tfRF2L_MNIST(m=1000,n_new_features=1000,
    feature='ReLU',mode='layer 2',opt_rate=1.):
    # set up timer and progress tracker
    mylog = log.log('log/tmp.log','MNIST classification starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,
        test_size=int(m/3))
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    rand_list = np.random.permutation(Xtrain.shape[0])
    Xtr = Xtrain[rand_list[:int(m*0.9)]]
    Ytr = Ytrain[rand_list[:int(m*0.9)]]
    Xval = Xtrain[rand_list[int(m*0.9):]]
    Yval = Ytrain[rand_list[int(m*0.9):]]

    # set up parameters
    params = {
        'feature': feature,
        'n_old_features': len(Xtrain[0]),
        'n_new_features': n_new_features,
        'Lambda': np.float32(0.),
        'Gamma': np.float32(10.**(-3)),
        'classes': [0,1,2,3,4,5,6,7,8,9],
    }
    fit_params = {
        'mode': mode,
        'opt_method': 'sgd',
        'opt_rate': opt_rate,
        'batch_size': 10,
        'n_iter': m,
        'bd': 100000
    }

    # performance test
    best_clf = rff.tfRF2L(**params)
    best_clf.log = False
    mylog.time_event('model load')
    best_clf.fit(Xtr,Ytr,**fit_params)
    mylog.time_event('best model trained')
    train_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]
    Ypred,_,sparsity = best_clf.predict(Xval)
    mylog.time_event('test done')
    test_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]
    C_matrix = confusion_matrix(Yval,Ypred)
    score = np.sum(Ypred == Yval) / len(Yval)

    # plot confusion matrix
    # fig = plt.figure()
    # plot_confusion_matrix(C_matrix,classes=range(10),normalize=True)
    # plt.savefig('image/tfRF2L_{2:s}{3:s}_m_{0:.2e}_N_{1:.2e}.eps'.format(m,
    #     n_new_features,feature,mode))
    # plt.close(fig)

    print('''
    score:{0:.4f}
    sparsity:{1:.4f}
    train_time:{2:.4f}
    test_time:{3:.4f}'''.format(score,sparsity,train_time,test_time))
    return [score,sparsity,train_time,test_time]

def main():
    prefix = argv[1]
    feature = 'ReLU'
    mode = 'layer 2'
    m_max = 60000
    # run with best opt rate
    best_opt_rate = 10**1.0
    score = tfRF2L_MNIST(m=m_max,n_new_features=500, #int(np.sqrt(m)),
            feature=feature,mode=mode,opt_rate=best_opt_rate)
    np.savetxt('result/best_mnist_{0:s}{2:s}{1:s}'.format(feature,
        str(prefix),mode),np.array(score))

    # select best opt rate
    # score_list = []
    # for log_opt_rate in np.arange(-2.,3,0.5):
    #     opt_rate = 10 ** log_opt_rate
    #     score = tfRF2L_MNIST(m=m_max,n_new_features=500,
    #     feature=feature,mode=mode,opt_rate=opt_rate)
    #     score_list.append(score)
    # np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,
    #     str(prefix),mode),np.array(score_list))
    # score_list = []

    mode = 'layer 2'
    feature = 'Gaussian'
    # run with best opt rate
    best_opt_rate = 10**0.5
    score = tfRF2L_MNIST(m=m_max,n_new_features=500, #int(np.sqrt(m)),
            feature=feature,mode=mode,opt_rate=best_opt_rate)
    np.savetxt('result/best_mnist_{0:s}{2:s}{1:s}'.format(feature,
        str(prefix),mode),np.array(score))

    # for log_opt_rate in np.arange(-2.,3,0.5):
    #     opt_rate = 10 ** log_opt_rate
    #     score = tfRF2L_MNIST(m=m_max,n_new_features=500,
    #     feature=feature,mode=mode,opt_rate=opt_rate)
    #     score_list.append(score)
    # np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,
    #     str(prefix),mode),np.array(score_list))

if __name__ == '__main__':
    main()
