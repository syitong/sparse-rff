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
    mylog = log.log('log/tmp.log','MNIST classification starts')
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data(train_size=m,
        test_size=int(m/3))

    # set up parameters
    params = {
        'feature': feature,
        'n_old_features': len(Xtrain[0]),
        'n_components': n_components,
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

    # performance test
    best_clf = rff.tfRF2L(**params)
    best_clf.log = False
    best_clf.fit(Xtrain,Ytrain,**fit_params)
    mylog.time_event('best model trained')
    train_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]
    Ypred,_,sparsity = best_clf.predict(Xtest)
    C_matrix = confusion_matrix(Ytest,Ypred)
    score = np.sum(Ypred == Ytest) / len(Ytest)
    mylog.time_event('test done')
    test_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]

    # plot confusion matrix
    fig = plt.figure()
    plot_confusion_matrix(C_matrix,classes=range(10),normalize=True)
    plt.savefig('image/tfRF2L_{2:s}{3:s}_m_{0:.2e}_N_{1:.2e}.eps'.format(m,
        n_components,feature,mode))
    plt.close(fig)

    print('''
    score:{0:.4f}
    sparsity:{1:.4f}
    train_time:{2:.4f}
    test_time:{3:.4f}'''.format(score,sparsity,train_time,test_time))
    return [score,sparsity,train_time,test_time]

def main():
    prefix = argv[1]
    score_list = []
    feature = 'ReLU'
    mode = 'layer 2'
    increment = 500000
    for m in range(5000,60001,increment):
        score = tfRF2L_MNIST(m=m,n_components=int(np.sqrt(m)),feature=feature,mode=mode)
        score_list.append(score)
    np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,str(prefix),mode),np.array(score_list))
    score_list = []
    mode = 'layer 2'
    feature = 'Gaussian'
    for m in range(5000,60001,increment):
        score = tfRF2L_MNIST(m=m,n_components=int(np.sqrt(m)),feature=feature,mode=mode)
        score_list.append(score)
    np.savetxt('result/tfRF2L_{0:s}{2:s}{1:s}'.format(feature,str(prefix),mode),np.array(score_list))

if __name__ == '__main__':
    main()
