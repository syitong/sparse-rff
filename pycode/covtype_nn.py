import numpy as np
import rff
import log
from sys import argv
from uci_pre import read_data

Xtrain = read_data('covtype-train-data.npy')
# Ytrain = read_data('covtype-train-label.npy')
Ytrain = read_data('covtype-train-binary-label.npy')
print(len(Ytrain))
Xtest = read_data('covtype-test-data.npy')
# Ytest = read_data('covtype-test-label.npy')
Ytest = read_data('covtype-test-binary-label.npy')

def covtype_nn(m=1000,n_new_features=1000,feature='ReLU',
    mode='layer 2',loss_fn='log',opt_rate=1.,bd = 1000):
    # set up timer and progress tracker
    mylog = log.log('log/tmp.log','Covtype classification starts')
    rand_list = np.random.permutation(Xtrain.shape[0])
    Xtr = Xtrain[rand_list[:int(m*0.9)]]
    Ytr = Ytrain[rand_list[:int(m*0.9)]]
    Xval = Xtrain[rand_list[int(m*0.9):m]]
    Yval = Ytrain[rand_list[int(m*0.9):m]]
    # set up parameters
    params = {
        'feature': feature,
        'n_old_features': len(Xtrain[0]),
        'n_new_features': n_new_features,
        'loss_fn': loss_fn,
        'Lambda': np.float32(0.),
        'Gamma': np.float32(0.27), # rff.gamma_est(Xtrain)
        'classes': [0,1]
        # 'classes': [1,2,3,4,5,6,7]
    }
    # print(rff.gamma_est(Xtrain))
    fit_params = {
        'mode': mode,
        'opt_method': 'sgd',
        'opt_rate': opt_rate,
        'batch_size': 50,
        'n_iter': m,
        'bd': bd
    }

    # performance test
    best_clf = rff.tfRF2L(**params)
    best_clf.log = True
    mylog.time_event('model load')
    best_clf.fit(Xtr,Ytr,**fit_params)
    mylog.time_event('best model trained')
    train_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]
    Ypred,_,sparsity = best_clf.predict(Xval,50)
    # Ypred,_,sparsity = best_clf.predict(Xtest,50)
    mylog.time_event('test done')
    test_time = mylog.progress['time'][-1] - mylog.progress['time'][-2]
    score = np.sum(Ypred == Yval) / len(Yval)
    # score = np.sum(Ypred == Ytest) / len(Ytest)

    print('''
    score:{0:.4f}
    sparsity:{1:.4f}
    train_time:{2:.4f}
    test_time:{3:.4f}'''.format(score,sparsity,train_time,test_time))
    return [score,sparsity,train_time,test_time]

def main():
    prefix = argv[1]
    m_max = 5229#11
    feature = 'ReLU'
    mode = 'layer 2'
    # run with best opt rate
    best_opt_rate = 10**2.0
    score = covtype_nn(m=m_max,n_new_features=500, #int(np.sqrt(m)),
            feature=feature,mode=mode,opt_rate=best_opt_rate,bd=1000)
    np.savetxt('result/best_covtype_b_{0:s}{2:s}{1:s}'.format(feature,
        str(prefix),mode),np.array(score))

    # select best opt rate
    # score_list = []
    # for log_opt_rate in np.arange(-2.,3.,0.5):
    #     opt_rate = 10**log_opt_rate
    #     score = covtype_nn(m=m_max,n_new_features=5000, #int(np.sqrt(m)),
    #         feature=feature,mode=mode,opt_rate=opt_rate)
    #     score_list.append(score)
    # np.savetxt('result/covtype_{0:s}{2:s}{1:s}'.format(feature,
    #     str(prefix),mode),np.array(score_list))

    mode = 'layer 2'
    feature = 'Gaussian'
    # run with best opt rate
    best_opt_rate = 10**1.0
    score = covtype_nn(m=m_max,n_new_features=500, #int(np.sqrt(m)),
        feature=feature,mode=mode,opt_rate=best_opt_rate,bd=1000)
    np.savetxt('result/best_covtype_b_{0:s}{2:s}{1:s}'.format(feature,
        str(prefix),mode),np.array(score))

    # select best opt rate
    # score_list = []
    # for log_opt_rate in np.arange(-2.,3.,0.5):
    #     opt_rate = 10**log_opt_rate
    #     score = covtype_nn(m=m_max,n_new_features=5000, #int(np.sqrt(m)),
    #         feature=feature,mode=mode,opt_rate=opt_rate)
    #     score_list.append(score)
    # np.savetxt('result/covtype_{0:s}{2:s}{1:s}'.format(feature,
    #     str(prefix),mode),np.array(score_list))

if __name__ == '__main__':
    main()
