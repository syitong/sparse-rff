import numpy as np
import librf
import time
from sys import argv
from libmnist import get_train_test_data
from multiprocessing import Pool
from functools import partial
from uci_pre import read_data
from result_show import print_params

def _validate(data,labels,folds,index,**params):
    kfolds_data = np.split(data,folds)
    kfolds_labels = np.split(labels,folds)
    Xts = kfolds_data.pop(index)
    Yts = kfolds_labels.pop(index)
    Xtr = np.concatenate(kfolds_data)
    Ytr = np.concatenate(kfolds_labels)
    modelparams = params['model']
    clf = librf.RF(**modelparams)
    # clf.log = True
    fitparams = params['fit']
    clf.fit(Xtr,Ytr,**fitparams)
    score = clf.score(Xts,Yts)
    return score

def validate(data,labels,val_size,folds=5,**params):
    # set up timer and progress tracker
    rand_list = np.random.permutation(len(data))
    X = data[rand_list[:val_size]]
    Y = labels[rand_list[:val_size]]
    f = partial(_validate,X,Y,folds,**params)
    # with Pool() as p:
    #     score_list = p.map(f,range(folds))
    score_list = []
    for idx in range(folds):
        score_list.append(f(idx))
    return sum(score_list) / folds

def train_and_test(dataset):
    if dataset == 'mnist':
        Xtrain,Ytrain,Xtest,Ytest = get_train_test_data()
        N = 2000 # 10000
        bd = 1000 # 100000
        n_iter = 1000 # 5000
        classes = list(range(10))
        loss_fn = 'log'
        F_gamma,F_rate,R_rate = print_params(dataset)
    elif dataset == 'adult':
        Xtrain = read_data('adult-train-data.npy')
        Ytrain = read_data('adult-train-label.npy')
        Xtest = read_data('adult-test-data.npy')
        Ytest = read_data('adult-test-label.npy')
        N = 2000 # 10000
        bd = 1000 # 100000
        n_iter = 1000 # 5000
        Gamma_list = 10. ** np.arange(-6.,2,1) # np.arange(-2.,4,0.5)
        rate_list = 10. ** np.arange(-2.,4,0.5) # np.arange(0.8,2.8,0.2)
        classes = [0.,1.]
        loss_fn = 'hinge'
    elif dataset == 'covtype':
        Xtrain = read_data('covtype-train-data.npy')
        # Ytrain = read_data('covtype-train-binary-label.npy')
        Xtest = read_data('covtype-test-data.npy')
        # Ytest = read_data('covtype-test-binary-label.npy')
        Ytrain = read_data('covtype-train-label.npy')
        Ytest = read_data('covtype-test-label.npy')
        N = 10000
        bd = 100000
        n_iter = 5000
        Gamma_list = 10. ** np.arange(-2.,4,0.5)
        rate_list = 10. ** np.arange(0.8,2.8,0.2)
        classes = [0.,1.] # list(range(1,8)) # list(range(10))
        loss_fn = 'log'
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    prefix = argv[1]
    params = {}
    feature = 'Gaussian'
    params['model'] = {
        'feature':feature,
        'n_old_features':len(Xtrain[0]),
        'n_new_features':N,
        'classes':classes,
        'loss_fn':loss_fn
    }
    params['fit'] = {
        'opt_rate':rate_list[int(prefix)],
        'n_iter':n_iter,
        'bd':bd
    }
    modelparams = params['model']
    clf = librf.RF(**modelparams)
    fitparams = params['fit']
    t1 = time.process_time()
    clf.fit(Xtr,Ytr,**fitparams)
    t2 = time.process_time()
    Ypr,_,sparsity = clf.predict(Xts)
    t3 = time.process_time()
    score = sum(Ypr == Yts) / len(Yts)
    return score,sparsity,t2-t1,t3-t2

def screen_params(dataset,val_size=30000,folds=5):
    if dataset == 'mnist':
        Xtrain,Ytrain,Xtest,Ytest = get_train_test_data()
        N = 2000 # 10000
        bd = 1000 # 100000
        n_iter = 1000 # 5000
        Gamma_list = 10. ** np.arange(-6.,2,1) # np.arange(-2.,4,0.5)
        rate_list = 10. ** np.arange(-2.,4,0.5) # np.arange(0.8,2.8,0.2)
        classes = list(range(10))
        loss_fn = 'log'
    elif dataset == 'adult':
        Xtrain = read_data('adult-train-data.npy')
        Ytrain = read_data('adult-train-label.npy')
        Xtest = read_data('adult-test-data.npy')
        Ytest = read_data('adult-test-label.npy')
        N = 2000 # 10000
        bd = 1000 # 100000
        n_iter = 1000 # 5000
        Gamma_list = 10. ** np.arange(-6.,2,1) # np.arange(-2.,4,0.5)
        rate_list = 10. ** np.arange(-2.,4,0.5) # np.arange(0.8,2.8,0.2)
        classes = [0.,1.]
        loss_fn = 'hinge'
    elif dataset == 'covtype':
        Xtrain = read_data('covtype-train-data.npy')
        # Ytrain = read_data('covtype-train-binary-label.npy')
        Xtest = read_data('covtype-test-data.npy')
        # Ytest = read_data('covtype-test-binary-label.npy')
        Ytrain = read_data('covtype-train-label.npy')
        Ytest = read_data('covtype-test-label.npy')
        N = 10000
        bd = 100000
        n_iter = 5000
        Gamma_list = 10. ** np.arange(-2.,4,0.5)
        rate_list = 10. ** np.arange(0.8,2.8,0.2)
        classes = [0.,1.] # list(range(1,8)) # list(range(10))
        loss_fn = 'log'

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    prefix = argv[1]
    params = {}
    feature = 'Gaussian'
    params['model'] = {
        'feature':feature,
        'n_old_features':len(Xtrain[0]),
        'n_new_features':N,
        'classes':classes,
        'loss_fn':loss_fn
    }
    params['fit'] = {
        'opt_rate':rate_list[int(prefix)],
        'n_iter':n_iter,
        'bd':bd
    }
    results = []
    for Gamma in Gamma_list:
        params['model']['Gamma'] = Gamma
        score = validate(Xtrain,Ytrain,val_size,folds,**params)
        results.append({'Gamma':Gamma,'score':score})
    feature = 'ReLU'
    params['model']['feature'] = feature
    score = validate(Xtrain,Ytrain,val_size,folds,**params)
    results.append({'Gamma':'ReLU','score':score})
    filename = 'result/{1:s}-{0:s}'.format(prefix,dataset)
    with open(filename,'w') as f:
        f.write(str(results))

if __name__ == '__main__':
    screen_params()
