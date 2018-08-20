import numpy as np
import librf
from sys import argv
from libmnist import get_train_test_data
from multiprocessing import Pool
from functools import partial

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

def main():
    val_size = 3000 # 30000
    folds = 2 # 5
    Xtrain,Ytrain,Xtest,Ytest = get_train_test_data()
    Gamma_list = [1.,2.] # 10. ** np.arange(-6.,2,1)
    rate_list = [1.,10.] # 10. ** np.arange(-2.,3,0.5)
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    prefix = argv[1]
    params = {}
    feature = 'Gaussian'
    params['model'] = {
        'feature':feature,
        'n_old_features':len(Xtrain[0]),
        'n_new_features':2000,
        'classes':list(range(10)),
        'loss_fn':'log'
    }
    params['fit'] = {
        'opt_rate':rate_list[int(prefix)],
        'n_iter':1000,
        'bd':1000
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
    filename = 'result/mnist_{0:s}-{1:s}'.format(feature,
        prefix)
    with open(filename,'w') as f:
        f.write(str(results))

if __name__ == '__main__':
    main()
