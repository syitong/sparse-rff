import numpy as np
import librf
import libnn
import time
from datetime import datetime
from log import log
from sys import argv
from sklearn.preprocessing import StandardScaler
from libmnist import get_train_test_data
from multiprocessing import Pool
from functools import partial
from result_show import print_params
from sklearn.svm import SVC
DATA_PATH = 'data/'

def read_params(filename='params'):
    with open(filename,'r') as f:
        params = eval(f.read())
    return params

def _read_data(filename):
    data = np.load(DATA_PATH + filename)
    return data

def _unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data(dataset):
    if dataset == 'mnist':
        Xtr,Ytr,Xts,Yts = get_train_test_data()
    elif dataset == 'cifar':
        X = []
        Y = []
        for idx in range(1): # set to 5 for the complete data set
            X.append(unpickle('data/cifar-10/data_batch_'+idx)[b'data'])
            Y.append(unpickle('data/cifar-10/data_batch_'+idx)[b'labels'])
        Xtr = np.concatenate(X,axis=0)
        Ytr = np.concatenate(Y,axis=0)
        Xts = unpickle('data/cifar-10/test_batch')[b'data']
        Yts = unpickle('data/cifar-10/test_batch')[b'labels']
    else:
        Xtr = _read_data(dataset+'-train-data.npy')
        Ytr = _read_data(dataset+'-train-label.npy')
        Xts = _read_data(dataset+'-test-data.npy')
        Yts = _read_data(dataset+'-test-label.npy')
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xts = scaler.transform(Xts)
    return Xtr,Ytr,Xts,Yts

def _validate(data,labels,folds,model_type,model_params,fit_params,index):
    kfolds_data = np.split(data,folds)
    kfolds_labels = np.split(labels,folds)
    Xts = kfolds_data.pop(index)
    Yts = kfolds_labels.pop(index)
    Xtr = np.concatenate(kfolds_data)
    Ytr = np.concatenate(kfolds_labels)
    clf = model_type(**model_params)
    clf.fit(Xtr,Ytr,**fit_params)
    score = clf.score(Xts,Yts)
    return score

def validate(data,labels,val_size,model_type,model_params,fit_params,folds=5):
    # set up timer and progress tracker
    rand_list = np.random.permutation(len(data))
    X = data[rand_list[:val_size]]
    Y = labels[rand_list[:val_size]]
    f = partial(_validate,X,Y,folds,model_type,model_params,fit_params)
    # with Pool() as p:
    #     score_list = p.map(f,range(folds))
    score_list = []
    for idx in range(folds):
        score_list.append(f(idx))
    return sum(score_list) / folds

def _train_and_test(Xtr,Ytr,Xts,Yts,model_type,model_params,fit_params):
    clf = model_type(**model_params)
    t1 = time.process_time()
    clf.fit(Xtr,Ytr,**fit_params)
    t2 = time.process_time()
    Ypr,_,sparsity = clf.predict(Xts)
    t3 = time.process_time()
    score = sum(Ypr == Yts) / len(Yts)
    return score,sparsity,t2-t1,t3-t2

def train_and_test(dataset,feature,params='auto'):
    '''
    params = {
        'dataset': ,
        'N': ,
        'bd': ,
        'n_epoch': ,
        'classes': ,
        'loss_fn': ,
        'feature': ,
        }
    '''
    prefix = argv[1]
    if params == 'auto':
        logGamma,lograte,params = print_params(dataset,feature)
    Xtrain,Ytrain,Xtest,Ytest = read_data(dataset)
    model_params = {
        'n_old_features':len(Xtrain[0]),
        'n_new_features':params['N'],
        'classes':params['classes'],
        'loss_fn':params['loss_fn']
    }
    fit_params = {
        'n_epoch':params['n_epoch'],
        'bd':params['bd']
    }
    model_params['Gamma'] = 10. ** logGamma
    model_params['feature'] = feature
    fit_params['opt_rate'] = 10. ** lograte
    if prefix == '0':
        # only write log file for trial 0
        logfile = log('log/experiments.log','train and test')
        logfile.record(str(datetime.now()))
        logfile.record('{0} = {1}'.format('dataset',dataset))
        for key,val in model_params.items():
            logfile.record('{0} = {1}'.format(key,val))
        for key,val in fit_params.items():
            logfile.record('{0} = {1}'.format(key,val))
        logfile.save()
    model_type = librf.RF
    score1,sparsity1,traintime1,testtime1 = _train_and_test(Xtrain,
        Ytrain,Xtest,Ytest,model_type,model_params,fit_params)
    output = {
            'accuracy':score1,
            'sparsity':sparsity1,
            'traintime':traintime1,
            'testtime':testtime1
        }
    filename = 'result/{0:s}-{1:s}-test-{2:s}'.format(dataset,feature,prefix)
    finalop = [output,dataset,model_params,fit_params]
    with open(filename,'w') as f:
        f.write(str(finalop))

def screen_params(params):
    '''
    params = {
        'dataset': ,
        'N': ,
        'bd': ,
        'n_epoch': ,
        'classes': ,
        'loss_fn': ,
        'feature': ,
        'logGamma': ,
        'lograte': ,
        'val_size': ,
        'folds': ,
        }
    '''
    dataset = params['dataset']
    prefix = argv[1]
    val_size = params['val_size']
    folds = params['folds']
    if prefix == '0':
        # only write log file for trial 0
        logfile = log('log/experiments.log','screen params')
        logfile.record(str(datetime.now()))
        for key,val in params.items():
            logfile.record('{0} = {1}'.format(key,val))
        logfile.save()
    Xtrain,Ytrain,_,_ = read_data(dataset)
    feature = params['feature']
    model_params = {
        'n_old_features':len(Xtrain[0]),
        'n_new_features':params['N'],
        'classes':params['classes'],
        'loss_fn':params['loss_fn'],
        'feature':feature,
    }
    fit_params = {
        'opt_method':'sgd',
        'n_epoch':params['n_epoch'],
        'opt_rate':10.**params['lograte'][int(prefix)],
        'bd':params['bd']
    }
    model_type = librf.RF
    results = []
    Gamma_list = 10. ** params['logGamma']
    for Gamma in Gamma_list:
        model_params['Gamma'] = Gamma
        score = validate(Xtrain,Ytrain,val_size,model_type,
            model_params, fit_params, folds)
        results.append({'Gamma':Gamma,'score':score})
    filename = 'result/{0:s}-{1:s}-screen-{2:s}'.format(dataset,feature,prefix)
    with open(filename,'w') as f:
        f.write(str(results))

# def screen_params_fnn_covtype(val_size=30000,folds=5):
#     Xtrain = read_data('covtype-train-data.npy')
#     # Ytrain = read_data('covtype-train-binary-label.npy')
#     Xtest = read_data('covtype-test-data.npy')
#     # Ytest = read_data('covtype-test-binary-label.npy')
#     Ytrain = read_data('covtype-train-label.npy')
#     Ytest = read_data('covtype-test-label.npy')
#     rate_list = 10. ** np.arange(-3.,3,0.5) # np.arange(0.8,2.8,0.2)
#     classes = list(range(1,8)) # list(range(10))
#     loss_fn = 'log'
# 
#     prefix = argv[1]
#     params = {}
#     params['model'] = {
#         'dim':len(Xtrain[0]),
#         'width':20,
#         'depth':4,
#         'classes':classes,
#         'learn_rate':rate_list[int(prefix)]
#     }
#     params['fit'] = {
#         'n_epoch':3,
#         'batch_size':100
#     }
#     model_type = libnn.fullnn
#     score = validate(Xtrain,Ytrain,val_size,model_type,folds,**params)
#     filename = 'result/covtype-nn-{0:s}'.format(prefix)
#     with open(filename,'w') as f:
#         f.write(str(score))
# 
# def screen_params_svm_covtype(val_size=30000,folds=5):
#     Xtrain = read_data('covtype-train-data.npy')
#     Ytrain = read_data('covtype-train-binary-label.npy')
#     Xtest = read_data('covtype-test-data.npy')
#     Ytest = read_data('covtype-test-binary-label.npy')
#     C_list = 10. ** np.arange(2.,6,1)
#     gamma = 10. ** 1.5
#     prefix = argv[1]
#     params = {}
#     params['model'] = {
#         'C':C_list[int(prefix)],
#         'gamma':gamma
#     }
#     params['fit'] = {}
#     model_type = SVC
#     score = validate(Xtrain,Ytrain,val_size,model_type,folds,**params)
#     filename = 'result/covtype-svm-{0:s}'.format(prefix)
#     with open(filename,'w') as f:
#         f.write(str(score))
# 
# def train_test_covtype_nn():
#     Xtrain = read_data('covtype-train-data.npy')
#     Ytrain = read_data('covtype-train-binary-label.npy')
#     Xtest = read_data('covtype-test-data.npy')
#     Ytest = read_data('covtype-test-binary-label.npy')
#     # Ytrain = read_data('covtype-train-label.npy')
#     # Ytest = read_data('covtype-test-label.npy')
#     rate_list = 10. ** np.arange(-3.,3,0.5) # np.arange(0.8,2.8,0.2)
#     classes = [0.,1.] # list(range(1,8))
#     loss_fn = 'log'
#     modelparams = {
#         'dim':len(Xtrain[0]),
#         'width':50,
#         'depth':5,
#         'classes':classes,
#         'learn_rate':rate_list[0]
#     }
#     fitparams = {
#         'n_epoch':5,
#         'batch_size':100
#     }
#     model_type = libnn.fullnn
#     clf = model_type(**modelparams)
#     clf.fit(Xtrain,Ytrain,**fitparams)
#     score = clf.score(Xtest,Ytest)
#     print(score)
# 
# def plot_clf_boundary(samplesize=500):
#     import matplotlib.pyplot as plt
#     Xtrain = read_data('checkboard-train-data.npy')[:samplesize]
#     N = 200 # 10000
#     bd = 1000 # 100000
#     n_epoch = 5 
#     Gamma = 10.
#     classes = [0.,1.]
#     loss_fn = 'hinge'
#     params = {}
#     feature = 'Gaussian'
#     params['model'] = {
#         'feature':feature,
#         'n_old_features':len(Xtrain[0]),
#         'n_new_features':N,
#         'classes':classes,
#         'loss_fn':loss_fn
#     }
#     model_type = librf.RF
#     params['model']['Gamma'] = Gamma
#     clf = model_type(**params['model'])
#     Ypred,_,_ = clf.predict(Xtrain)
#     c = []
#     for idx in range(samplesize):
#         if Ypred[idx] == 0:
#             c.append('r')
#         else:
#             c.append('b')
#     fig = plt.figure()
#     plt.scatter(Xtrain[:,0],Xtrain[:,1],s=0.5,c=c)
#     plt.show()
#     feature = 'ReLU'
#     params['model']['feature'] = feature
#     clf = model_type(**params['model'])
#     Ypred,_,_ = clf.predict(Xtrain)
#     c = []
#     for idx in range(samplesize):
#         if Ypred[idx] == 0:
#             c.append('r')
#         else:
#             c.append('b')
#     fig = plt.figure()
#     plt.scatter(Xtrain[:,0],Xtrain[:,1],s=0.5,c=c)
#     plt.show()

if __name__ == '__main__':
    # params = read_params()
    # screen_params(params)
    train_and_test('sine1-10','ReLU')
