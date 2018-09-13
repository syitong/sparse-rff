import numpy as np
from numpy import array
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def _extract_xy(dataset,feature):
    filename = 'result/{0:s}-{1:s}-screen-'.format(dataset,feature)
    with open(filename+'alloc','r') as f:
        result,params = eval(f.read())
    result_trim = [row[1:] for row in result[1:]]
    result_trim = np.array(result_trim)
    y = np.max(result_trim[:,:-1],axis=0)
    x = result[0][1:-1]
    return x,y

def plot_params(dataset):
    x,y1 = _extract_xy(dataset,'ReLU')
    _,y2 = _extract_xy(dataset,'Gaussian')
    fig = plt.figure()
    plt.title("Random Features Methods on "+dataset)
    plt.xlabel('log(Gamma)')
    plt.ylabel('accuracy')
    # plt.xticks(x)
    plt.ylim((0,1.01))
    plt.plot(x,y1,'x--',label='ReLU')
    plt.plot(x,y2,'o:',label='Fourier')
    plt.legend(loc=2)
    plt.savefig('image/{}-gamma.eps'.format(dataset))
    plt.close(fig)

def print_params(dataset,feature):
    filename = 'result/{0:s}-{1:s}-screen-'.format(dataset,feature)
    with open(filename+'alloc','r') as f:
        result, params = eval(f.read())
    for row in result:
        if type(row[0]) == str:
            print('{:^20}'.format(row[0]),end='')
        else:
            print('{:^ 20}'.format(row[0]),end='')
        for item in row[1:]:
            if type(item) == str:
                print('{:>7}'.format(item),end='')
            else:
                print('{:>7.2f}'.format(item),end='')
        print('')
    F_result = [row[1:] for row in result[1:]]
    F_result = np.array(F_result)
    x,y = np.unravel_index(np.argmax(F_result),
        F_result.shape)
    logGamma = result[0][y+1]
    lograte = result[x+1][0]
    print('best log(Gamma): ',logGamma)
    print('best log(rate): ',lograte)
    _dict_print(params)
    return logGamma,lograte,params

def _dict_print(dictx,loc=0):
    for key,value in dictx.items():
        print(' '*loc,end='')
        print(key,end='')
        if type(value) == dict:
            print('')
            _dict_print(value,loc+5)
        else:
            print(': {}'.format(value))
    return 1

def print_test_results(dataset,feature):
    filename = 'result/{0:s}-{1:s}-test-'.format(dataset,feature)
    with open(filename+'alloc','r') as f:
        result,_,model_params,fit_params = eval(f.read())
    print(dataset)
    _dict_print(result)
    _dict_print(model_params)
    _dict_print(fit_params)

if __name__ == '__main__':
    print_params('fmnist','Gaussian')
    print_params('fmnist','ReLU')
