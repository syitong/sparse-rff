import numpy as np

def screen_params_alloc(dataset):
    filename = 'result/' + dataset + '-'
    row = ['log(rate)\log(Gamma)']
    row.extend(np.arange(-2.,4,0.5))
    row.append('ReLU')
    output = [row]
    log_rate = np.arange(-3.,3,0.5)
    for idx,rate in enumerate(log_rate):
        row = []
        with open(filename + str(idx),'r') as f:
            result = eval(f.read())
        row.append(log_rate[idx])
        for item in result:
            row.append(item['score'])
        output.append(row)

    with open('result/'+dataset+'-adam-alloc','w') as f:
        f.write(str(output))

def train_and_test_alloc(dataset):
    tags = ['accuracy','sparsity','traintime','testtime']
    alloc = {'Gaussian':{},
        'ReLU':{}}
    for idx in range(4):
        tag = tags[idx]
        F_result = np.zeros(10)
        R_result = np.zeros(10)
        for prefix in range(10):
            with open('result/'+dataset+'-test-'+str(prefix),'r') as fr:
                dict1,dict2 = eval(fr.read())
            F_result[prefix] = dict1[tag]
            R_result[prefix] = dict2[tag]
        F_mean = np.mean(F_result)
        R_mean = np.mean(R_result)
        F_std = np.std(F_result)
        R_std = np.std(R_result)
        alloc['Gaussian'][tag] = {'mean':F_mean,'std':F_std}
        alloc['ReLU'][tag] = {'mean':R_mean,'std':R_std}

    with open('result/'+dataset+'_test-alloc','w') as fw:
        fw.write(str(alloc))

if __name__ == '__main__':
    screen_params_alloc('covtype')
