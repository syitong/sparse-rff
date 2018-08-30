import numpy as np

def screen_params_alloc(dataset,feature,lograte,logGamma):
    filename = 'result/{0:s}-{1:s}-screen-'.format(dataset,feature)
    row = ['log(rate)\log(Gamma)']
    row.extend(logGamma)
    output = [row]
    for idx,rate in enumerate(lograte):
        row = []
        with open(filename + str(idx),'r') as f:
            result = eval(f.read())
        row.append(log_rate[idx])
        for item in result:
            row.append(item['score'])
        output.append(row)
    with open('result/'+dataset+'alloc','w') as f:
        f.write(str(output))

def train_and_test_alloc(dataset,feature,trials):
    filename = 'result/{0:s}-{1:s}-test-'.format(dataset,feature)
    tags = ['accuracy','sparsity','traintime','testtime']
    alloc = {}
    for idx in range(4):
        tag = tags[idx]
        result = np.zeros(trials)
        for prefix in range(trials):
            with open('result/'+dataset+'-test-'+str(prefix),'r') as fr:
                dict1 = eval(fr.read())
            result[prefix] = dict1[tag]
        mean = np.mean(result)
        std = np.std(result)
        alloc[tag] = {'mean':F_mean,'std':F_std}

    with open(filename+'alloc','w') as fw:
        fw.write(str(alloc))

if __name__ == '__main__':
    screen_params_alloc('sine1-10','ReLU')
