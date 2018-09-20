import numpy as np

RAW_DATA_PATH = 'data/rawdata/'
DATA_PATH = 'data/'

def proc_adult():
    var_list = []
    with open(RAW_DATA_PATH + 'adult.info','r') as f:
        for line in f:
            _,value = line[:-2].split(': ')
            var_list.append(value.split(', '))
        label_list = var_list.pop(0)
    maxmin_list = [None] * len(var_list)
    data = []
    labels = []
    with open(RAW_DATA_PATH + 'adult.data.txt','r') as f:
        for line in f:
            row = line[:-1].split(', ')
            if '?' in row or row == ['']:
                pass
            else:
                labels.append(label_list.index(row.pop(-1)))
                for idx,item in enumerate(row):
                    if var_list[idx] == ['continuous']:
                        temp_value = float(item)
                        if maxmin_list[idx] == None:
                            maxmin_list[idx] = {
                                'max': temp_value,
                                'min': temp_value
                            }
                        elif maxmin_list[idx]['max'] < temp_value:
                            maxmin_list[idx]['max'] = temp_value
                        elif maxmin_list[idx]['min'] > temp_value:
                            maxmin_list[idx]['min'] = temp_value
                        row[idx] = temp_value
                    else:
                        pass
                data.append(row)
    newdata = []
    for row in data:
        newrow = []
        for idx,item in enumerate(row):
            if var_list[idx] == ['continuous']:
                newrow.append((item - maxmin_list[idx]['min']) /
                    (maxmin_list[idx]['max'] - maxmin_list[idx]['min']))
            else:
                one_hot = [0] * len(var_list[idx])
                one_hot[var_list[idx].index(item)] = 1
                newrow.extend(one_hot)
        newdata.append(newrow)
    print(len(newdata),len(labels))
    np.save(DATA_PATH+'adult-train-data',np.array(newdata))
    np.save(DATA_PATH+'adult-train-label',np.array(labels))
    # with open(DATA_PATH+'adult-train-data.txt','w') as f:
    #     f.write(str(newdata))
    # with open(DATA_PATH+'adult-train-label.txt','w') as f:
    #     f.write(str(labels))
    newdata = []
    labels = []
    with open(RAW_DATA_PATH + 'adult.test.txt','r') as f:
        for line in f:
            row = line[:-2].split(', ')
            if '?' in row or row == ['']:
                pass
            else:
                labels.append(label_list.index(row.pop(-1)))
                newrow = []
                for idx,item in enumerate(row):
                    if var_list[idx] == ['continuous']:
                        newrow.append((float(item) - maxmin_list[idx]['min']) /
                            (maxmin_list[idx]['max'] - maxmin_list[idx]['min']))
                    else:
                        one_hot = [0] * len(var_list[idx])
                        one_hot[var_list[idx].index(item)] = 1
                        newrow.extend(one_hot)
                newdata.append(newrow)
    np.save(DATA_PATH+'adult-test-data',np.array(newdata))
    np.save(DATA_PATH+'adult-test-label',np.array(labels))
    # with open(DATA_PATH+'adult-test-data.txt','w') as f:
    #     f.write(str(newdata))
    # with open(DATA_PATH+'adult-test-label.txt','w') as f:
    #     f.write(str(labels))

def split_train_test(data,labels,test_fraction):
    size = int(len(data) * test_fraction)
    rand_list = np.random.permutation(len(data))
    Xtest = [data[i] for i in rand_list[:size]]
    Ytest = [labels[i] for i in rand_list[:size]]
    Xtrain = [data[i] for i in rand_list[size:]]
    Ytrain = [labels[i] for i in rand_list[size:]]
    return Xtrain,Ytrain,Xtest,Ytest

def proc_covtype():
    data = []
    labels = []
    maxmin_list = []
    with open(RAW_DATA_PATH + 'covtype.data','r') as f:
        for line in f:
            row = line[:-1].split(',')
            labels.append(int(row.pop(-1)))
            for idx,item in enumerate(row):
                temp_value = float(item)
                if len(maxmin_list) <= idx:
                    maxmin_list.append({
                        'max': temp_value,
                        'min': temp_value
                    })
                elif maxmin_list[idx]['max'] < temp_value:
                    maxmin_list[idx]['max'] = temp_value
                elif maxmin_list[idx]['min'] > temp_value:
                    maxmin_list[idx]['min'] = temp_value
                row[idx] = temp_value
            data.append(row)
    print(maxmin_list)
    newdata = []
    for row in data:
        newrow = []
        for idx,item in enumerate(row):
            newrow.append((item - maxmin_list[idx]['min']) /
                (maxmin_list[idx]['max'] - maxmin_list[idx]['min']))
        newdata.append(newrow)
    Xtrain,Ytrain,Xtest,Ytest = split_train_test(newdata,labels,0.1)
    print(len(Xtrain))
    print(len(Ytrain))
    print(len(Xtest))
    print(len(Ytest))
    np.save(DATA_PATH+'covtype-train-data',np.array(Xtrain))
    np.save(DATA_PATH+'covtype-train-label',np.array(Ytrain))
    np.save(DATA_PATH+'covtype-test-data',np.array(Xtest))
    np.save(DATA_PATH+'covtype-test-label',np.array(Ytest))

def binary_covtype():
    Ytr = np.load(DATA_PATH+'covtype-train-label.npy')
    Ytil = (Ytr == 2)
    np.save(DATA_PATH+'covtype-train-binary-label',Ytil)
    Ytr = np.load(DATA_PATH+'covtype-test-label.npy')
    Ytil = (Ytr == 2)
    np.save(DATA_PATH+'covtype-test-binary-label',Ytil)

def proc_cifar():
    # generate grayscale images from cifar dataset
    from experiments import _read_cifar
    Xtr1 = []
    Xts1 = []
    Xtr,Ytr,Xts,Yts = _read_cifar(5)
    graycoef = np.array([0.299, 0.587, 0.114])
    for row in Xtr:
        img = graycoef.dot(row.reshape((3,-1)))
        Xtr1.append(img)
    for row in Xts:
        img = graycoef.dot(row.reshape((3,-1)))
        Xts1.append(img)
    np.save(DATA_PATH+'cifar-train-data',np.array(Xtr1))
    np.save(DATA_PATH+'cifar-test-data',np.array(Xts1))
    np.save(DATA_PATH+'cifar-train-label',np.array(Ytr))
    np.save(DATA_PATH+'cifar-test-label',np.array(Yts))

# def proc_kddcup():
#     data = []
#     labels = []
#     var_list = []
#     with open(RAW_DATA_PATH + 'kddcup99-10percent.info','r') as f:
#         for line in f:
#             _,value = line[:-2].split(': ')
#             var_list.append(value.split(','))
#         label_list = var_list.pop(0)
#     print(var_list)
#     print(len(var_list))
#     with open(RAW_DATA_PATH + 'kddcup99-10percent','r') as f:
#         for line in f:
#             newrow = []
#             row = line[:-2].split(',')
#             print(len(row))
#             labels.append(label_list.index(row.pop(-1)))
#             for idx,item in enumerate(row):
#                 if var_list[idx] == ['continuous']:
#                     newrow.append(float(item))
#                 else:
#                     newrow.append(item)
#             data.append(newrow)
#     for idx,var in enumerate(var_list):
#         if var == 'symbolic':
#             var_list[idx] = list(set([item[idx] for item in data]))
#         print(var_list)
#     newdata = []
#     for row in data:
#         newrow = []
#         for idx,var in enumerate(var_list):
#             if var != ['continuous']:
#                 one_hot = [0] * len(var_list[idx])
#                 one_hot[var_list[idx].index(item)] = 1
#                 newrow.extend(one_hot)
#             else:
#                 newrow.append(row[idx])
#         newdata.append(newrow)
#     Xtrain,Ytrain,Xtest,Ytest = split_train_test(newdata,labels,0.1)
#     print(len(Xtrain))
#     print(len(Ytrain))
#     print(len(Xtest))
#     print(len(Ytest))
#     with open(DATA_PATH+'kddcup99-train-data.txt','w') as f:
#         f.write(str(Xtrain))
#     with open(DATA_PATH+'kddcup99-train-label.txt','w') as f:
#         f.write(str(Ytrain))
#     with open(DATA_PATH+'kddcup99-test-data.txt','w') as f:
#         f.write(str(Xtest))
#     with open(DATA_PATH+'kddcup99-test-label.txt','w') as f:
#         f.write(str(Ytest))

if __name__ == '__main__':
    # proc_adult()
    # proc_covtype()
    # binary_covtype()
    proc_cifar()
