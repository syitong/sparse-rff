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
    data = []
    labels = []
    with open(RAW_DATA_PATH + 'adult.data.txt','r') as f:
        for line in f:
            newrow = []
            row = line[:-1].split(', ')
            if '?' in row or row == ['']:
                pass
            else:
                labels.append(label_list.index(row.pop(-1)))
                for idx,item in enumerate(row):
                    if var_list[idx] == ['continuous']:
                        newrow.extend([float(item)])
                    else:
                        one_hot = [0] * len(var_list[idx])
                        one_hot[var_list[idx].index(item)] = 1
                        newrow.extend(one_hot)
                data.append(newrow)
    print(len(data))
    print(len(labels))
    Xtrain,Ytrain,Xtest,Ytest = split_train_test(data,labels,0.1)
    print(len(Xtrain))
    print(len(Ytrain))
    print(len(Xtest))
    print(len(Ytest))
    with open(DATA_PATH+'adult-train-data.txt','w') as f:
        f.write(str(Xtrain))
    with open(DATA_PATH+'adult-train-label.txt','w') as f:
        f.write(str(Ytrain))
    with open(DATA_PATH+'adult-test-data.txt','w') as f:
        f.write(str(Xtest))
    with open(DATA_PATH+'adult-test-label.txt','w') as f:
        f.write(str(Ytest))

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
    with open(RAW_DATA_PATH + 'covtype.data','r') as f:
        for line in f:
            row = line[:-1].split(',')
            labels.append(int(row.pop(-1)))
            row = [float(item) for item in row]
            data.append(row)
    Xtrain,Ytrain,Xtest,Ytest = split_train_test(data,labels,0.1)
    print(len(Xtrain))
    print(len(Ytrain))
    print(len(Xtest))
    print(len(Ytest))
    with open(DATA_PATH+'covtype-train-data.txt','w') as f:
        f.write(str(Xtrain))
    with open(DATA_PATH+'covtype-train-label.txt','w') as f:
        f.write(str(Ytrain))
    with open(DATA_PATH+'covtype-test-data.txt','w') as f:
        f.write(str(Xtest))
    with open(DATA_PATH+'covtype-test-label.txt','w') as f:
        f.write(str(Ytest))

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

def read_data(filename):
    with open(DATA_PATH + filename,'r') as f:
        data = eval(f.read())
    return data

if __name__ == '__main__':
    # proc_adult()
    # proc_covtype()
    proc_kddcup()
    X = read_data('kddcup99-test-data.txt')
    print(len(X[0]))
