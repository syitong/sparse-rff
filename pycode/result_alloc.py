import numpy as np

dataset = 'adult'
filename = 'result/' + dataset + '_ReLU-'
row = ['log(rate)\log(Gamma)']
row.extend(np.arange(-6.,2,1))
row.append('ReLU')
output = [row]
log_rate = np.arange(-2.,4,0.5)
for idx,rate in enumerate(log_rate):
    row = []
    with open(filename + str(idx),'r') as f:
        result = eval(f.read())
    row.append(log_rate[idx])
    for item in result:
        row.append(item['score'])
    output.append(row)

with open('result/'+dataset+'-alloc','w') as f:
    f.write(str(output))
