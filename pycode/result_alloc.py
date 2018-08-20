import numpy as np

dataset = 'adult'
filename = 'result/' + dataset + '_ReLU-'
output = ['log(rate)\log(Gamma)']
output.extend(np.arange(-6.,2,1))
output.append('ReLU')
log_rate = np.arange(-2.,3,0.5)
for idx in range(10):
    row = []
    with open(filename + str(idx),'r') as f:
        result = eval(f.read())
    row.append(log_rate[idx])
    for item in result:
        row.append(item['score'])
    output.append(row)

with open('result/'+dataset+'-alloc','w') as f:
    f.write(str(output))
