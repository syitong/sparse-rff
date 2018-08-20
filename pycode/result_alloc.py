import numpy as np

dataset = 'covtype'
filename = 'result/' + dataset + '-'
row = ['log(rate)\log(Gamma)']
row.extend(np.arange(-2.,4,0.5))
row.append('ReLU')
output = [row]
log_rate = np.arange(0.8,2.8,0.2)
for idx in range(10):
    row = []
    with open(filename + str(idx),'r') as f:
        result = eval(f.read())
    row.append(log_rate[idx])
    for item in result:
        row.append(item['score'])
    output.append(row)

with open('result/'+dataset+'-refine-alloc','w') as f:
    f.write(str(output))
