import numpy as np
import matplotlib.pyplot as plt
import rff

with open('data/train-labels.idx1-ubyte','rb') as file:
    data = file.read()
    print(data[0:20])
    for row in range(20):
        print(data[row])
