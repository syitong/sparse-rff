import numpy as np
from multiprocessing import Pool
from functools import partial

def foo(*x,**y):
    print(x)
    print(y)

def bar(x):
    foob = partial(foo,x)
    with Pool() as p:
        p.map(foob,[1,2,3])
if __name__ == '__main__':
    bar(1)
