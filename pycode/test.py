import numpy as np
from multiprocessing import Pool
from functools import partial

class foo:
    def __init__(self,x):
        self.x = x
    def fooprint(self):
        print(self.x)

def bar(myclass,myparams):
    y = myclass(myparams)
    return y
if __name__ == '__main__':
    y = bar(foo,2)
    y.fooprint()
