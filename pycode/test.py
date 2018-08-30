import numpy as np
from multiprocessing import Pool
from functools import partial
from sys import argv

def bar():
    prefix = argv[1]
    print('No sys arg is fine!')

if __name__ == '__main__':
    bar()
