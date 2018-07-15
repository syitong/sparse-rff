import numpy as np
import time
import tensorflow as tf

def time_sparsity():
    a = np.random.randn(100)
    b = a.copy()
    b[50:100] = 0
    A = np.random.randn(1000).reshape((10,100))
    with tf.Session() as sess:
        a1 = tf.constant(a,shape=[100,1])
        b1 = tf.constant(b,shape=[100,1])
        A1 = tf.constant(A)
        c = tf.matmul(A1,a1)
        d = tf.matmul(A1,b1)
        t1 = time.process_time()
        for i in range(1000):
            sess.run(c)
        t2 = time.process_time()
        print(t2-t1)
        t1 = time.process_time()
        for i in range(1000):
            sess.run(d)
        t2 = time.process_time()
        print(t2-t1)

if __name__ == '__main__':
    time_sparsity()
