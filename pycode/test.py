import numpy as np
import time
import tensorflow as tf

def normalize_tensor():
    with tf.Session() as sess:
        a = np.random.randn(3,2)
        a = tf.constant(a)
        na = tf.stack(tf.norm(a,ord=np.inf,axis=0),axis=0)
        a1 = tf.divide(a,na)
        print(sess.run(a))
        print(sess.run(na))
        print(sess.run(a1))

if __name__ == '__main__':
    normalize_tensor()
