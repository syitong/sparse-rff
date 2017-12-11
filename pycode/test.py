import numpy as np
import matplotlib.pyplot as plt
import rff
import tensorflow as tf
import tfRF2L

n_old_features = 10
n_components = 20
Lambda = 1
Gamma = 1
n_classes = 5
clf = tfRF2L.tfRF2L(n_old_features,n_components,
    Lambda,Gamma,n_classes)
