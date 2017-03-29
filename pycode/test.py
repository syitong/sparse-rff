import numpy as np
import matplotlib.pyplot as plt
import datagen, dataplot

#[X,Y] = datagen.unit_circle(0.3,0.3,50)
#dataplot.plot_circle(X,Y)

[X,Y] = datagen.unit_interval(0.3,0.7,50)
dataplot.plot_interval(X,Y)
