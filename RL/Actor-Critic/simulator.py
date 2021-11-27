import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras






def data_builder(interval, length):
    distance = 0
    px = 0
    py = 0
    trace_x = []
    trace_y = []

    counter = -interval
    x = []
    y = []
    theta = 0
    std = 0
    for i in range(length):
        if i - counter == interval:
            counter = i
            theta = np.random.rand()*3.1415926
            distance = np.random.rand()*10
            
            px += 2*(np.random.rand()-0.5)
            trace_x.append(px)
            py += 2*(np.random.rand()-0.5)
            trace_y.append(py)
            theta = np.math.atan(py/px)
            distance = np.math.sqrt(px*px+py*py)
        
        std = (10-distance)*0.02
        x.append(theta)
        err = np.random.randn()*std
        y.append(theta+err)


        