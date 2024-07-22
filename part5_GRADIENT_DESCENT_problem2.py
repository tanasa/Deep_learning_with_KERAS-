# Commented out IPython magic to ensure Python compatibility.

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
import keras as keras
import math
from keras.models import Sequential
from keras.layers import Dense, Activation
from datascience import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# %matplotlib inline

def z(x,y) : 
    z = -1 * math.sqrt(25 - (x-2)**2 - (y-3)**2)
    return(z)

def dz_dx(x,y) : 
    res = (x-2) / math.sqrt(25 - (x-2)**2 - (y-3)**2)
    return(res)

def dz_dy(x,y) : 
    res = (y-3) / math.sqrt(25 - (x-2)**2 - (y-3)**2)
    return(res)

#print(z(1,1))
#print(dz_dx(1,1))
#print(dz_dy(1,1))

maxLimit = 5000
xHistory = np.tile(np.float32(0), maxLimit)
yHistory = np.tile(np.float32(0), maxLimit)

learning_rate = 0.01
xStart = 0.3
yStart = 0.3

for i in range(1,maxLimit):
      xHistory[i] = xStart
      yHistory[i] = yStart

      deltaW = dz_dx(xStart, yStart)
      deltaB = dz_dy(xStart, yStart) 

      xStart = xStart - learning_rate * deltaW 
      yStart = yStart - learning_rate * deltaB

fig = plt.figure()
ax = plt.axes()

ax.plot(xHistory)
ax.plot(yHistory)







def dz_dx(x1,y1) : 
    z1 = 2 * x1 - 2.0
    return z1

def dz_dy(x1, y1) : 
   z1 = 2 * y1 - 6.0 
   return z1 

xStart = 0.1 
yStart = 2.0
learningRate = 0.01
maxLimit = 500

xHistory = np.tile(np.float32(0), maxLimit)
yHistory = np.tile(np.float32(0), maxLimit)


for i in range(1,maxLimit):
    
    xHistory[i] = xStart
    yHistory[i] = yStart

    dW = dz_dx(xStart,yStart)
    db = dz_dy(xStart,yStart)
    
    xStart = xStart - learningRate * dW
    yStart = yStart - learningRate * db

fig = plt.figure()
ax = plt.axes()

ax.plot(xHistory)
ax.plot(yHistory)





#
