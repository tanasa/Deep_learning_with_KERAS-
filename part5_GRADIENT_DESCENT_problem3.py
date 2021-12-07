#!/usr/bin/env python
# coding: utf-8

##############################################################
## here it is code in SCIKIT LEARN
##############################################################

import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model 
 
n_samples = 30 
train_x = np.linspace(0,20,n_samples) 
train_y = 3.7 * train_x + 14 + 4 * np.random.randn(n_samples) 
 
plt.plot(train_x, train_y,'o') 

x = train_x.reshape(-1, 1) 
y = train_y.reshape(-1, 1)

linreg = linear_model.LinearRegression()
linreg.fit(x, y) 
print("SLOPE : ")
print(linreg.coef_)
print("INTERCEPT : ")
print(linreg.intercept_)


## conda install -c r r-irkernel
 ##############################################################
## here it is code by using SGD
##############################################################


def dRSS_dm(SLOPE, INTERCEPT):
    return(-2 * sum((y - SLOPE * x - INTERCEPT) * x))

def dRSS_db(SLOPE, INTERCEPT):
    return(-2 * sum((y - SLOPE * x - INTERCEPT)))

SLOPE_Start = 0
INTERCEPT_Start = 0
learning_rate = 0.0001

LIMIT = 6000

SLOPE_History = np.tile(np.float32(0), LIMIT)
INTERCEPT_History = np.tile(np.float32(0), LIMIT)

for i in range(LIMIT):
    
    SLOPE_History[i] = SLOPE_Start
    INTERCEPT_History[i] = INTERCEPT_Start
    
    dW = dRSS_dm(SLOPE_Start, INTERCEPT_Start)
    db = dRSS_db(SLOPE_Start, INTERCEPT_Start)

    SLOPE_Start = SLOPE_Start - learning_rate * dW
    INTERCEPT_Start = INTERCEPT_Start - learning_rate * db
    

print("SLOPE :", SLOPE_History[LIMIT-1])
print("INTERCEPT :", INTERCEPT_History[LIMIT-1])

fig = plt.figure()
ax = plt.axes()

ax.plot(SLOPE_History)
ax.plot(INTERCEPT_History)

