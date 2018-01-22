# copyright@Mark Guo
# 2018/1/10
# Pre-processor

import scipy.io as sio
import numpy as np
import os
import math
import pandas as pd
from scipy.interpolate import interp1d, splev, splrep
# np.set_printoptions(threshold=np.inf)

'''STEP1. Interpolation'''

n_windmachines = 6
pred = np.empty([1920, n_windmachines]) # 96 * 20

x_all = [np.empty([0]) for i in range(6)]
y_all = [np.empty([0]) for i in range(6)]

available_day_per_month = [30, 31, 30, 30]

cnt = 0

for month in range(4,8):
    for date in range(1, available_day_per_month[month-4]+1):
        print(date)
        if os.path.exists("data/HisRawData/2016%02d/2016%02d%02d.mat" %(month, month, date)):
            today_speed = sio.loadmat("data/HisRawData/2016%02d/2016%02d%02d.mat" %(month, month, date))

            ''' Extract forecast for today from yesterday's data '''
            data = today_speed['FJDATA'][0]['speed']

            ''' Interpolate for each machine '''
            for k in range(n_windmachines):
                x = data[k][:, 0]
                x = x + 86400 * cnt
                y = data[k][:, 1]
                x_all[k] = np.append(x_all[k], x, axis=0)
                y_all[k] = np.append(y_all[k], y, axis=0)

        cnt = cnt + 1


'''reason why cubic spline is bad'''
# print(x_all[5], y_all[5])
# np.savetxt("x.txt", X = x_all[4])
# for i in range(n_windmachines):
#     print(np.argmax(np.diff(x_all[i])))
#     print(np.max(np.diff(x_all[i])))
# print(y_all[0].size)

x_pred = []
y_pred = []

'''cubic spline(bad effect)'''
# for k in range(n_windmachines):
#     x = x_all[k]
#     y = y_all[k]
#     tck = splrep(x, y, s=0)
#     xnew = np.linspace(0, 86400*121, num=96*121+1, endpoint=True)
#     print(xnew)
#     y_pred.append(splev(xnew, tck, der=0)[:-1])
#     print((y_pred[k]))

for k in range(n_windmachines):
    x = x_all[k]
    y = y_all[k]
    f = interp1d(x, y, kind='slinear', fill_value="extrapolate")  # Modify this line for different interp1d #
    xnew = np.linspace(0, 86400*121, num=96*121+1, endpoint=True)
    print(xnew)
    y_pred.append(f(xnew)[:-1])
    # print(np.max(y_pred[k]))

'''STEP 2. Transform into LSTM format'''

seq_len_ = 60
x_dim_ = n_windmachines
y_dim_ = n_windmachines
all_size_ = 96 * 121 - seq_len_
train_size_ = 10000
test_size_ = all_size_ - train_size_

X = np.zeros((all_size_, seq_len_, x_dim_))
Y = np.zeros((all_size_, y_dim_))

for i in range(seq_len_, 96 * 121):
    time = i * 900
    for j in range(seq_len_):
        for k in range(n_windmachines):
            X[i - seq_len_][j][k] = y_all[k][i - j - 1]
    for k in range(n_windmachines):
        Y[i - seq_len_][k] = y_all[k][i]

train_x = X[0:10000]
test_x = X[10000:96*121]
train_y = Y[0:10000]
test_y = Y[10000:96*121]

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

train_x.tofile('train_x.dat')
train_y.tofile('train_y.dat')
test_x.tofile('test_x.dat')
test_y.tofile('test_y.dat')

'''STEP 3. Plot pressure, temperature & velocity'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
P = np.loadtxt("data/PredictData/201604/01/P_out_ncl.txt")
T = np.loadtxt("data/PredictData/201604/01/T_out_ncl.txt")
ax=plt.subplot(111,projection='3d')

ax.scatter(y_all[0][8:96],P[:88, 0],T[:88, 0])

plt.show()


