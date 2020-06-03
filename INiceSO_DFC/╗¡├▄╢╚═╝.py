#  _*_ coding: UTF-8 _*_
import random
import scipy as sp
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.special import gamma
from scipy.integrate import quad,dblquad,nquad
import math
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.special import polygamma
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from mpl_toolkits.mplot3d import Axes3D
import operator
from functools import reduce
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import scipy.stats as stats
import itertools
import time
import csv
import xlwt
import seaborn as sns
import time

import rpy2
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpacks
from rpy2.robjects.vectors import FloatVector
from sklearn import cluster
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
mixtools = importr('mixtools')
fpc = importr('cluster')

def GenerateManualData():
    '''2-- 2Ddata'''
    size = 50
    mean1 = [30, 40]
    cov1 = [[1, 0], [0,2]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 40).T
    # plt.plot(x1, y1, 'x')

    mean2 = [40, 50]
    cov2 = [[2, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 35).T
    # plt.plot(x2, y2, 'x')

    mean3 = [33, 38]
    cov3 = [[2, 0], [0, 1]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 25).T
    # plt.plot(x3, y3, 'x')

    mean4 = [5, 70]
    cov4 = [[3, 0], [0, 3]]
    x4, y4 = np.random.multivariate_normal(mean4, cov4, size).T
    # plt.plot(x4, y4, 'x')

    mean5 = [5, 12]
    cov5 = [[3, 0], [0, 3]]
    x5, y5 = np.random.multivariate_normal(mean5, cov5, size).T
    # plt.plot(x4, y4, 'x')
    # plt.show()

    mean6 = [50, 50]
    cov6 = [[2, 0], [0, 2]]
    x6, y6 = np.random.multivariate_normal(mean6, cov6, size).T

    mean7 = [60, 23]
    cov7 = [[1, 0], [0, 1]]
    x7, y7 = np.random.multivariate_normal(mean7, cov7, size).T

    mean8 = [74, 36]
    cov8 = [[1, 0], [0, 1]]
    x8, y8 = np.random.multivariate_normal(mean8, cov8, size).T

    mean9 = [62, 48]
    cov9 = [[3, 0], [0, 3]]
    x9, y9 = np.random.multivariate_normal(mean9, cov9, size).T

    mean10 = [88, 20]
    cov10 = [[3, 0], [0, 3]]
    x10, y10 = np.random.multivariate_normal(mean10, cov10, size).T

    mean11 = [40, 90]
    cov11 = [[3, 0], [0, 3]]
    x11, y11 = np.random.multivariate_normal(mean11, cov11, size).T

    mean12 = [38, 76]
    cov12 = [[3, 0], [0, 3]]
    x12, y12 = np.random.multivariate_normal(mean12, cov12, size).T

    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data6 = []
    data7 = []
    data8 = []
    data9 = []
    data10 = []
    data11 = []
    data12 = []
    data1.append(x1)
    data1.append(y1)
    data2.append(x2)
    data2.append(y2)
    data3.append(x3)
    data3.append(y3)
    data4.append(x4)
    data4.append(y4)
    data5.append(x5)
    data5.append(y5)
    data6.append(x6)
    data6.append(y6)
    data7.append(x7)
    data7.append(y7)
    data8.append(x8)
    data8.append(y8)
    data9.append(x9)
    data9.append(y9)
    data10.append(x10)
    data10.append(y10)
    data11.append(x11)
    data11.append(y11)
    data12.append(x12)
    data12.append(y12)

    data1 = np.array(data1)
    data1 = DataFrame(data1)
    data1 = data1.T
    data2 = np.array(data2)
    data2 = DataFrame(data2)
    data2 = data2.T
    data3 = np.array(data3)
    data3 = DataFrame(data3)
    data3 = data3.T
    data4 = np.array(data4)
    data4 = DataFrame(data4)
    data4 = data4.T
    data5 = np.array(data5)
    data5 = DataFrame(data5)
    data5 = data5.T
    data6 = np.array(data6)
    data6 = DataFrame(data6)
    data6 = data6.T
    data7 = np.array(data7)
    data7 = DataFrame(data7)
    data7 = data7.T
    data8 = np.array(data8)
    data8 = DataFrame(data8)
    data8 = data8.T
    data9 = np.array(data9)
    data9 = DataFrame(data9)
    data9 = data9.T
    data10 = np.array(data10)
    data10 = DataFrame(data10)
    data10 = data10.T
    data11 = np.array(data11)
    data11 = DataFrame(data11)
    data11 = data11.T
    data12= np.array(data12)
    data12 = DataFrame(data12)
    data12 = data12.T

    '''将各个不同分布的数据合并在一起'''
    #, data7, data5, data3, data4,data2, data6, data9, data10, data11, data12
    # data = data1.append([data2])  # , data8 , data6, data7, data8, data9, data10, data11, data12
    data = data1
    data = data.reset_index(drop=True)
    data = (data / 100)  # 将原始数据标准化
    data0 = data2
    data0 = data0.reset_index(drop=True)
    data0 = (data0 / 100)  # 将原始数据标准化

    '''如果数据是二维的，就画图展示数据'''
    if data.shape[1] == 2:
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], '+')
        plt.plot(data0.iloc[:, 0], data0.iloc[:, 1], '*')
        plt.xlabel("1st Dimension")
        plt.ylabel("2nd Dimension")
        # plt.title("normalized data")
        plt.show()

    return data

GenerateManualData()