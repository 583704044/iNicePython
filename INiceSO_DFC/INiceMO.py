"""I_niceSO算法，主要步骤如下：
第一步：先估计一个M值
第二步：生成若干个观测点并且计算各个观测点与数据之间的距离
第三步：利用gammamixEM求出最优的GMM模型，GMM模型里面包含K值以及概率矩阵P
第四步：利用概率矩阵P对数据进行分类，分为K个类
第五步：利用KNN算法找出每个类的中心点
"""
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
import time
import seaborn as sns

'''生成实验数据'''
'''3D数据'''
def GenerateManualData1():
    mean1 = [5, 5, 5]
    cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x1, y1, z1 = np.random.multivariate_normal(mean1, cov1, 1000).T
    # print([x1,y1,z1])

    mean2 = [15, 15, 5]
    cov2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x2, y2, z2 = np.random.multivariate_normal(mean2, cov2, 1000).T
    # print([x1,y1,z1])

    mean3 = [5, 5, 15]
    cov3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x3, y3, z3 = np.random.multivariate_normal(mean3, cov3, 1000).T
    # print([x1,y1,z1])

    mean4 = [15, 15, 15]
    cov4 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x4, y4, z4 = np.random.multivariate_normal(mean4, cov4, 1000).T
    # print([x1,y1,z1])

    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x1, y1, z1, c='y')  # 绘制数据点
    ax.scatter(x2, y2, z2, c='r')
    ax.scatter(x3, y3, z3, c='g')
    ax.scatter(x4, y4, z4, c='b')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

    data1 = []
    data2 = []
    data3 = []
    data4 = []
    data5 = []
    data1.append(x1)
    data1.append(y1)
    data1.append(z1)

    data2.append(x2)
    data2.append(y2)
    data2.append(z2)

    data3.append(x3)
    data3.append(y3)
    data3.append(z3)

    data4.append(x4)
    data4.append(y4)
    data4.append(z4)

    # data5.append(x5)
    # data5.append(y5)

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

    '''将各个不同分布的数据合并在一起'''
    data = data1.append([data2, data3, data4])  # data5
    data = data.reset_index(drop=True)
    data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化

    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c='y')  # 绘制数据点
    plt.show()
    return data

'''12个簇'''
# def GenerateManualData():
#     '''2-- 2Ddata'''
#     size = 100
#     mean1 = [30, 30]
#     cov1 = [[3, 0], [0, 4]]
#     x1, y1 = np.random.multivariate_normal(mean1, cov1, size).T
#     # plt.plot(x1, y1, 'x')
#
#     mean2 = [70, 20]
#     cov2 = [[2, 0], [0, 1]]
#     x2, y2 = np.random.multivariate_normal(mean2, cov2, size).T
#     # plt.plot(x2, y2, 'x')
#
#     mean3 = [0, 0]
#     cov3 = [[4, 0], [0, 3]]
#     x3, y3 = np.random.multivariate_normal(mean3, cov3, size).T
#     # plt.plot(x3, y3, 'x')
#
#     mean4 = [5, 70]
#     cov4 = [[3, 0], [0, 3]]
#     x4, y4 = np.random.multivariate_normal(mean4, cov4, size).T
#     # plt.plot(x4, y4, 'x')
#
#     mean5 = [12, 12]
#     cov5 = [[3, 0], [0, 3]]
#     x5, y5 = np.random.multivariate_normal(mean5, cov5, size).T
#     # plt.plot(x4, y4, 'x')
#     # plt.show()
#
#     mean6 = [50, 50]
#     cov6 = [[2, 0], [0, 2]]
#     x6, y6 = np.random.multivariate_normal(mean6, cov6, size).T
#
#     mean7 = [60, 23]
#     cov7 = [[4, 0], [0, 3]]
#     x7, y7 = np.random.multivariate_normal(mean7, cov7, size).T
#
#     mean8 = [74, 36]
#     cov8 = [[3, 0], [0, 3]]
#     x8, y8 = np.random.multivariate_normal(mean8, cov8, size).T
#
#     mean9 = [62, 48]
#     cov9 = [[3, 0], [0, 3]]
#     x9, y9 = np.random.multivariate_normal(mean9, cov9, size).T
#
#     mean10 = [88, 20]
#     cov10 = [[3, 0], [0, 3]]
#     x10, y10 = np.random.multivariate_normal(mean10, cov10, size).T
#
#     mean11 = [40, 90]
#     cov11 = [[3, 0], [0, 3]]
#     x11, y11 = np.random.multivariate_normal(mean11, cov11, size).T
#
#     mean12 = [38, 76]
#     cov12 = [[3, 0], [0, 3]]
#     x12, y12 = np.random.multivariate_normal(mean12, cov12, size).T
#
#     data1 = []
#     data2 = []
#     data3 = []
#     data4 = []
#     data5 = []
#     data6 = []
#     data7 = []
#     data8 = []
#     data9 = []
#     data10 = []
#     data11 = []
#     data12 = []
#     data1.append(x1)
#     data1.append(y1)
#     data2.append(x2)
#     data2.append(y2)
#     data3.append(x3)
#     data3.append(y3)
#     data4.append(x4)
#     data4.append(y4)
#     data5.append(x5)
#     data5.append(y5)
#     data6.append(x6)
#     data6.append(y6)
#     data7.append(x7)
#     data7.append(y7)
#     data8.append(x8)
#     data8.append(y8)
#     data9.append(x9)
#     data9.append(y9)
#     data10.append(x10)
#     data10.append(y10)
#     data11.append(x11)
#     data11.append(y11)
#     data12.append(x12)
#     data12.append(y12)
#
#     data1 = np.array(data1)
#     data1 = DataFrame(data1)
#     data1 = data1.T
#     data2 = np.array(data2)
#     data2 = DataFrame(data2)
#     data2 = data2.T
#     data3 = np.array(data3)
#     data3 = DataFrame(data3)
#     data3 = data3.T
#     data4 = np.array(data4)
#     data4 = DataFrame(data4)
#     data4 = data4.T
#     data5 = np.array(data5)
#     data5 = DataFrame(data5)
#     data5 = data5.T
#     data6 = np.array(data6)
#     data6 = DataFrame(data6)
#     data6 = data6.T
#     data7 = np.array(data7)
#     data7 = DataFrame(data7)
#     data7 = data7.T
#     data8 = np.array(data8)
#     data8 = DataFrame(data8)
#     data8 = data8.T
#     data9 = np.array(data9)
#     data9 = DataFrame(data9)
#     data9 = data9.T
#     data10 = np.array(data10)
#     data10 = DataFrame(data10)
#     data10 = data10.T
#     data11 = np.array(data11)
#     data11 = DataFrame(data11)
#     data11 = data11.T
#     data12= np.array(data12)
#     data12 = DataFrame(data12)
#     data12 = data12.T
#
#     '''将各个不同分布的数据合并在一起'''
#     #, , data10,data5,data2, data6,,data12, data2, data6, data11,data9, data7, data8
#     data = data1.append([data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12])  #  , data6, data7, data8, data9, data10, data11, data12
#     data = data.reset_index(drop=True)
#     data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化
#
#     '''如果数据是二维的，就画图展示数据'''
#     if data.shape[1] == 2:
#         plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
#         plt.xlabel("x_axis")
#         plt.ylabel("y_axis")
#         # plt.title("normalized data")
#         plt.show()
#
#     return data

'''25个簇'''
def GenerateManualData():
    '''25个-- 2Ddata '''
    size = 100
    mean1 = [24, 27] # 30, 30
    cov1 = [[3, 0], [0, 4]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, size).T
    # plt.plot(x1, y1, 'x')

    mean2 = [93, 64] # 70, 20
    cov2 = [[2, 0], [0, 1]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, size).T
    # plt.plot(x2, y2, 'x')

    mean3 = [58, 23] # 0, 0
    cov3 = [[4, 0], [0, 3]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, size).T
    # plt.plot(x3, y3, 'x')

    mean4 = [90, 41] # 5, 70
    cov4 = [[3, 0], [0, 3]]
    x4, y4 = np.random.multivariate_normal(mean4, cov4, size).T
    # plt.plot(x4, y4, 'x')

    mean5 = [5, 2] # 12, 12
    cov5 = [[3, 0], [0, 3]]
    x5, y5 = np.random.multivariate_normal(mean5, cov5, size).T
    # plt.plot(x4, y4, 'x')
    # plt.show()

    mean6 = [10, 70] # 50, 50
    cov6 = [[2, 0], [0, 2]]
    x6, y6 = np.random.multivariate_normal(mean6, cov6, size).T

    mean7 = [40, 50] # 60, 23
    cov7 = [[4, 0], [0, 3]]
    x7, y7 = np.random.multivariate_normal(mean7, cov7, size).T

    mean8 = [81, 90] # 74, 36
    cov8 = [[3, 0], [0, 3]]
    x8, y8 = np.random.multivariate_normal(mean8, cov8, size).T

    mean9 = [1, 88] # 62, 48
    cov9 = [[3, 0], [0, 3]]
    x9, y9 = np.random.multivariate_normal(mean9, cov9, size).T

    mean10 = [79, 4] #88, 20
    cov10 = [[3, 0], [0, 3]]
    x10, y10 = np.random.multivariate_normal(mean10, cov10, size).T

    mean11 = [67, 38] # 40, 90
    cov11 = [[3, 0], [0, 3]]
    x11, y11 = np.random.multivariate_normal(mean11, cov11, size).T

    mean12 = [37, 96] # 38, 76
    cov12 = [[3, 0], [0, 3]]
    x12, y12 = np.random.multivariate_normal(mean12, cov12, size).T

    mean13 = [97, 21]
    cov13 = [[3, 0], [0, 4]]
    x13, y13 = np.random.multivariate_normal(mean13, cov13, size).T

    mean14 = [30, 7]
    cov14 = [[3, 0], [0, 4]]
    x14, y14 = np.random.multivariate_normal(mean14, cov14, size).T

    mean15 = [60, 80]
    cov15 = [[3, 0], [0, 4]]
    x15, y15 = np.random.multivariate_normal(mean15, cov15, size).T

    mean16 = [70, 60]
    cov16 = [[3, 0], [0, 4]]
    x16, y16 = np.random.multivariate_normal(mean16, cov16, size).T

    mean17 = [7, 46]
    cov17 = [[3, 0], [0, 4]]
    x17, y17 = np.random.multivariate_normal(mean17, cov17, size).T

    mean18 = [45, 70]
    cov18 = [[3, 0], [0, 4]]
    x18, y18 = np.random.multivariate_normal(mean18, cov18, size).T

    mean19 = [20, 86]
    cov19 = [[3, 0], [0, 4]]
    x19, y19 = np.random.multivariate_normal(mean19, cov19, size).T

    mean20 = [26, 65]
    cov20 = [[3, 0], [0, 4]]
    x20, y20 = np.random.multivariate_normal(mean20, cov20, size).T

    mean21 = [50, 2]
    cov21 = [[3, 0], [0, 4]]
    x21, y21 = np.random.multivariate_normal(mean21, cov21, size).T

    mean22 = [45, 36]
    cov22 = [[3, 0], [0, 4]]
    x22, y22 = np.random.multivariate_normal(mean22, cov22, size).T

    mean23 = [2, 35]
    cov23 = [[3, 0], [0, 4]]
    x23, y23 = np.random.multivariate_normal(mean23, cov23, size).T

    mean24 = [12, 18]
    cov24 = [[3, 0], [0, 4]]
    x24, y24 = np.random.multivariate_normal(mean24, cov24, size).T

    mean25 = [76, 26]
    cov25 = [[3, 0], [0, 4]]
    x25, y25 = np.random.multivariate_normal(mean25, cov25, size).T




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
    data13 = []
    data14 = []
    data15 = []
    data16 = []
    data17 = []
    data18 = []
    data19 = []
    data20 = []
    data21 = []
    data22 = []
    data23 = []
    data24 = []
    data25 = []
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
    data13.append(x13)
    data13.append(y13)
    data14.append(x14)
    data14.append(y14)
    data15.append(x15)
    data15.append(y15)
    data16.append(x16)
    data16.append(y16)
    data17.append(x17)
    data17.append(y17)
    data18.append(x18)
    data18.append(y18)
    data19.append(x19)
    data19.append(y19)
    data20.append(x20)
    data20.append(y20)
    data21.append(x21)
    data21.append(y21)
    data22.append(x22)
    data22.append(y22)
    data23.append(x23)
    data23.append(y23)
    data24.append(x24)
    data24.append(y24)
    data25.append(x25)
    data25.append(y25)

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
    data13 = np.array(data13)
    data13 = DataFrame(data13)
    data13 = data13.T
    data14 = np.array(data14)
    data14 = DataFrame(data14)
    data14 = data14.T
    data15 = np.array(data15)
    data15 = DataFrame(data15)
    data15 = data15.T
    data16 = np.array(data16)
    data16 = DataFrame(data16)
    data16 = data16.T
    data17 = np.array(data17)
    data17 = DataFrame(data17)
    data17 = data17.T
    data18 = np.array(data18)
    data18 = DataFrame(data18)
    data18 = data18.T
    data19 = np.array(data19)
    data19 = DataFrame(data19)
    data19 = data19.T
    data20 = np.array(data20)
    data20 = DataFrame(data20)
    data20 = data20.T
    data21 = np.array(data21)
    data21 = DataFrame(data21)
    data21 = data21.T
    data22 = np.array(data22)
    data22 = DataFrame(data22)
    data22 = data22.T
    data23 = np.array(data23)
    data23 = DataFrame(data23)
    data23 = data23.T
    data24 = np.array(data24)
    data24 = DataFrame(data24)
    data24 = data24.T
    data25 = np.array(data25)
    data25 = DataFrame(data25)
    data25 = data25.T

    '''将各个不同分布的数据合并在一起'''
    #, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25
    #
    data = data1.append([data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14,
                         data15])
    data = data.reset_index(drop=True)
    data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化

    '''如果数据是二维的，就画图展示数据'''
    if data.shape[1] == 2:
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
        # plt.plot(0.1, 0.5, 'g*')
        plt.plot(0.06, 0.41, 'g*')
        plt.xlabel("1st Dimension")
        plt.ylabel("2nd Dimension")
        # plt.title("normalized data")
        plt.show()

    return data

'''50个簇'''
# def GenerateManualData():
#     '''25个-- 2Ddata '''
#     size = 100
#     mean1 = [24, 27] # 30, 30
#     cov1 = [[3, 0], [0, 4]]
#     x1, y1 = np.random.multivariate_normal(mean1, cov1, size).T
#     # plt.plot(x1, y1, 'x')
#
#     mean2 = [93, 64] # 70, 20
#     cov2 = [[2, 0], [0, 1]]
#     x2, y2 = np.random.multivariate_normal(mean2, cov2, size).T
#     # plt.plot(x2, y2, 'x')
#
#     mean3 = [58, 23] # 0, 0
#     cov3 = [[4, 0], [0, 3]]
#     x3, y3 = np.random.multivariate_normal(mean3, cov3, size).T
#     # plt.plot(x3, y3, 'x')
#
#     mean4 = [90, 41] # 5, 70
#     cov4 = [[3, 0], [0, 3]]
#     x4, y4 = np.random.multivariate_normal(mean4, cov4, size).T
#     # plt.plot(x4, y4, 'x')
#
#     mean5 = [5, 2] # 12, 12
#     cov5 = [[3, 0], [0, 3]]
#     x5, y5 = np.random.multivariate_normal(mean5, cov5, size).T
#     # plt.plot(x4, y4, 'x')
#     # plt.show()
#
#     mean6 = [10, 70] # 50, 50
#     cov6 = [[2, 0], [0, 2]]
#     x6, y6 = np.random.multivariate_normal(mean6, cov6, size).T
#
#     mean7 = [40, 50] # 60, 23
#     cov7 = [[4, 0], [0, 3]]
#     x7, y7 = np.random.multivariate_normal(mean7, cov7, size).T
#
#     mean8 = [81, 90] # 74, 36
#     cov8 = [[3, 0], [0, 3]]
#     x8, y8 = np.random.multivariate_normal(mean8, cov8, size).T
#
#     mean9 = [1, 88] # 62, 48
#     cov9 = [[3, 0], [0, 3]]
#     x9, y9 = np.random.multivariate_normal(mean9, cov9, size).T
#
#     mean10 = [79, 4] #88, 20
#     cov10 = [[3, 0], [0, 3]]
#     x10, y10 = np.random.multivariate_normal(mean10, cov10, size).T
#
#     mean11 = [67, 38] # 40, 90
#     cov11 = [[3, 0], [0, 3]]
#     x11, y11 = np.random.multivariate_normal(mean11, cov11, size).T
#
#     mean12 = [37, 96] # 38, 76
#     cov12 = [[3, 0], [0, 3]]
#     x12, y12 = np.random.multivariate_normal(mean12, cov12, size).T
#
#     mean13 = [97, 21]
#     cov13 = [[3, 0], [0, 4]]
#     x13, y13 = np.random.multivariate_normal(mean13, cov13, size).T
#
#     mean14 = [30, 7]
#     cov14 = [[3, 0], [0, 4]]
#     x14, y14 = np.random.multivariate_normal(mean14, cov14, size).T
#
#     mean15 = [60, 80]
#     cov15 = [[3, 0], [0, 4]]
#     x15, y15 = np.random.multivariate_normal(mean15, cov15, size).T
#
#     mean16 = [70, 60]
#     cov16 = [[3, 0], [0, 4]]
#     x16, y16 = np.random.multivariate_normal(mean16, cov16, size).T
#
#     mean17 = [7, 46]
#     cov17 = [[3, 0], [0, 4]]
#     x17, y17 = np.random.multivariate_normal(mean17, cov17, size).T
#
#     mean18 = [45, 70]
#     cov18 = [[3, 0], [0, 4]]
#     x18, y18 = np.random.multivariate_normal(mean18, cov18, size).T
#
#     mean19 = [20, 86]
#     cov19 = [[3, 0], [0, 4]]
#     x19, y19 = np.random.multivariate_normal(mean19, cov19, size).T
#
#     mean20 = [26, 65]
#     cov20 = [[3, 0], [0, 4]]
#     x20, y20 = np.random.multivariate_normal(mean20, cov20, size).T
#
#     mean21 = [50, 2]
#     cov21 = [[3, 0], [0, 4]]
#     x21, y21 = np.random.multivariate_normal(mean21, cov21, size).T
#
#     mean22 = [45, 36]
#     cov22 = [[3, 0], [0, 4]]
#     x22, y22 = np.random.multivariate_normal(mean22, cov22, size).T
#
#     mean23 = [2, 35]
#     cov23 = [[3, 0], [0, 4]]
#     x23, y23 = np.random.multivariate_normal(mean23, cov23, size).T
#
#     mean24 = [12, 18]
#     cov24 = [[3, 0], [0, 4]]
#     x24, y24 = np.random.multivariate_normal(mean24, cov24, size).T
#
#     mean25 = [76, 26]
#     cov25 = [[3, 0], [0, 4]]
#     x25, y25 = np.random.multivariate_normal(mean25, cov25, size).T
#
#     mean26 = [94, 127]  # 30, 30
#     cov26 = [[3, 0], [0, 4]]
#     x26, y26 = np.random.multivariate_normal(mean26, cov26, size).T
#     # plt.plot(x1, y1, 'x')
#
#     mean27 = [193, 64]  # 70, 20
#     cov27 = [[2, 0], [0, 1]]
#     x27, y27 = np.random.multivariate_normal(mean27, cov27, size).T
#     # plt.plot(x2, y2, 'x')
#
#     mean28 = [108, 123]  # 0, 0
#     cov28 = [[4, 0], [0, 3]]
#     x28, y28 = np.random.multivariate_normal(mean28, cov28, size).T
#     # plt.plot(x3, y3, 'x')
#
#     mean29 = [190, 121]  # 5, 70
#     cov29 = [[3, 0], [0, 3]]
#     x29, y29 = np.random.multivariate_normal(mean29, cov29, size).T
#     # plt.plot(x4, y4, 'x')
#
#     mean30 = [85, 102]  # 12, 12
#     cov30 = [[3, 0], [0, 3]]
#     x30, y30 = np.random.multivariate_normal(mean30, cov30, size).T
#     # plt.plot(x4, y4, 'x')
#     # plt.show()
#
#     mean31 = [170, 110]  # 50, 50
#     cov31 = [[2, 0], [0, 2]]
#     x31, y31 = np.random.multivariate_normal(mean31, cov31, size).T
#
#     mean32 = [140, 150]  # 60, 23
#     cov32 = [[4, 0], [0, 3]]
#     x32, y32 = np.random.multivariate_normal(mean32, cov32, size).T
#
#     mean33 = [181, 190]  # 74, 36
#     cov33 = [[3, 0], [0, 3]]
#     x33, y33 = np.random.multivariate_normal(mean33, cov33, size).T
#
#     mean34 = [101, 188]  # 62, 48
#     cov34 = [[3, 0], [0, 3]]
#     x34, y34 = np.random.multivariate_normal(mean34, cov34, size).T
#
#     mean35 = [179, 104]  # 88, 20
#     cov35 = [[3, 0], [0, 3]]
#     x35, y35 = np.random.multivariate_normal(mean35, cov35, size).T
#
#     mean36 = [167, 138]  # 40, 90
#     cov36 = [[3, 0], [0, 3]]
#     x36, y36 = np.random.multivariate_normal(mean36, cov36, size).T
#
#     mean37 = [137, 196]  # 38, 76
#     cov37 = [[3, 0], [0, 3]]
#     x37, y37 = np.random.multivariate_normal(mean37, cov37, size).T
#
#     mean38 = [197, 121]
#     cov38 = [[3, 0], [0, 4]]
#     x38, y38 = np.random.multivariate_normal(mean38, cov38, size).T
#
#     mean39 = [130, 107]
#     cov39 = [[3, 0], [0, 4]]
#     x39, y39 = np.random.multivariate_normal(mean39, cov39, size).T
#
#     mean40 = [170, 160]
#     cov40 = [[3, 0], [0, 4]]
#     x40, y40 = np.random.multivariate_normal(mean40, cov40, size).T
#
#     mean41 = [107, 146]
#     cov41 = [[3, 0], [0, 4]]
#     x41, y41 = np.random.multivariate_normal(mean41, cov41, size).T
#
#     mean42 = [145, 170]
#     cov42 = [[3, 0], [0, 4]]
#     x42, y42 = np.random.multivariate_normal(mean42, cov42, size).T
#
#     mean43 = [120, 186]
#     cov43 = [[3, 0], [0, 4]]
#     x43, y43 = np.random.multivariate_normal(mean43, cov43, size).T
#
#     mean44 = [126, 165]
#     cov44 = [[3, 0], [0, 4]]
#     x44, y44 = np.random.multivariate_normal(mean44, cov44, size).T
#
#     mean45 = [150, 102]
#     cov45 = [[3, 0], [0, 4]]
#     x45, y45 = np.random.multivariate_normal(mean45, cov45, size).T
#
#     mean46 = [145, 136]
#     cov46 = [[3, 0], [0, 4]]
#     x46, y46 = np.random.multivariate_normal(mean46, cov46, size).T
#
#     mean47 = [102, 135]
#     cov47 = [[3, 0], [0, 4]]
#     x47, y47 = np.random.multivariate_normal(mean47, cov47, size).T
#
#     mean48 = [112, 118]
#     cov48 = [[3, 0], [0, 4]]
#     x48, y48 = np.random.multivariate_normal(mean48, cov48, size).T
#
#     mean49 = [176, 126]
#     cov49 = [[3, 0], [0, 4]]
#     x49, y49 = np.random.multivariate_normal(mean49, cov49, size).T
#
#     mean50 = [160, 180]
#     cov50 = [[3, 0], [0, 4]]
#     x50, y50 = np.random.multivariate_normal(mean50, cov50, size).T
#
#     data1 = []
#     data2 = []
#     data3 = []
#     data4 = []
#     data5 = []
#     data6 = []
#     data7 = []
#     data8 = []
#     data9 = []
#     data10 = []
#     data11 = []
#     data12 = []
#     data13 = []
#     data14 = []
#     data15 = []
#     data16 = []
#     data17 = []
#     data18 = []
#     data19 = []
#     data20 = []
#     data21 = []
#     data22 = []
#     data23 = []
#     data24 = []
#     data25 = []
#     data1.append(x1)
#     data1.append(y1)
#     data2.append(x2)
#     data2.append(y2)
#     data3.append(x3)
#     data3.append(y3)
#     data4.append(x4)
#     data4.append(y4)
#     data5.append(x5)
#     data5.append(y5)
#     data6.append(x6)
#     data6.append(y6)
#     data7.append(x7)
#     data7.append(y7)
#     data8.append(x8)
#     data8.append(y8)
#     data9.append(x9)
#     data9.append(y9)
#     data10.append(x10)
#     data10.append(y10)
#     data11.append(x11)
#     data11.append(y11)
#     data12.append(x12)
#     data12.append(y12)
#     data13.append(x13)
#     data13.append(y13)
#     data14.append(x14)
#     data14.append(y14)
#     data15.append(x15)
#     data15.append(y15)
#     data16.append(x16)
#     data16.append(y16)
#     data17.append(x17)
#     data17.append(y17)
#     data18.append(x18)
#     data18.append(y18)
#     data19.append(x19)
#     data19.append(y19)
#     data20.append(x20)
#     data20.append(y20)
#     data21.append(x21)
#     data21.append(y21)
#     data22.append(x22)
#     data22.append(y22)
#     data23.append(x23)
#     data23.append(y23)
#     data24.append(x24)
#     data24.append(y24)
#     data25.append(x25)
#     data25.append(y25)
#     data26 = []
#     data27 = []
#     data28 = []
#     data29 = []
#     data30 = []
#     data31 = []
#     data32 = []
#     data33 = []
#     data34 = []
#     data35 = []
#     data36 = []
#     data37 = []
#     data38 = []
#     data39 = []
#     data40 = []
#     data41 = []
#     data42 = []
#     data43 = []
#     data44 = []
#     data45 = []
#     data46 = []
#     data47 = []
#     data48 = []
#     data49 = []
#     data50 = []
#     data26.append(x26)
#     data26.append(y26)
#     data27.append(x27)
#     data27.append(y27)
#     data28.append(x28)
#     data28.append(y28)
#     data29.append(x29)
#     data29.append(y29)
#     data30.append(x30)
#     data30.append(y30)
#     data31.append(x31)
#     data31.append(y31)
#     data32.append(x32)
#     data32.append(y32)
#     data33.append(x33)
#     data33.append(y33)
#     data34.append(x34)
#     data34.append(y34)
#     data35.append(x35)
#     data35.append(y35)
#     data36.append(x36)
#     data36.append(y36)
#     data37.append(x37)
#     data37.append(y37)
#     data38.append(x38)
#     data38.append(y38)
#     data39.append(x39)
#     data39.append(y39)
#     data40.append(x40)
#     data40.append(y40)
#     data41.append(x41)
#     data41.append(y41)
#     data42.append(x42)
#     data42.append(y42)
#     data43.append(x43)
#     data43.append(y43)
#     data44.append(x44)
#     data44.append(y44)
#     data45.append(x45)
#     data45.append(y45)
#     data46.append(x46)
#     data46.append(y46)
#     data47.append(x47)
#     data47.append(y47)
#     data48.append(x48)
#     data48.append(y48)
#     data49.append(x49)
#     data49.append(y49)
#     data50.append(x50)
#     data50.append(y50)
#
#     data1 = np.array(data1)
#     data1 = DataFrame(data1)
#     data1 = data1.T
#     data2 = np.array(data2)
#     data2 = DataFrame(data2)
#     data2 = data2.T
#     data3 = np.array(data3)
#     data3 = DataFrame(data3)
#     data3 = data3.T
#     data4 = np.array(data4)
#     data4 = DataFrame(data4)
#     data4 = data4.T
#     data5 = np.array(data5)
#     data5 = DataFrame(data5)
#     data5 = data5.T
#     data6 = np.array(data6)
#     data6 = DataFrame(data6)
#     data6 = data6.T
#     data7 = np.array(data7)
#     data7 = DataFrame(data7)
#     data7 = data7.T
#     data8 = np.array(data8)
#     data8 = DataFrame(data8)
#     data8 = data8.T
#     data9 = np.array(data9)
#     data9 = DataFrame(data9)
#     data9 = data9.T
#     data10 = np.array(data10)
#     data10 = DataFrame(data10)
#     data10 = data10.T
#     data11 = np.array(data11)
#     data11 = DataFrame(data11)
#     data11 = data11.T
#     data12= np.array(data12)
#     data12 = DataFrame(data12)
#     data12 = data12.T
#     data13 = np.array(data13)
#     data13 = DataFrame(data13)
#     data13 = data13.T
#     data14 = np.array(data14)
#     data14 = DataFrame(data14)
#     data14 = data14.T
#     data15 = np.array(data15)
#     data15 = DataFrame(data15)
#     data15 = data15.T
#     data16 = np.array(data16)
#     data16 = DataFrame(data16)
#     data16 = data16.T
#     data17 = np.array(data17)
#     data17 = DataFrame(data17)
#     data17 = data17.T
#     data18 = np.array(data18)
#     data18 = DataFrame(data18)
#     data18 = data18.T
#     data19 = np.array(data19)
#     data19 = DataFrame(data19)
#     data19 = data19.T
#     data20 = np.array(data20)
#     data20 = DataFrame(data20)
#     data20 = data20.T
#     data21 = np.array(data21)
#     data21 = DataFrame(data21)
#     data21 = data21.T
#     data22 = np.array(data22)
#     data22 = DataFrame(data22)
#     data22 = data22.T
#     data23 = np.array(data23)
#     data23 = DataFrame(data23)
#     data23 = data23.T
#     data24 = np.array(data24)
#     data24 = DataFrame(data24)
#     data24 = data24.T
#     data25 = np.array(data25)
#     data25 = DataFrame(data25)
#     data25 = data25.T
#     data26 = np.array(data26)
#     data26 = DataFrame(data26)
#     data26 = data26.T
#     data27 = np.array(data27)
#     data27 = DataFrame(data27)
#     data27 = data27.T
#     data28 = np.array(data28)
#     data28 = DataFrame(data28)
#     data28 = data28.T
#     data29 = np.array(data29)
#     data29 = DataFrame(data29)
#     data29 = data29.T
#     data30 = np.array(data30)
#     data30 = DataFrame(data30)
#     data30 = data30.T
#     data31 = np.array(data31)
#     data31 = DataFrame(data31)
#     data31 = data31.T
#     data32 = np.array(data32)
#     data32 = DataFrame(data32)
#     data32 = data32.T
#     data33 = np.array(data33)
#     data33 = DataFrame(data33)
#     data33 = data33.T
#     data34 = np.array(data34)
#     data34 = DataFrame(data34)
#     data34 = data34.T
#     data35 = np.array(data35)
#     data35 = DataFrame(data35)
#     data35 = data35.T
#     data36 = np.array(data36)
#     data36 = DataFrame(data36)
#     data36 = data36.T
#     data37 = np.array(data37)
#     data37 = DataFrame(data37)
#     data37 = data37.T
#     data38 = np.array(data38)
#     data38 = DataFrame(data38)
#     data38 = data38.T
#     data39 = np.array(data39)
#     data39 = DataFrame(data39)
#     data39 = data39.T
#     data40 = np.array(data40)
#     data40 = DataFrame(data40)
#     data40 = data40.T
#     data41 = np.array(data41)
#     data41 = DataFrame(data41)
#     data41 = data41.T
#     data42 = np.array(data42)
#     data42 = DataFrame(data42)
#     data42 = data42.T
#     data43 = np.array(data43)
#     data43 = DataFrame(data43)
#     data43 = data43.T
#     data44 = np.array(data44)
#     data44 = DataFrame(data44)
#     data44 = data44.T
#     data45 = np.array(data45)
#     data45 = DataFrame(data45)
#     data45 = data45.T
#     data46 = np.array(data46)
#     data46 = DataFrame(data46)
#     data46 = data46.T
#     data47 = np.array(data47)
#     data47 = DataFrame(data47)
#     data47 = data47.T
#     data48 = np.array(data48)
#     data48 = DataFrame(data48)
#     data48 = data48.T
#     data49 = np.array(data49)
#     data49 = DataFrame(data49)
#     data49 = data49.T
#     data50 = np.array(data50)
#     data50 = DataFrame(data50)
#     data50 = data50.T
#
#     '''将各个不同分布的数据合并在一起'''
#     #  data13, data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25
#     # data26, data27, data28, data29, data30, data31, data32, data33, data34, data35, data36, data37, data38, data39, data40,
#     # data41, data42, data43, data44, data45, data46, data47, data48, data49, data50
#
#     #,data36, data37, data38, data39, data40, data41, data42, data43, data44, data45, data46, data47, data48, data49, data50
#     data = data1.append([data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
#                          data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25,
#                          data26, data27, data28, data29, data30, data31, data32, data33, data34, data35])
#     data = data.reset_index(drop=True)
#     data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化
#
#     '''如果数据是二维的，就画图展示数据'''
#     if data.shape[1] == 2:
#         plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
#         plt.plot(0.5, 0.05, 'g*')
#         plt.xlabel("1st Dimension")
#         plt.ylabel("2nd Dimension")
#         # plt.title("normalized data")
#         plt.show()
#
#     return data

def GenerateData(d, Max, k, size):
    """先求出所有的中心点的坐标，也就是每个方块中数据分布的均值"""
    if d == 0 or max == 0:
        print("k and max can not be zero!")
    else:
        mean = []
        medium = []
        '''当k<d时，选前k个维度进行对折k次'''
        if k < d: # k=2,d=3,max=100
            # medium = [] # 存放各个维度下的中心点medium=[[25.0, 75.0], [25.0, 75.0], [50.0]]
            for i in range(d):
                if i < k:
                    temp = [Max/4, Max/4+Max/2]
                    medium.append(temp)
                else:
                    temp = [Max/2]
                    medium.append(temp)
            for item in itertools.product(*medium): # 求medium各个维度之间的笛卡尔积，
                # mean=[(25.0, 25.0, 50.0), (25.0, 75.0, 50.0), (75.0, 25.0, 50.0), (75.0, 75.0, 50.0)]
                mean.append(item)
        else:
            m = k % d
            '''当k==nd时，在每个维度下对折k/d=n次，得到(2^k)^(1/d)个中心点，得到大小一样的2^k个块'''
            if m == 0: # k=4,d=2,max=100
                n = k / d
                num_of_each_dim = ((2**k)**(1/d))
                for i in range(d):
                    temp = []
                    for j in range(int(num_of_each_dim)):
                        dex = Max / (2 * num_of_each_dim) + j * (Max/num_of_each_dim)
                        temp.append(dex)
                    medium.append(temp)
                for item in itertools.product(*medium):  # 求medium各个维度之间的笛卡尔积，
                    mean.append(item)
                '''当k>d且k%d==m时，先在前m个维度下对折k/d+m=n+m次，再在剩下的维度下对折k/d=n次'''
            else:
                n = k / d
                num_of_each_front_dim = 2 * ((2 ** (k - m)) ** (1 / d))
                num_of_each_back_dim = ((2 ** (k - m)) ** (1 / d))
                for i in range(d):
                    temp = []
                    if i < m:
                        for j in range(int(num_of_each_front_dim)):
                            dex = Max / (2 * num_of_each_front_dim) + j * (Max / num_of_each_front_dim)
                            temp.append(dex)
                    else:
                        for j in range(int(num_of_each_back_dim)):
                            dex = Max / (2 * num_of_each_back_dim) + j * (Max / num_of_each_back_dim)
                            temp.append(dex)
                    medium.append(temp)
                for item in itertools.product(*medium):  # 求medium各个维度之间的笛卡尔积，
                    mean.append(item)

    '''再根据上面求得的各个均值生成相应的数据，各个维度之间互不相关'''
    num_of_distribution = 2**k
    cov = np.eye(d, dtype=int)
    cov = 1*cov

    mean_i = mean[0]
    data = np.random.multivariate_normal(mean_i, cov, size).T #
    data = DataFrame(data)
    data = data.T
    for i in range(1, num_of_distribution):
        mean_i = mean[i]
        data_temp = np.random.multivariate_normal(mean_i, cov, size).T
        data_temp = DataFrame(data_temp)
        data_temp = data_temp.T
        data = data.append([data_temp])

    data = data.reset_index(drop=True)
    # data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化
    data = data / 100
    # mean = mean / 100

    return mean, data

'''计算观察点与各个数据点之间的距离'''
def EuclideanDistances(A, B):
    BT = B.T
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    ED = DataFrame(ED)
    return (ED)

'''初始评估K值'''
'''在对数据I_niceSO算法前:
输入：数据集data
输出：Mmax的值
步骤
1------随机选择一个观测点，计算观测点到每个数据的距离;
2------将距离值排序
3------将距离值带入联合密度中求得密度值
4------判断密度值有几个峰值
5------该峰值设置为I_niceSO中的Mmax'''
def estimateOfM(data):
    # 利用均匀分布生成观测点
    # P = [[random.uniform(0, 1) for i in range(data.shape[1])] for j in range(1)]
    # P = [0.828346, 0.919648, 0.545383, 0.766754, 0.301461, 0.737973, 0.068853, 0.891594, 0.739138]
    # P = [[0.1 for i in range(Dim_of_data)] for j in range(Num_of_observation)] # 固定观测点的位置
    P = [[0.46, 0.15]]
    P = DataFrame(P)

    '''计算观测点与每个数据的距离'''
    XP = EuclideanDistances(P, data)

    '''将距离值一一带入联合概率密度函数中，求得密度值'''
    dis = XP.iloc[0]
    dis = np.array(dis)
    dis = dis.tolist()
    dis.sort()

    '''计算每一行的标准差'''
    std = np.std(dis)
    counts = len(dis)

    '''设置delta值'''
    #         delta = 1/(counts)
    delta = 2.06 * std * (counts ** -0.2) #h

    pdf_value = []
    for i in range(counts):  # 去掉了0值那一行
        temp = 0.0
        for j in range(counts):
            if (i == j):
                continue
            temp += (1 / (((2 * np.pi) ** 0.5) * delta)) * np.exp((-0.5) * (((dis[i] - dis[j]) / delta) ** 2))
        temp = (1 / (counts - 1)) * temp
        pdf_value.append(temp)

    # print(pdf_value)
    '''画图观察距离的概率密度'''
    plt.plot(dis, pdf_value, 'b-')
    plt.xlabel('Normalized Distance Value')
    plt.ylabel('Density')
    plt.title('Gamma Mixture Distribution')
    plt.show()

    '''判断峰值个数'''
    estimate_of_M = 0
    for i in range(1, counts - 1):
        if (pdf_value[i - 1] < pdf_value[i] and pdf_value[i] > pdf_value[i + 1]):
            estimate_of_M += 1

    return (estimate_of_M)

'''I_niceSO函数'''
def I_niceSO(dataset):
    """I_niceSO算法实现"""
    # estimate_of_M = estimateOfM(data)
    '''初始预估混合模型的个数，并根据这个数目设置混合模型个数的上下限'''
    # estimate_of_M = estimateOfM(dataset)  # I-nice+
    # if estimate_of_M < 3:
    #     Mmin = estimate_of_M+9
    #     Mmax = estimate_of_M+15
    # else:
    #     Mmin = estimate_of_M+9
    #     Mmax = estimate_of_M+13
    # Mmin = 4
    # Mmax = 8

    estimate_of_M =35 # I-nice
    Mmin = 15
    Mmax = estimate_of_M


    '''设置观察点个数'''
    Num_of_observation = 1

    '''生成Num_of_observation个观测点'''
    ## 法1：利用均匀分布生成Num_of_observation个观测点----1
    # viewPoint = [[random.uniform(0, 1) for i in range(dataset.shape[1])] for j in range(Num_of_observation)]
    # viewPoint = [[0.5, 0.2]]
    viewPoint = [[0.06, 0.41]]
    # 法2：生成固定位置的Num_of_observation个观测点----2
    # viewPoint = [[0.06, 0.41]] # 15个簇
    viewPoint = [[0.6, 0.08]]  # 20个簇，25， 30个簇
    # viewPoint = [[0.52, 0.81]]  # 35， 40个簇
    # viewPoint = [[0.28, 0.78]]  # 45个簇
    # viewPoint = [[0.31, 0.56]]  # 50个簇
    # viewPoint = [[0.6 for i in range(dataset.shape[1])] for j in range(Num_of_observation)]

    viewPoint = DataFrame(viewPoint)

    '''第一阶段：根据混合gamma模型得到的K值，找对应的初始聚类中心，如下所述：'''
    '''选择最好的混合gamma模型'''
    XP = EuclideanDistances(viewPoint, dataset)  # 各个观察点与数据的距离矩阵 distance vector: Xp  *********可并行执行
    N = XP.shape[1]  # length of data
    AICc_min = FloatVector([np.inf])  # 定义全局AICc最小值为默认最大值
    Num_of_cluster = 0
    for p in range(Num_of_observation):  # *******第400行--第445行的两个for循环可以并行
        Xp = XP.iloc[p]
        Xp_backup = Xp

        Xp_backup = Xp_backup.tolist()
        # sns.distplot(Xp_backup, rug=True)  # 画图
        # kde = stats.gaussian_kde(dataset)  # 构造多变量核密度评估函数
        # plt.show()

        Xp = FloatVector(Xp)
        # print(Xp)#画图观察
        '''画直方图看一下概率密度'''
        # plt.hist(Xp, bins=50, color='steelblue', normed=True)
        # plt.title("hist of distance")
        # plt.show()
        # Xp = MaxMinNormalization(Xp)#normalize distance values 会出现0值和1值，造成后面出现nan值
        AICc_p_min = FloatVector([np.inf])  # 定义为第p个观察点对应AICc的最小值
        Num_of_component = 0
        for M in range(Mmin, Mmax):  # model Xp to GMM(p,M)
            '''model Xp to GMM(p,M)'''
            # 模型参数初始化
            pai = [1 / M for i in range(M)]
            pai = FloatVector(pai)

            # EM算法求gammamix模型
            RES = mixtools.gammamixEM(Xp, pai, k=M)

            # 计算AICc值
            q = 3 * M
            tail = 2 * q * (N / (N - q - 1))
            head = (FloatVector([-2.0]).ro * RES[3])
            AICc_value_p_M = FloatVector([tail]).ro + head

            # print("*********p_M_after_EM*********" + str(p) + "----" + str(M) + "-----" + "AICc_value" + "**********" + str(AICc_value_p_M))
            # print(RES)  # 是一个FloatVector

            # EM结束后，求观察点p所对应的最小的AICc值 ，并且保存最小值对应的混合gamma模型参数:pai,alpha,beta,pro_matrix
            if ((AICc_value_p_M[0] - AICc_p_min[0]) < 0):
                AICc_p_min = AICc_value_p_M
                Num_of_component = M
                Pai_loc_optimal = RES[1]
                Alpha_beta_loc_optimal = RES[2]
                Pro_matrix_loc = RES[4]

        # 求所有观察点中AICc值最小的点对应的参数个数（即K值）以及其他模型参数（alpha，beta，pai），并且保存这些参数
        if ((AICc_p_min[0] - AICc_min[0]) < 0):
            AICc_min = AICc_p_min
            Num_of_cluster = Num_of_component
            observation_point = p
            Pai_global_optimal = Pai_loc_optimal
            Alpha_beta_global_optimal = Alpha_beta_loc_optimal
            Pro_matrix_global = Pro_matrix_loc

    #     print("Num_of_cluster: "+str(Num_of_cluster))
    #     print(Alpha_beta_global_optimal)
    # R-python的数据格式转换----将包含alpha和beta的matrix转换为dataframe.该dataframe的第一行为alpha，第二行为beta。然后分别将alpha和beta转换为list，便于后续使用
    Alpha_beta_global_optimal = np.array(list(Alpha_beta_global_optimal))
    #     print("将R中的格式转化为python中的格式")
    #     print(Alpha_beta_global_optimal)
    Alpha_beta_global_optimal = DataFrame(Alpha_beta_global_optimal.reshape(Num_of_cluster, 2))
    Alpha_beta_global_optimal = Alpha_beta_global_optimal.T
    #     print("Alpha_beta_global_optimal:**********")
    #     print(Alpha_beta_global_optimal)
    # 将alpha和beta分别提取出来
    Alpha_global_optimal = Alpha_beta_global_optimal.loc[0]
    #     print("Alpha_global_optimal###########")
    #     print(Alpha_global_optimal)
    Beta_global_optimal = Alpha_beta_global_optimal.loc[1]
    # 转化为list
    Alpha_global_optimal = Alpha_global_optimal.tolist()
    Beta_global_optimal = Beta_global_optimal.tolist()

    Pai_global_optimal = list(Pai_global_optimal)

    '''初始化簇心'''
    print("Num_of_cluster", Num_of_cluster)

    initial_centers = []
    initial_centers1 = []
    index = []

    '''第一阶段：根据混合gamma模型得到的K值，找对应的初始聚类中心，如下所述：'''

    P = np.array(list(Pro_matrix_global))  # 概率矩阵,可以通过该矩阵对数据进行划分。具体做法：给该概率矩阵增加两列，一列为最大值，一列为最大值所在的列（类别）
    P = DataFrame(P.reshape(Num_of_cluster, N))
    P = P.T
    # print("原始概率矩阵：")
    # print(P)
    P1 = np.array(P)
    P['max_value'] = P.max(axis=1)
    P['max_index'] = np.argmax(P1, axis=1)

    # print("添加了两列的概率矩阵：")
    # print(P)

    loc_of_viewPoint = viewPoint.iloc[observation_point]
    dis_Based_On_Best_ViewPoint = XP.iloc[observation_point]
    dis_Based_On_Best_ViewPoint = np.array(dis_Based_On_Best_ViewPoint)

    # sns.distplot(dis_Based_On_Best_ViewPoint, hist=False, rug=True)  # 画图
    # plt.show()

    '''分别计算出每个component中所包含的数据个数以及相对于的数据id'''
    counts_of_each_component = P['max_index'].value_counts().tolist()  # 每个component中所包含的数据个数
    # print("各个component中所包含的数据个数：")
    # print(counts_of_each_component)

    '''求每个component中的中心个数和中心'''
    objectsId_for_each_component = []  # 每个component中的数据对应的id
    len_of_counts_of_each_component = len(counts_of_each_component)
    print("len_of_counts_of_each_component",len_of_counts_of_each_component)
    for i in range(len_of_counts_of_each_component):
        temp = P[P['max_index'] == i].index.tolist()  # temp是component i 包含的object_id
        print("componet ", i, "baohandeid", temp)
        objectsId_for_each_component.append(temp)

        '''步骤1：找出包含在该component i 中的原始数据，并对其进行求相异度矩阵(两两距离矩阵)，利用KNN，找出该类的中心点'''
        rawdata_in_component_i = []
        dis_Based_On_Best_ViewPoint_Of_Class_i = []
        for j in objectsId_for_each_component[i]:  # 每个component中所包含的原始数据ID
            rawdata_in_component_i.append(dataset.iloc[j])  # 利用ID取出包含在类i中的原始数据
            dis_Based_On_Best_ViewPoint_Of_Class_i.append(dis_Based_On_Best_ViewPoint[j])
        rawdata_in_component_i = DataFrame(rawdata_in_component_i)
        # print("rawdata_in_component_i",rawdata_in_component_i)
        rawdata_in_component_i = rawdata_in_component_i.reset_index(drop=True)

        '''附加部分：画概率密度图'''
        sns.distplot(dis_Based_On_Best_ViewPoint_Of_Class_i, hist=False, rug=True)  # 画图
        plt.show()
        # print("类"+str(i)+"中的数据的ID")
        # print(DataFrame(objectsId_for_each_component[i]))

        '''用KNN找出每个类的中心:一个component中含有一个中心，密度最大的那个点为中心点'''
        '''先计算每个类中数据之间的距离，即相异度矩阵'''
        dissimilarity_matrix = EuclideanDistances(rawdata_in_component_i, rawdata_in_component_i)
        # print("类"+str(i)+"的相异度矩阵")
        # print(dissimilarity_matrix)
        # print("*******************************************************************************")

        '''方案：----对相异度矩阵中的每一行数据进行如下操作：求出每一行中距离值排名前k的距离，然后求平均.-->
        先取出密度值排行前50%的点，再计算两两间的距离，当距离大于一个阈值时，认为是不同的点'''
        k_of_KNN = math.floor(0.7*rawdata_in_component_i.shape[0]) #KNN的k值
        ks_dis_of_dissimilarity_matrix = []
        # print(counts_of_each_component[i])
        # print("*************************")
        candidates = []
        for j in range(rawdata_in_component_i.shape[0]):
            dissimilarity_matrix_j = dissimilarity_matrix.loc[j]
            sort_of_dissimilarity_matrix_j = dissimilarity_matrix_j.argsort() # 返回的是升序排序过后对应的序号\
            # print("第"+str(i)+"类的sort_of_dissimilarity_matrix"+str(j)+"的第"+str(k_of_KNN)+"个数据")
            # print(sort_of_dissimilarity_matrix_j)
            '''实施方案'''
            k_num = sort_of_dissimilarity_matrix_j[0:k_of_KNN] # 方案二
            ks_value = []
            for value in k_num:
                ks_value.append(dissimilarity_matrix.iloc[j, value])
            k_means = np.mean(ks_value)
            ks_dis_of_dissimilarity_matrix.append(k_means)

        ks_dis_of_dissimilarity_matrix = np.array(ks_dis_of_dissimilarity_matrix)
        sort_of_ks_dis_of_dissimilarity_matrix = ks_dis_of_dissimilarity_matrix.argsort()
        '''判断两两间的距离是否大于一个给定的阈值'''
        # 首先取出密度最高的点放入候选点candidates中
        # center_id_of_component_i = sort_of_ks_dis_of_dissimilarity_matrix[1]  # 中心点的序号
        # center_id = temp[center_id_of_component_i]
        # # index.append(center_id)
        # candidates.append(dataset.loc[center_id])

        '''取出密度值排行前50%的点'''
        for k in range(1,k_of_KNN):
            center_id_of_component_i = sort_of_ks_dis_of_dissimilarity_matrix[k]  # 中心点的序号
            center_id = temp[center_id_of_component_i]
            # index.append(center_id)
            candidates.append(dataset.loc[center_id])
        candidates = np.array(candidates)
        candidates = np.around(candidates, decimals=2)
        dis = EuclideanDistances(candidates, candidates)
        dis1 = []
        print(dis)
        for i1 in range(len(candidates)):
            for j1 in range(len(candidates)):
                if dis.iloc[i1, j1] > 0.0001:
                    dis1.append(dis.iloc[i1, j1])
                else:
                    print(dis.iloc[i1, j1], "location", i1, ",", j1)
        print(len(dis1))
        dis1.sort()
        print(dis1)
        temp = []
        counts = math.floor(len(dis1) * 0.50)
        print(counts)
        for i2 in range(counts):
            temp.append(dis1[i2])
        print(temp)

        # 计算阈值
        # threshold = np.mean(temp)
        threshold = 0.4
        print(threshold)

        for i3 in range(len(candidates)):
            dis.iloc[i3, i3] = float('inf')

        # 利用阈值合并中心点
        S = []
        i4 = 0
        for i4 in range(len(candidates)):
            point = []
            index = []
            point.append(candidates[i4])
            index.append(i4)
            for j4 in range(len(candidates)):
                if i4 != j4 and dis.iloc[i4, j4] < threshold:
                    point.append(candidates[j4])
                    index.append(j4)

            if len(index) > 1:
                print("index:", index)
                print("point:", point)
                new = np.mean(point, axis=0)
                print("new:", new)
                # S.append(new)
                loc = len(point)
                S.append(point[loc//2])
                initial_centers.append(point[loc//2])
            else:
                S.append(candidates[i])
                initial_centers.append(candidates[i])
        S = np.array(S)
        print(S)

        '''再来一次'''
        dis = EuclideanDistances(S, S)
        dis1 = []
        print(dis)
        for i1 in range(len(S)):
            for j1 in range(len(S)):
                if dis.iloc[i1, j1] > 0.0001:
                    dis1.append(dis.iloc[i1, j1])
                else:
                    print(dis.iloc[i1, j1], "location", i1, ",", j1)
        print(len(dis1))
        dis1.sort()
        print(dis1)
        temp = []
        counts = math.floor(len(dis1) * 0.30)
        print(counts)
        for i2 in range(counts):
            temp.append(dis1[i2])
        print(temp)

        # 计算阈值
        # threshold = np.mean(temp)
        threshold = 0.2
        print(threshold)

        for i3 in range(len(S)):
            dis.iloc[i3, i3] = float('inf')

        # 利用阈值合并中心点
        S1 = []
        i4 = 0
        for i4 in range(len(S)):
            point = []
            index = []
            point.append(S[i4])
            index.append(i4)
            for j4 in range(len(S)):
                if i4 != j4 and dis.iloc[i4, j4] < threshold:
                    point.append(S[j4])
                    index.append(j4)

            if len(index) > 1:
                print("index:", index)
                print("point:", point)
                new = np.mean(point, axis=0)
                print("new:", new)
                S1.append(new)
                initial_centers1.append(new)
                # loc = len(point)
                # S1.append(point[loc // 2])
                # initial_centers1.append(point[loc // 2])
            else:
                S1.append(S[i])
                initial_centers1.append(S[i])
        S1 = np.array(S1)
        print(S1)

        plt.plot(dataset.iloc[:, 0], dataset.iloc[:, 1], 'c+')
        plt.plot(rawdata_in_component_i.iloc[:, 0], rawdata_in_component_i.iloc[:, 1], 'x')
        plt.plot(S1[:, 0], S1[:, 1], 'ro')
        plt.plot(0.06, 0.41, 'g*')
        plt.xlabel("1st Dimension")
        plt.ylabel("2nd Dimension")
        plt.show()



        # ks_dis_of_dissimilarity_matrix = np.array(ks_dis_of_dissimilarity_matrix)
        # sort_of_ks_dis_of_dissimilarity_matrix = ks_dis_of_dissimilarity_matrix.argsort()
        #
        # center_id_of_component_i = sort_of_ks_dis_of_dissimilarity_matrix[1]  # 中心点的序号
        # center_id = temp[center_id_of_component_i]
        # # index.append(center_id)
        # initial_centers.append(dataset.loc[center_id])  # 将中心点对应的原数据加入中心点集合中
        #
        # center_id_of_component_i1 = sort_of_ks_dis_of_dissimilarity_matrix[2*k_of_KNN//3]  # 中心点的序号
        # center_id1 = temp[center_id_of_component_i1]
        # initial_centers.append(dataset.loc[center_id1])  # 将中心点对应的原数据加入中心点集合中

        # plt.plot(dataset.iloc[:, 0], dataset.iloc[:, 1], 'c+')
        # plt.plot(rawdata_in_component_i.iloc[:, 0], rawdata_in_component_i.iloc[:, 1], 'x')
        # # for i in range(len(center_Point_Subscript)):  # 画中心
        # plt.plot(dataset.iloc[center_id, 0], dataset.iloc[center_id, 1], 'ro')
        # plt.plot(dataset.iloc[center_id1, 0], dataset.iloc[center_id1, 1], 'ro')
        # plt.plot(0.06, 0.41, 'g*')
        # plt.xlabel("1st Dimension")
        # plt.ylabel("2nd Dimension")
        # # plt.title("data and cluster centers")
        # plt.show()


    # plt.show()
    initial_centers = DataFrame(initial_centers)
    initial_centers1 = DataFrame(initial_centers1)

    '''画图，当数据是二维的时候，可以画图观察'''
    if dataset.shape[1] == 2:
        plt.plot(dataset.iloc[:, 0], dataset.iloc[:, 1], 'x') # 画标准化后原始数据
        initial_centers1 = initial_centers1.reset_index(drop=True)
        # new_initial_center = new_initial_center.reset_index(drop=True)
        # for i in range(initial_centers.shape[1]): # 画中心
        plt.plot(0.06, 0.41, 'g*')
        plt.plot(initial_centers1.iloc[:, 0], initial_centers1.iloc[:, 1], 'ro')
            # plt.plot(new_initial_center.iloc[i, 0], new_initial_center.iloc[i, 1], 'ro')

        # for i in range(Num_of_observation): # 画观测点
        # plt.plot(viewPoint.iloc[observation_point, 0], viewPoint.iloc[observation_point, 1], 'g*')
        plt.xlabel('1st Dimension')
        plt.ylabel('2nd Dimension')
        # plt.title('cluster centers and data by I-nice')
        plt.show()

    # if dataset.shape[1] == 3:
    #     ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #     #  将数据点分成三部分画，在颜色上有区分度
    #     ax.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], dataset.iloc[:, 2], c='y')  # 绘制数据点
    #     initial_centers = initial_centers.reset_index(drop=True)
    #     ax.scatter(initial_centers.iloc[:, 0], initial_centers.iloc[:, 1], initial_centers.iloc[:, 2], c='b')  # 绘制数据点
    #     ax.scatter(viewPoint.iloc[:, 0], viewPoint.iloc[:, 1], viewPoint.iloc[:, 2], c='r')  # 绘制数据点
    #
    #     ax.set_zlabel('Z')  # 坐标轴
    #     ax.set_ylabel('Y')
    #     ax.set_xlabel('X')
    #     plt.title('cluster centers and data')
    #     plt.show()

    return observation_point, Num_of_cluster, initial_centers, viewPoint, index


if __name__ == '__main__':
    #### 测试数据集
    # -*-coding:utf-8 -*-

    '''1-- UCI数据集'''
    # path = 'E:\\I-nice\\iris_for_python.csv'
    # file = pd.read_csv(path, low_memory=False)

    # ***************************************************************
    # file = pd.read_csv('F:\\UCIdataset\\led7digit.txt')  # 以逗号为分隔符的格式默认为csv格式
    # # # file = np.loadtxt('E:\\UCIdataset\\twonorm.txt')  # 以空格分隔默认为txt格式
    # file = pd.DataFrame(file)
    # print(file.shape)
    # data = file.iloc[:, 0:7]
    # data = (data - data.min()) / (data.max() - data.min())  # 先将原始数据标准化
    # print(data.head(5))
    # print(type(data))
    # # data = np.array(data)
    #
    # sumTime = 0.0
    # for i in range(10):
    #     start = time.process_time()
    #     observation_point, Num_of_cluster, initial_centers, viewPoint, index = I_niceSO(data)
    #     end = time.process_time()
    #     runtime = end - start
    #     sumTime = sumTime + runtime
    #     print("第" + str(i) + "次运行时间为：" + str(runtime))
    #     print("Num_of_cluster: "+str(Num_of_cluster))
    #
    # print("INiceBasedOnKNN的平均运行时间为：" + str(sumTime / 10))
    #***************************************************************

    #
    # start = time.clock()
    # '''调用I_niceSO算法'''
    # observation_point, Num_of_cluster, initial_centers, viewPoint, index = I_niceSO(data)#, Num_of_cluster_improve
    # end = time.clock()
    # print("算法运行时间为：")
    # print(str(end - start))
    #
    # # print("观测点和所观测到的聚类数目：")
    # # print("observation_point："+str(observation_point)+"          Num_of_cluster："+str(Num_of_cluster))
    # print("*********************************************************************************************")
    # print("观察点的坐标：")
    # print(viewPoint.iloc[observation_point,:])
    # viewPoint.to_csv('E:\\UCIdataset\\viewpoint.csv')
    # print("*********************************************************************************************")
    # # print("聚类中心：")
    # # print(initial_centers)
    # # print("*********************************************************************************************")
    # # print("中心点的id：")
    # # print(index)
    #
    # '''将I_niceSO找出的中心点对应到原数据里面,使用原数据进行聚类'''
    # init_centers = []
    # for id in index:
    #     init_centers.append(data.loc[id])
    # init_centers = np.array(init_centers)
    # # print(init_centers)
    # # print(init_centers)
    # n_clusters = Num_of_cluster  # 聚类数目
    # [centroids, label, inertia] = cluster.k_means(data, n_clusters, init=init_centers)
    #
    # '''直接使用I_niceSO在降维后的数据集上找出来的中心点和降维后的数据进行聚类'''
    # # init_centers = np.array(initial_centers)
    # # n_clusters = Num_of_cluster  # 聚类数目
    # # [centroids, label, inertia] = cluster.k_means(newData, n_clusters, init=init_centers)
    # # cols_of_centroids = centroids.shape[1]
    #
    # h = np.array(centroids)
    # h = h.tolist()
    # plt.clf()
    # for i in range(Num_of_cluster):
    #     plt.plot(h[i], lw=1, label='cluster'+str(i))
    #
    # plt.grid(True)
    # plt.legend(loc=0)
    # plt.axis('tight')
    # plt.xlabel('date')
    # plt.ylabel('status')
    # # plt.title(str(n_clusters)+' centroids of '+str(size)+' data')
    # # plt.savefig(str(n_clusters)+' centroids of '+str(size)+' data')
    # plt.show()

    '''将聚类中心标记出来'''
    # if data.shape[1] == 2:
    #     plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
    #     initial_centers = initial_centers.reset_index(drop=True)
    #     # initial_centers = np.array(initial_centers)
    #     # initial_centers = initial_centers.tolist()
    #     # plt.clf()
    #     for i in range(Num_of_cluster):
    #         plt.plot(initial_centers.iloc[i, 0], initial_centers.iloc[i, 1], 'ro')
    #         # plt.plot(initial_centers[i], lw=1, label='cluster_'+str(i))
    #
    #     plt.show()
    # if data.shape[1] == 3:
    #     ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #     #  将数据点分成三部分画，在颜色上有区分度
    #     # ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c='y')  # 绘制数据点
    #     # ax.scatter(mean.iloc[:, 0], mean.iloc[:, 1], mean.iloc[:, 2], c='y')  # 绘制数据点
    #     initial_centers = initial_centers.reset_index(drop=True)
    #     ax.scatter(initial_centers.iloc[:, 0], initial_centers.iloc[:, 1], initial_centers.iloc[:, 2], c='b')  # 绘制数据点
    #     ax.scatter(viewPoint.iloc[:, 0], viewPoint.iloc[:, 1], viewPoint.iloc[:, 2], c='r')  # 绘制数据点
    #
    #     ax.set_zlabel('Z')  # 坐标轴
    #     ax.set_ylabel('Y')
    #     ax.set_xlabel('X')
    #     plt.title('cluster centers and data')
    #     plt.show()


    '''2-- 人工生成的数据'''
    # data = GenerateManualData()
    # data = np.array(data)
    data = np.loadtxt(open("30.csv", "rb"), delimiter=",", skiprows=0)
    print("读取成功！")
    print(type(data))
    data = DataFrame(data)
    sumTime = 0.0
    for i in range(3):
        start = time.process_time()
        observation_point, Num_of_cluster, initial_centers, viewPoint, index = I_niceSO(data)
        end = time.process_time()
        runtime = end - start
        sumTime = sumTime + runtime
        print("第" + str(i) + "次运行时间为：" + str(runtime))
        print("Num_of_cluster: "+str(Num_of_cluster))
        if data.shape[1] == 2:
            plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
            initial_centers = initial_centers.reset_index(drop=True)
            plt.plot(viewPoint.iloc[observation_point, 0], viewPoint.iloc[observation_point, 1], 'g*')
            # initial_centers = np.array(initial_centers)
            # initial_centers = initial_centers.tolist()
            # plt.clf()
            # for i in range(Num_of_cluster):
            plt.plot(initial_centers.iloc[:, 0], initial_centers.iloc[:, 1], 'ro')
                # plt.plot(initial_centers[i], lw=1, label='cluster_'+str(i))

            plt.show()
    print("INiceBasedOnKNN的平均运行时间为：" + str(sumTime / 10))
    # print("INiceBasedOnKNN的运行时间为：" + str(runtime))

    # 输出结果以及画图


    '''电网数据'''
    # path = 'E:\I-nice\sample_data_NONaN.csv'
    # data = pd.read_csv(path, low_memory=False)
    # sample_data = data.sample(frac=0.1)
    # dataset = sample_data.iloc[:, 0:93]
    # dataset = dataset.reset_index(drop=True)
    # print(dataset)

    '''电网数据'''
    # path = 'F:\Clustering_to_classifying\\indus_data.csv'
    # data_all = pd.read_csv(path, names=range(353))
    # # 工厂id和数据分离
    # data_all = data_all.iloc[1:4422, 0:348]
    # data_all = data_all.drop_duplicates(subset=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], keep='first')  # 去掉前面10列的重复项
    # num_of_rows = data_all.shape[0]
    # num_of_cols = data_all.shape[1]
    # factory_id = data_all.iloc[1:num_of_rows, 0:1]
    # data = data_all.iloc[1:num_of_rows, 1:num_of_cols]
    # # 对4421个工业类型的工厂中的一部分进行分析
    # # 随机提取出30%的数据
    # data_train = data.sample(frac=0.8, replace=False, weights=None, random_state=None, axis=0)
    # # print(data_train)
    # '''
    # rows = data.shape[0]
    # count = 0.6
    # size = round(count * rows)
    # train_id = factory_id.iloc[0:size, 0:1]
    # x_train = data.iloc[0:size, 0:347]
    # data = x_train
    # '''
    # data_train = data_train.reset_index(drop=True)
    # data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())  # 先将原始数据标准化
    # x_train = data_train.copy() # 用于聚类
    # # print(data_train)
    #
    # '''利用PCA(SVD)方法对数据进行降维'''
    # # x = np.array(data_train)
    # # pca = PCA(n_components=3)  # n_components=10时，3个类;n_components=8时，占比90%，1个类；n_components=2，3时，占比75%，80%，2个类；
    # # newData = pca.fit_transform(x)
    # # print(pca.explained_variance_)
    # # print(pca.explained_variance_ratio_)
    # # data_train = DataFrame(newData)
    # # # print(data_train)
    # # #再次标准化
    # # data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())  # 先将原始数据标准化
    #
    # '''直接选取方差大于0.3的那些列,总共有8列'''
    # num = []
    # for i in range(0, 348):
    #     num.append(i)
    # num = DataFrame(num)
    # # print(num)
    # std_value = data_train.std(axis=0, ddof=1)
    # std = pd.concat([num.iloc[1:, :], DataFrame(std_value)], axis=1)
    # std.columns = ['num', 'std_value']  # 对列命名
    # high_std_value = std[std['std_value'] > 0.29]
    # # print(high_std_value)
    # high_std_value.sort_values("std_value", inplace=True, ascending=False)
    # # print(high_std_value)
    #
    # # 取出方差大于0.3的那些列
    # cols = high_std_value['num']
    # # print(cols)
    # newData = []
    # for col in cols:
    #     temp = data_train.iloc[:, col]
    #     temp = np.array(temp)
    #     newData.append(temp)
    # newData = DataFrame(newData)
    # newData = newData.T
    # # print(newData)
    # # data_train = newData
    #
    # '''调用I_niceSO算法'''
    # observation_point, Num_of_cluster, initial_centers, viewPoint, index = I_niceSO(newData)#, Num_of_cluster_improve
    # print("观测点和所观测到的聚类数目：")
    # print("observation_point："+str(observation_point)+"          Num_of_cluster："+str(Num_of_cluster))
    # print("*********************************************************************************************")
    # print("观察点的坐标：")
    # print(viewPoint)
    # print("*********************************************************************************************")
    # print("聚类中心：")
    # print(initial_centers)
    # print("*********************************************************************************************")
    # print("中心点的id：")
    # print(index)
    #
    # '''将I_niceSO找出的中心点对应到原数据里面,使用原数据进行聚类'''
    # init_centers = []
    # for x in index:
    #     init_centers.append(data_train.loc[x])
    #
    # print("******************************************")
    # A = DataFrame(init_centers)
    # print(type(A))
    # # A = A.reset_index(drop=True, inplace=True)
    # print(A)
    # A = A.T
    # # init_centers1 = DataFrame(init_centers)
    # # init_centers1 = init_centers1.reset_index(drop=True, inplace=True)
    # # print("init_centers1:")
    # # print(type(init_centers1))
    # # # init_centers1 = init_centers1.iloc[:, 1:347]
    # # A.to_excel("F:\\lunwenxiugai\\centerByInice1.xls")  # 将结果写入excel直接画图
    #
    # init_centers = np.array(init_centers)
    # print(init_centers)
    #
    #
    # n_clusters = Num_of_cluster  # 聚类数目
    # [centroids, label, inertia] = cluster.k_means(data_train, n_clusters, init=init_centers)
    #
    # B = DataFrame(centroids)
    # B = B.T
    # B.to_excel("F:\\lunwenxiugai\\centerByKmeans1.xls")
    #
    # '''直接使用I_niceSO在降维后的数据集上找出来的中心点和降维后的数据进行聚类'''
    # # init_centers = np.array(initial_centers)
    # # n_clusters = Num_of_cluster  # 聚类数目
    # # [centroids, label, inertia] = cluster.k_means(newData, n_clusters, init=init_centers)
    # # cols_of_centroids = centroids.shape[1]
    #
    # h = np.array(centroids)
    # h = h.tolist()
    # plt.clf()
    # for i in range(Num_of_cluster):
    #     plt.plot(h[i], lw=1, label='cluster'+str(i))
    #
    # plt.grid(True)
    # plt.legend(loc=0)
    # plt.axis('tight')
    # plt.xlabel('date')
    # plt.ylabel('status')
    # # plt.title(str(n_clusters)+' centroids of '+str(size)+' data')
    # # plt.savefig(str(n_clusters)+' centroids of '+str(size)+' data')
    # plt.show()

    '''将聚类中心标记出来'''
    if data.shape[1] == 2:
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
        initial_centers = initial_centers.reset_index(drop=True)
        # initial_centers = np.array(initial_centers)
        # initial_centers = initial_centers.tolist()
        # plt.clf()
        for i in range(Num_of_cluster):
            plt.plot(initial_centers.iloc[i, 0], initial_centers.iloc[i, 1], 'ro')
            # plt.plot(initial_centers[i], lw=1, label='cluster_'+str(i))

        plt.show()
    # if data.shape[1] == 3:
    #     ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #     #  将数据点分成三部分画，在颜色上有区分度
    #     # ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c='y')  # 绘制数据点
    #     # ax.scatter(mean.iloc[:, 0], mean.iloc[:, 1], mean.iloc[:, 2], c='y')  # 绘制数据点
    #     initial_centers = initial_centers.reset_index(drop=True)
    #     ax.scatter(initial_centers.iloc[:, 0], initial_centers.iloc[:, 1], initial_centers.iloc[:, 2], c='b')  # 绘制数据点
    #     ax.scatter(viewPoint.iloc[:, 0], viewPoint.iloc[:, 1], viewPoint.iloc[:, 2], c='r')  # 绘制数据点
    #
    #     ax.set_zlabel('Z')  # 坐标轴
    #     ax.set_ylabel('Y')
    #     ax.set_xlabel('X')
    #     plt.title('cluster centers and data')
    #     plt.show()

