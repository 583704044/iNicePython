import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

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

    mean26 = [94, 127]  # 30, 30
    cov26 = [[3, 0], [0, 4]]
    x26, y26 = np.random.multivariate_normal(mean26, cov26, size).T
    # plt.plot(x1, y1, 'x')

    mean27 = [193, 64]  # 70, 20
    cov27 = [[2, 0], [0, 1]]
    x27, y27 = np.random.multivariate_normal(mean27, cov27, size).T
    # plt.plot(x2, y2, 'x')

    mean28 = [108, 123]  # 0, 0
    cov28 = [[4, 0], [0, 3]]
    x28, y28 = np.random.multivariate_normal(mean28, cov28, size).T
    # plt.plot(x3, y3, 'x')

    mean29 = [190, 121]  # 5, 70
    cov29 = [[3, 0], [0, 3]]
    x29, y29 = np.random.multivariate_normal(mean29, cov29, size).T
    # plt.plot(x4, y4, 'x')

    mean30 = [85, 102]  # 12, 12
    cov30 = [[3, 0], [0, 3]]
    x30, y30 = np.random.multivariate_normal(mean30, cov30, size).T
    # plt.plot(x4, y4, 'x')
    # plt.show()

    mean31 = [170, 110]  # 50, 50
    cov31 = [[2, 0], [0, 2]]
    x31, y31 = np.random.multivariate_normal(mean31, cov31, size).T

    mean32 = [140, 150]  # 60, 23
    cov32 = [[4, 0], [0, 3]]
    x32, y32 = np.random.multivariate_normal(mean32, cov32, size).T

    mean33 = [181, 190]  # 74, 36
    cov33 = [[3, 0], [0, 3]]
    x33, y33 = np.random.multivariate_normal(mean33, cov33, size).T

    mean34 = [101, 188]  # 62, 48
    cov34 = [[3, 0], [0, 3]]
    x34, y34 = np.random.multivariate_normal(mean34, cov34, size).T

    mean35 = [179, 104]  # 88, 20
    cov35 = [[3, 0], [0, 3]]
    x35, y35 = np.random.multivariate_normal(mean35, cov35, size).T

    mean36 = [167, 138]  # 40, 90
    cov36 = [[3, 0], [0, 3]]
    x36, y36 = np.random.multivariate_normal(mean36, cov36, size).T

    mean37 = [137, 196]  # 38, 76
    cov37 = [[3, 0], [0, 3]]
    x37, y37 = np.random.multivariate_normal(mean37, cov37, size).T

    mean38 = [197, 121]
    cov38 = [[3, 0], [0, 4]]
    x38, y38 = np.random.multivariate_normal(mean38, cov38, size).T

    mean39 = [130, 107]
    cov39 = [[3, 0], [0, 4]]
    x39, y39 = np.random.multivariate_normal(mean39, cov39, size).T

    mean40 = [170, 160]
    cov40 = [[3, 0], [0, 4]]
    x40, y40 = np.random.multivariate_normal(mean40, cov40, size).T

    mean41 = [107, 146]
    cov41 = [[3, 0], [0, 4]]
    x41, y41 = np.random.multivariate_normal(mean41, cov41, size).T

    mean42 = [145, 170]
    cov42 = [[3, 0], [0, 4]]
    x42, y42 = np.random.multivariate_normal(mean42, cov42, size).T

    mean43 = [120, 186]
    cov43 = [[3, 0], [0, 4]]
    x43, y43 = np.random.multivariate_normal(mean43, cov43, size).T

    mean44 = [126, 165]
    cov44 = [[3, 0], [0, 4]]
    x44, y44 = np.random.multivariate_normal(mean44, cov44, size).T

    mean45 = [150, 102]
    cov45 = [[3, 0], [0, 4]]
    x45, y45 = np.random.multivariate_normal(mean45, cov45, size).T

    mean46 = [145, 136]
    cov46 = [[3, 0], [0, 4]]
    x46, y46 = np.random.multivariate_normal(mean46, cov46, size).T

    mean47 = [102, 135]
    cov47 = [[3, 0], [0, 4]]
    x47, y47 = np.random.multivariate_normal(mean47, cov47, size).T

    mean48 = [112, 118]
    cov48 = [[3, 0], [0, 4]]
    x48, y48 = np.random.multivariate_normal(mean48, cov48, size).T

    mean49 = [176, 126]
    cov49 = [[3, 0], [0, 4]]
    x49, y49 = np.random.multivariate_normal(mean49, cov49, size).T

    mean50 = [160, 180]
    cov50 = [[3, 0], [0, 4]]
    x50, y50 = np.random.multivariate_normal(mean50, cov50, size).T

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
    data26 = []
    data27 = []
    data28 = []
    data29 = []
    data30 = []
    data31 = []
    data32 = []
    data33 = []
    data34 = []
    data35 = []
    data36 = []
    data37 = []
    data38 = []
    data39 = []
    data40 = []
    data41 = []
    data42 = []
    data43 = []
    data44 = []
    data45 = []
    data46 = []
    data47 = []
    data48 = []
    data49 = []
    data50 = []
    data26.append(x26)
    data26.append(y26)
    data27.append(x27)
    data27.append(y27)
    data28.append(x28)
    data28.append(y28)
    data29.append(x29)
    data29.append(y29)
    data30.append(x30)
    data30.append(y30)
    data31.append(x31)
    data31.append(y31)
    data32.append(x32)
    data32.append(y32)
    data33.append(x33)
    data33.append(y33)
    data34.append(x34)
    data34.append(y34)
    data35.append(x35)
    data35.append(y35)
    data36.append(x36)
    data36.append(y36)
    data37.append(x37)
    data37.append(y37)
    data38.append(x38)
    data38.append(y38)
    data39.append(x39)
    data39.append(y39)
    data40.append(x40)
    data40.append(y40)
    data41.append(x41)
    data41.append(y41)
    data42.append(x42)
    data42.append(y42)
    data43.append(x43)
    data43.append(y43)
    data44.append(x44)
    data44.append(y44)
    data45.append(x45)
    data45.append(y45)
    data46.append(x46)
    data46.append(y46)
    data47.append(x47)
    data47.append(y47)
    data48.append(x48)
    data48.append(y48)
    data49.append(x49)
    data49.append(y49)
    data50.append(x50)
    data50.append(y50)

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
    data26 = np.array(data26)
    data26 = DataFrame(data26)
    data26 = data26.T
    data27 = np.array(data27)
    data27 = DataFrame(data27)
    data27 = data27.T
    data28 = np.array(data28)
    data28 = DataFrame(data28)
    data28 = data28.T
    data29 = np.array(data29)
    data29 = DataFrame(data29)
    data29 = data29.T
    data30 = np.array(data30)
    data30 = DataFrame(data30)
    data30 = data30.T
    data31 = np.array(data31)
    data31 = DataFrame(data31)
    data31 = data31.T
    data32 = np.array(data32)
    data32 = DataFrame(data32)
    data32 = data32.T
    data33 = np.array(data33)
    data33 = DataFrame(data33)
    data33 = data33.T
    data34 = np.array(data34)
    data34 = DataFrame(data34)
    data34 = data34.T
    data35 = np.array(data35)
    data35 = DataFrame(data35)
    data35 = data35.T
    data36 = np.array(data36)
    data36 = DataFrame(data36)
    data36 = data36.T
    data37 = np.array(data37)
    data37 = DataFrame(data37)
    data37 = data37.T
    data38 = np.array(data38)
    data38 = DataFrame(data38)
    data38 = data38.T
    data39 = np.array(data39)
    data39 = DataFrame(data39)
    data39 = data39.T
    data40 = np.array(data40)
    data40 = DataFrame(data40)
    data40 = data40.T
    data41 = np.array(data41)
    data41 = DataFrame(data41)
    data41 = data41.T
    data42 = np.array(data42)
    data42 = DataFrame(data42)
    data42 = data42.T
    data43 = np.array(data43)
    data43 = DataFrame(data43)
    data43 = data43.T
    data44 = np.array(data44)
    data44 = DataFrame(data44)
    data44 = data44.T
    data45 = np.array(data45)
    data45 = DataFrame(data45)
    data45 = data45.T
    data46 = np.array(data46)
    data46 = DataFrame(data46)
    data46 = data46.T
    data47 = np.array(data47)
    data47 = DataFrame(data47)
    data47 = data47.T
    data48 = np.array(data48)
    data48 = DataFrame(data48)
    data48 = data48.T
    data49 = np.array(data49)
    data49 = DataFrame(data49)
    data49 = data49.T
    data50 = np.array(data50)
    data50 = DataFrame(data50)
    data50 = data50.T

    '''将各个不同分布的数据合并在一起'''
    #  data13, data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25
    # data26, data27, data28, data29, data30, data31, data32, data33, data34, data35, data36, data37, data38, data39, data40,
    # data41, data42, data43, data44, data45, data46, data47, data48, data49, data50
    data = data1.append([data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13,
                         data14, data15, data16, data17, data18, data19, data20, data21, data22, data23, data24, data25,
                         data26, data27, data28, data29, data30, data31, data32, data33, data34, data35, data36, data37,
                         data38, data39, data40, data41, data42, data43, data44, data45, data46, data47, data48, data49, data50])
    data = data.reset_index(drop=True)
    data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化

    '''如果数据是二维的，就画图展示数据'''
    if data.shape[1] == 2:
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'x')
        plt.xlabel("x_axis")
        plt.ylabel("y_axis")
        plt.title("normalized data")
        plt.show()

    return data

def GenerateManualData1():
    mean1 = [24, 60] # 30, 30
    cov1 = [[3, 0], [0, 3]]
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 12).T
    plt.plot(x1, y1, 'o')
    print(x1)
    print(y1)

    mean2 = [55, 25]  # 70, 20
    cov2 = [[3, 0], [0, 3]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 15).T
    plt.plot(x2, y2, 'o')
    print(x2)
    print(y2)

    mean3 = [30, 30]  # 30, 30
    cov3 = [[3, 0], [0, 3]]
    x3, y3 = np.random.multivariate_normal(mean3, cov3, 1).T
    plt.plot(x3, y3, 'o')

    mean4 = [20, 20]  # 30, 30
    cov4 = [[3, 0], [0, 4]]
    x4, y4 = np.random.multivariate_normal(mean4, cov4, 1).T
    plt.plot(x4, y4, 'o')

    mean5 = [75, 75]  # 30, 30
    cov5 = [[3, 0], [0, 4]]
    x5, y5 = np.random.multivariate_normal(mean5, cov5, 1).T
    plt.plot(x5, y5, 'o')

    plt.show()


    data1 = []
    data2 = []
    data1.append(x1)
    data1.append(y1)
    data2.append(x2)
    data2.append(y2)
    data1 = np.array(data1)
    data1 = DataFrame(data1)
    data1 = data1.T
    data2 = np.array(data2)
    data2 = DataFrame(data2)
    data2 = data2.T
    data = data1.append([data2])
    data = data.reset_index(drop=True)
    # data = (data - data.min()) / (data.max() - data.min())  # 将原始数据标准化

    if data.shape[1] == 2:
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'o')
        plt.xlabel("x_axis")
        plt.ylabel("y_axis")
        plt.title("normalized data")
        plt.show()
    return data

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

def DPC(data, t):
    N = len(data) # 数据个数
    '''1、初始化及预处理'''
    # 1---计算距离矩阵
    dis_Matrix = EuclideanDistances(data, data)
    print("type dis_Matrix:", type(dis_Matrix))
    # print("dis_Matrix.shape", dis_Matrix.shape)
    ascend_Sort_Of_disMatrix = []
    for i in range(1, N):
        for j in range(i):
            ascend_Sort_Of_disMatrix.append(dis_Matrix[i, j])
    # print("len(ascend_Sort_Of_disMatrix):", len(ascend_Sort_Of_disMatrix))

    # 2---确定截断距离dc,将M=row*(row-1)/2个距离进行升序排序
    ascend_Sort_Of_disMatrix.sort()
    # print("ascend_Sort_Of_disMatrix:", ascend_Sort_Of_disMatrix)
    M = N*(N-1)/2
    dc = ascend_Sort_Of_disMatrix[round(M*t)]
    # print("Mt:",round(M*t))
    # print("截断距离为：", dc)

    #  3--计算密度每个点的density（ρi）以及生成其降序排序的下序标lower_Sequence（qi），使用高斯核
    density = []
    for i in range(N):
        density_i = 0
        for j in range(N):
            if i != j:
                # 高斯核
                density_i += np.exp(-((dis_Matrix[i, j]/dc)**2))
                # 线性核
                # if (dis_Matrix[i, j]-dc)>0:
                #     density_i += 1
        density.append(density_i)
    # density = DataFrame(density)
    density = np.array(density)
    # print("np.array(density):", density)
    # print("density len:", len(density))
    descend_Subscript = np.argsort(-density)

    # print("密度值的下标序：", descend_Subscript)

    # 4--计算每个点与比它密度更大的数据点之间的距离中的最小距离sigma（σi）及σi对应着的编号（ni）
    sigma = [max(ascend_Sort_Of_disMatrix)+0.001 for i in range(N)]
    # print("initial sigma:", sigma)

    ni = [0 for i in range(N)] # 每个点与比它密度更大的数据点之间的距离中的最小距离对应的编号
    for i in range(1, N):
        for j in range(i):
            # if dist(Xqi,Xqj)<σqi
            if dis_Matrix[descend_Subscript[i], descend_Subscript[j]] < sigma[descend_Subscript[i]]:
                sigma[descend_Subscript[i]] = dis_Matrix[descend_Subscript[i], descend_Subscript[j]]
                ni[descend_Subscript[i]] = descend_Subscript[j]
    sigma[descend_Subscript[1]] = min(sigma[1:])
    # print("final sigma:", sigma)
    # print("每个点与比它密度更大的数据点之间的距离中的最小距离:", ni)

    # 图1 ------画图观察密度值与对应的距离值
    plt.plot(density, sigma, 'x')
    plt.xlabel("density")
    plt.ylabel("sigma")
    # plt.title("sigma and density")
    plt.show()

    ''' 2、确定聚类中心 '''
    center_Point_Subscript = []                      # 中心点对应的下标
    initial_Class_Of_Data = [-1 for i in range(N)]  # 数据点归类属性标志，即属于哪一个类
    # 找出中心点：密度和距离都很大 怎么度量呢？*************************************************************************
    threhold_of_density = (np.max(density) + np.min(density)) / 8  # +np.mean(density)/2
    threhold_of_sigma = (np.max(sigma) + np.min(sigma)) / 12
    # threhold_of_sigma = np.mean(sigma)
    for i in range(N):
        if density[i] > threhold_of_density and sigma[i] > threhold_of_sigma:
            center_Point_Subscript.append(i)

    # 图2-------画图观察一下所找到的中心点是什么
    plt.plot(data[:, 0], data[:, 1], 'x')
    for i in range(len(center_Point_Subscript)):  # 画中心
        plt.plot(data[center_Point_Subscript[i], 0], data[center_Point_Subscript[i], 1], 'ro')
    plt.xlabel("x_axis")
    plt.ylabel("y_axis")
    plt.title("data and cluster centers by DPC")
    plt.show()

    # 初始化数据点归类属性标记
    for i in range(N):
        if i in center_Point_Subscript:
            initial_Class_Of_Data[i] = center_Point_Subscript.index(i)

    print("center_Point_Subscript:", center_Point_Subscript)
    print("first----initial_Class_Of_Data:", initial_Class_Of_Data)

    ''' 3、对非聚类中心数据点进行归类 '''
    for i in range(N):
        if initial_Class_Of_Data[descend_Subscript[i]] == -1:
            initial_Class_Of_Data[descend_Subscript[i]] = initial_Class_Of_Data[ni[descend_Subscript[i]]]

    print("second----initial_Class_Of_Data:", initial_Class_Of_Data)
    # 统计各个簇中数据点的个数
    print("各个簇中数据点的个数", pd.value_counts(initial_Class_Of_Data))

    ''' 4、（若类别数nc大于1）将每个类中的数据点进一步分为cluster core和cluster halo '''
    # 1--初始化标记----0标识cluster core，1表示cluster halo
    tag = [0 for i in range(N)]
    # 2--为每一个cluster生成一个平均局部密度上界-----先为每一个类初始化一个平均密度上界值
    density_Upper_Bound_Of_Cluster = [0 for i in range(len(center_Point_Subscript))]
    for i in range(N-1):
        for j in range(i+1, N):
            if initial_Class_Of_Data[i] != initial_Class_Of_Data[j] and dis_Matrix[i, j] < dc:
                mean_density = (density[i]+density[j])/2
                if mean_density > density_Upper_Bound_Of_Cluster[initial_Class_Of_Data[i]]:
                    density_Upper_Bound_Of_Cluster[initial_Class_Of_Data[i]] = mean_density
                if mean_density > density_Upper_Bound_Of_Cluster[initial_Class_Of_Data[j]]:
                    density_Upper_Bound_Of_Cluster[initial_Class_Of_Data[j]] = mean_density

    # 3--标识cluster halo
    for i in range(N):
        if density[i] < density_Upper_Bound_Of_Cluster[initial_Class_Of_Data[i]]:
            tag[i] = 1

    print("density_Upper_Bound_Of_Cluster", density_Upper_Bound_Of_Cluster)
    print("数据点标记tag：", tag)

    return density

if __name__=="__main__":
    GenerateManualData()
    # data = [[26.2, 59], [25.8,58], [24.4,58.5], [22.6,6], [23.6,62.4], [26,63], [23.4,59], [25,59.5], [25.8,56], [26.8,61.8], [27,57],
    #         [24.6, 59.3], [52.9,24.3], [56.4,26.1], [55.6,25.2], [54.8,28.6], [58.4,25.6], [54.2,26.5], [53.8,27], [54.8,28], [56.2,22],
    #         [54.1,26], [52.1,23.3], [56,25.6], [55.7,28], [54.4,24], [56.9,22.5], [30,30], [20,20], [75,75]]
    # data = np.array(data)
    # # print(data)
    # # print(data[0])
    # plt.plot(data[:,0], data[:,1], 'x')
    # plt.show()
    # DPC(data, 0.1)