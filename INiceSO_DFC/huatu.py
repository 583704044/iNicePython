import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
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
    print("np.array(density):", density)
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
    print("final sigma:", sigma)
    # print("每个点与比它密度更大的数据点之间的距离中的最小距离:", ni)

    # 图1 ------画图观察密度值与对应的距离值
    plt.plot(density, sigma, 'o')
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

    return center_Point_Subscript

if __name__=="__main__":
    # x = [26.2,25.8,24.4,22.6,23.6,26,23.4,25,25.8,26.8,27,24.6,52.9,56.4,55.6,54.8,60.4,54.2,53.8,54.8,56.2,54.1,52.1,56,55.7,54.4,56.9,30,20,75]
    # y = [59,58,58.5,6,62.4,63,59,59.5,56,61.8,57,59.3,24.3,26.1,25.2,28.6,25.6,26.5,27,28,22,26,23.3,25.6,28,24,22.5,30,20,75]
    # n=np.arange(30)
    #
    # fig, ax=plt.subplots()
    # ax.scatter(x,y,c='r')
    #
    # for i,txt in enumerate(n):
    #     ax.annotate(txt,(x[i],y[i]))
    # plt.show()



    data = [[26.2,28,29,26,28,29,28,27,24,23,23,24,55,56,58,60,60.4,50,52,53,56.2,54.1,52.1,58,51,54.4,56.9,30,20,75],
            [60, 50, 52, 52, 56, 59, 65, 68, 64, 60, 50, 54,25,28,30,32,25.6,26.5,31,28,33,32,24,25.6,22,20,22.5,30,20,75]]
    data = np.array(data)
    data = DataFrame(data)
    data = data.T
    data = np.array(data)
    # print(data)

    center_Point_Subscript = DPC(data, 0.1)
    for i in range(len(center_Point_Subscript)):  # 画中心
        plt.plot(data[center_Point_Subscript[i], 0], data[center_Point_Subscript[i], 1], 'ro')
    x1 = [26.2, 28, 29, 26, 28, 29, 28, 27, 24, 23, 23, 24]
    # [26.2,25.8,24.4,22.6,23.6,26,23.4,25,25.8,26.8,27,24.6]

    y1 = [60, 50, 52, 52, 56, 59, 65, 68, 64, 60, 50, 54]
    # [59,58,58.5,60,62.4,63,59,59.5,56,61.8,57,59.3]
    n1 = np.arange(12)
    n2 = np.arange(15)
    n3 = np.arange(3)
    # print(n)
    fig, ax = plt.subplots()
    ax.scatter(x1, y1, c='r')

    x2 = [55, 56, 58, 60, 60.4, 50, 52, 53, 56.2, 54.1, 52.1, 58, 51, 54.4, 56.9]
    # [52.9,56.4,55.6,54.8,60.4,54.2,53.8,54.8,56.2,54.1,52.1,56,55.7,54.4,56.9]

    y2 = [25, 28, 30, 32, 25.6, 26.5, 31, 28, 33, 32, 24, 25.6, 22, 20, 22.5]
    # [24.3,26.1,25.2,28.6,25.6,26.5,27,28,22,26,23.3,25.6,28,24,22.5]

    ax.scatter(x2, y2, c='b')

    x3 = [30, 20, 75]
    y3 = [30, 20, 75]
    ax.scatter(x3, y3, c='g')

    for i, txt in enumerate(n1):
        ax.annotate(txt, (x1[i], y1[i]))
    for i, txt in enumerate(n2):
        ax.annotate(txt + 12, (x2[i], y2[i]))
    for i, txt in enumerate(n3):
        ax.annotate(txt + 27, (x3[i], y3[i]))
    plt.show()

    density1 = [2.62,2.23,2.45,3.11,2.7,2.04,1.64,1.03,1.71,1.74,1.53,2.44]
    sigma1 = [4.39,2.24,3,34.49,4.47,2.97,4.12,3.16,4.12,3.2,3.61,2.83]

    n1 = np.arange(12)
    n2 = np.arange(15)
    n3 = np.arange(3)
    # print(n)
    fig, ax = plt.subplots()
    ax.scatter(density1, sigma1, c='r')

    density2 = [4.72,4.88,3.83,1.99,2.35,2.65,2.93,4.42,2.87,3.35,3.79,3.98,2.44,2.03,3.15]
    sigma2 = [2.24,77.78,2.83,2.83,2.4,3.26,2.33,3,3.33,4.15,3.07,3.06,2.28,3.54,3.14]
    ax.scatter(density2, sigma2, c='b')

    density3 = [0.0008,0.00082,0.00005]
    sigma3 = [20.1,14.14,45.54]
    ax.scatter(density3, sigma3, c='g')

    for i, txt in enumerate(n1):
        ax.annotate(txt, (density1[i], sigma1[i]))
    for i, txt in enumerate(n2):
        ax.annotate(txt + 12, (density2[i], sigma2[i]))
    for i, txt in enumerate(n3):
        ax.annotate(txt + 27, (density3[i], sigma3[i]))
    plt.show()

