import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import math

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
    ED = DataFrame(ED)
    return ED

# C = [[0.07933844, 0.75400322], [0.32472789, 0.36725959], [0.33721523, 0.35500932], [0.3470787,  0.35028246],
#      [0.08318953, 0.16576949], [0.02286318, 0.06865006], [0.41634497, 0.79991237], [0.01688832, 0.05068599],
#      [0.5566844,  0.54603132], [0.02138639, 0.0363244 ], [0.56909524, 0.55465393], [0.66559822, 0.52764802],
#      [0.4417049,  0.95053118], [0.44711842, 0.97210234], [0.66846033, 0.28121496], [0.81199358, 0.41361252],
#      [0.96312649, 0.25303859]]
# C = [[0.26839579, 0.27441629], [0.1297667,  0.69790625], [0.08473123, 0.04254084], [0.41375021, 0.50368963],
#      [0.07689271, 0.02398829], [0.43283131, 0.52062319], [0.31523679, 0.07709492], [0.05035519, 0.87619462],
#      [0.59441872, 0.23323904], [0.37007129, 0.93698705], [0.67221023, 0.39107056], [0.69186161, 0.37977741],
#      [0.39165391, 0.9578445 ], [0.61049283, 0.79932035], [0.39159976, 0.97765099], [0.79296894, 0.04991226],
#      [0.87526263, 0.41516337], [0.81050403, 0.04237358], [0.90161257, 0.41639073], [0.80187158, 0.89348126],
#      [0.91339764, 0.6367877 ], [0.82479695, 0.89512102], [0.93067574, 0.64321113], [0.96283421, 0.21556653]]

C = [[0.1297667,  0.69790625], [0.26560705, 0.2753216 ], [0.08090504, 0.03913251], [0.41375021, 0.50368963],
     [0.43662775, 0.51498082], [0.31567878, 0.07857178], [0.05035519, 0.87619462], [0.57829408, 0.23612026],
     [0.38711649, 0.94585234], [0.67946564, 0.38521599], [0.39168971, 0.96888178], [0.60781669, 0.80229738],
     [0.89294744, 0.41794391], [0.79296894, 0.04991226], [0.91441359, 0.63693318], [0.93479227, 0.6429793 ],
     [0.82061272, 0.90484577], [0.96239107, 0.21743729]]

C = np.array(C)
C = np.around(C, decimals=2)
print(C)

plt.plot(C[:, 0], C[:, 1], 'ro')
plt.plot(0.06, 0.41, 'g*')
plt.xlabel("1st Dimension")
plt.ylabel("2nd Dimension")
plt.show()


dis = EuclideanDistances(C, C)
dis1 = []
print(dis)
for i in range(len(C)):
    for j in range(len(C)):
        if dis.iloc[i,j]>0.0001:
            dis1.append(dis.iloc[i,j])
        else:
            print(dis.iloc[i,j],"location",i,",",j)
print(len(dis1))
dis1.sort()
print(dis1)
temp = []
counts = math.ceil(len(dis1)*0.05)
print(counts)
for i in range(counts):
    temp.append(dis1[i])
print(temp)

# 计算阈值
threshold = np.mean(temp)
print(threshold)

for i in range(len(C)):
    dis.iloc[i,i] = float('inf')

# 利用阈值合并中心点
S = []
i = 0
for i in range(len(C)):
    point = []
    index = []
    point.append(C[i])
    index.append(i)
    for j in range(len(C)):
        if i!=j and dis.iloc[i,j]<threshold:
            point.append(C[j])
            index.append(j)

    if len(index)>1:
        print("index:", index)
        print("point:", point)
        new = np.mean(point, axis=0)
        print("new:", new)
        S.append(new)
    else:
        S.append(C[i])

S = np.array(S)
print(S)

plt.plot(S[:, 0], S[:, 1], 'go')
plt.plot(0.06, 0.41, 'g*')
plt.xlabel("1st Dimension")
plt.ylabel("2nd Dimension")
plt.show()

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

S = np.around(S, decimals=2)
S1 = np.array(list(set([tuple(t) for t in S])))
# S1 = unique(S)
print(S1)
print(len(S1))
plt.plot(S1[:, 0], S1[:, 1], 'bo')
plt.plot(0.06, 0.41, 'g*')
plt.xlabel("1st Dimension")
plt.ylabel("2nd Dimension")

plt.show()