from numpy.random import gamma
import numpy as np
import pandas as pd

def generateData(shape, scale, file_name):
    x = []
    np.random.seed(123)
    for i in range(len(shape)):
        x.append(gamma(shape[i], scale[i], 200))
    x = np.asarray(x)

    x = x.reshape(-1)

    print('x.reshape(-1)', x)

    x = pd.DataFrame(x)
    x.to_csv(file_name)
    print(x)
    return x

def readFile(fileName):

    df = pd.read_csv(fileName)
    x = df.iloc[:, 1:2].to_numpy()
    x = x.reshape(-1)
    print(x)

def SetTrueValue1():
    shape1 = [0.2, 32, 5]
    scale1 = [14, 10, 6]
    data1 = generateData(shape1, scale1, "GammaData11.csv")
    return data1

def SetTrueValue2():
    shape2 = [2, 32, 5, 24, 9, 40, 44, 3, 7, 69,
        7, 3, 50, 44, 50, 35, 79, 11, 34, 28,
        13, 20, 27, 70, 90, 28, 36, 35, 42, 92,
        34, 108, 32, 115, 73, 80, 150, 70, 200, 142,
        160, 117, 120, 105, 150, 130, 163, 142, 174, 138]
    scale2 = [14, 10, 6, 60, 20, 5, 53, 40, 34, 69,
        8, 9, 43, 40, 24, 56, 21, 42, 24, 64,
        12, 32, 49, 81, 19, 53, 80, 43, 17, 43,
        82, 49, 109, 180, 100, 81, 150, 98, 63, 122,
        156, 133, 80, 105, 115, 130, 121, 144, 190, 125]
    data2 = generateData(shape2, scale2, "GammaData2.csv")
    return data2

def SetTrueValue3():
    shape3 = [7, 12, 4, 15, 20, 25, 29, 39, 35, 24,
        43, 49, 52, 42, 45, 51, 32, 62, 33, 46,
        12, 58, 53, 35, 60, 67, 65, 69, 85, 40,
        39, 71, 83, 29, 24, 81, 72, 93, 64, 43,
        92, 100, 105, 107, 130, 111, 130, 132, 142, 124,
        127, 152, 157, 112, 160, 198, 173, 112, 154, 172,
        143, 149, 144, 152, 132, 120, 183, 177, 180, 163,
        173, 190, 192, 212, 221, 133, 254, 243, 184, 145,
        210, 214, 264, 221, 242, 200, 213, 215, 223, 249,
        201, 24, 84, 95, 47, 152, 23, 42, 21, 14]
    scale3 = [47, 18, 5, 35, 26, 25, 46, 32, 14, 24,
        64, 12, 52, 24, 32, 36, 23, 72, 64, 31,
        200, 120, 43, 54, 94, 12, 45, 85, 27, 23,
        53, 70, 23, 34, 43, 60, 23, 65, 41, 94,
        54, 33, 42, 51, 22, 12, 42, 33, 42, 12,
        40, 91, 44, 85, 12, 40, 143, 149, 144, 152,
        25, 14, 22, 40, 171, 32, 12, 21, 141, 124,
        24, 64, 15, 94, 40, 44, 33, 27, 69, 164,
        121, 102, 224, 67, 0.4, 32, 0.9, 73, 123, 98,
        542, 121, 123, 54, 139, 125, 134, 124, 18, 19]
    data3 = generateData(shape3, scale3, "GammaData3.csv")
    return data3


# SetTrueValue1()
#SetTrueValue2()
#SetTrueValue3()
readFile('./GammaData1.csv')
