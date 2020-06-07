import numpy as np
import random

## Method of Moment estimation
def gammamixinit(x, lambda1 = None, alpha = None, beta = None, k = 2): 
    n = len(x)
    if type(lambda1) != np.ndarray:
        if lambda1 == None:
            lambda1 = np.zeros(shape=(k,1))
            for i in range(k):
                lambda1[i] = random.uniform(0,1)
            lambda1 = lambda1 / sum(lambda1)
        else:
            print("wrong format of lambda1")
            return 0
    elif type(lambda1) == np.ndarray:
        k = len(lambda1)
    else:
        print("wrong format of lambda1")
        return 0
    if k == 1:
        xbar = np.mean(x)
        x2bar = np.mean(x**2)
    else:
        xpart = []
        xbar = np.zeros(shape=(k,1))
        x2bar = np.zeros(shape=(k,1))
        xsort =  np.sort(x)
        ind = np.floor(n * np.cumsum(lambda1))
        xpart.append(xsort[0: int(ind[0])]) 
        xbar[0] = np.mean(xpart[0])
        x2bar[0] = np.mean(xpart[0]**2)
        for j in range(1,k):
            xpart.append(xsort[int(ind[j - 1]):int(ind[j])]) 
            xbar[j] = np.mean(xpart[j])
            x2bar[j] = np.mean(xpart[j]**2)
    if alpha == None:
        alpha = xbar**2 / (x2bar - xbar**2)
    if beta == None:
        beta = (x2bar -xbar**2) / xbar
    EstimatedPara = [lambda1, alpha, beta, k]
    print("estimated alpha:", alpha)
    print("estimtated beta:", beta)
    return EstimatedPara





x1 = np.random.gamma(0.2,14,1000)
x2 = np.random.gamma(32,10,1000)
x3 = np.random.gamma(5,6,1000)
x = np.hstack((x1, x2, x3))
lambda1 = np.array([1/3, 1/3, 1/3])
##Method of Moment
gammamixinit(x, lambda1, alpha = None, beta = None)
# estimated alpha: [[ 0.39265969]
#  [ 6.28847572]
#  [36.77140173]]
# estimtated beta: [[5.53862075]
#  [4.85861928]
#  [8.79808874]]

