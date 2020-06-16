
from numpy.random import gamma
import numpy as np

from MixedGamma import MixedGamma

class TestMixedGamma:

    def __init__(self):
        pass

    @staticmethod
    def testSimple():
        shape = [0.2, 32, 5]
        scale = [14, 10, 6]

        x = []

        for i in range(len(shape)):
            x.append(gamma(shape[i], scale[i], 200))

        x = np.asarray(x)
        x = x.reshape(-1)

        # print('xVec: ', x)
        # out <- gammamixEM(x, lambda = c(1, 1, 1) / 3, verb = TRUE, maxrestarts = 30)

        pai = np.ones(len(shape), dtype=np.float64)
        pai = pai/np.sum(pai)

        mg = MixedGamma(maxRestarts=30, verb=True)
        r = mg.estimateResult(x, pai= pai, k=3)
        # r = mg.estimateResult(x, k=3)

        r.display()
        # print('r.pai: ', r.pai)
        # print('r.shape: ', r.shape)
        # print('r.scale: ', r.scale)

        #################################################################################
        # gammamixEM.R results
        # # $lambda
        # #[1] 0.3245213 0.3421453 0.3333334
        # log-lik: -2538.996
        # all.loglik: -2539.123 -2538.997 -2538.996 -2538.996 -2538.996 -2538.996
        # # $gamma.pars
        # #         comp.1   comp.2   comp.3
        # #alpha 0.3571404   6.187210 26.80537
        # #beta  5.4571960   4.938647 12.13725
        # iteration = 1  log-lik diff = 0.1259981  log-lik = -2538.997
        # iteration = 2  log-lik diff = 0.001051349  log-lik = -2538.996
        # iteration = 3  log-lik diff = 8.653812e-06  log-lik = -2538.996
        # iteration = 4  log-lik diff = 7.114295e-08  log-lik = -2538.996
        # iteration = 5  log-lik diff = 5.848051e-10  log-lik = -2538.996
        # number of iterations= 5
        ###################################################################################

        # MixedGamma result:
        # iteration =  5 log-lik diff =  1.7280399333685637e-10 log-lik =  -2464.28394726437
        # number of iterations= 5
        # r.pai:  [0.32468025 0.34198641 0.33333334]
        # r.shape:  [ 0.3582566   5.79507659 32.09955357]
        # r.scale:  [5.02686678 5.18461491 9.85746174]

        #####################################################################################
        # terminate called after throwing an instance of 'std::overflow_error'
        #   what():  ERROR: non-finite value supplied by 'nlm'

if __name__ == '__main__':
    t = TestMixedGamma()
    t.testSimple()