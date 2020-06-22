
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

        np.random.seed(12312)

        x = []

        for i in range(len(shape)):
            x.append(gamma(shape[i], scale[i], 5000))

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

        # ================> DEB...end...np.log(posiInfi)... [[ 2.45745000e+00 -4.70611022e+01 -3.93040352e+02]
        #  [ 2.21408499e-01 -3.21577147e+01 -2.83667524e+02]
        #  [-4.73043317e+00 -7.36252329e+00 -9.55205602e+01]
        #  ...
        #  [-9.62901535e+00 -4.51349337e+00 -5.21770372e+01]
        #  [-1.21658620e+01 -4.93114110e+00 -4.21983227e+01]
        #  [-1.29008913e+01 -5.14489976e+00 -3.99179155e+01]]
        # log(pdfBuff).inf:  [[ 48   2]    --- inifinity max
        #  [206   2]
        #  [208   2]
        #  [301   2]
        #  [495   2]]
        # zBuff.nan:  []
        # ==+=+===+===+==> DEB...end...pdfBuff*zBuff... [[ 2.45745000e+000 -1.36837460e-020 -6.54996964e-170]
        #  [ 2.21408499e-001 -2.59649528e-013 -1.39871016e-121]
        #  [-4.43339574e+000 -4.62313885e-001 -3.21067475e-038]
        #  ...
        #  [-6.16555615e-002 -4.48459301e+000 -1.07105977e-019]
        #  [-9.41088706e-003 -4.92732662e+000 -2.85210770e-015]
        #  [-5.92732743e-003 -5.14253593e+000 -3.26871870e-014]]
        # (pdfB * zBuff).nan:  [[ 48   2]   ---- NaN = -inf * any ?
        #  [206   2]
        #  [208   2]
        #  [301   2]
        #  [495   2]]
        # 				*********** loss:  nan

if __name__ == '__main__':
    t = TestMixedGamma()
    t.testSimple()