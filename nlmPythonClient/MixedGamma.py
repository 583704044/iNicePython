# Adpated from mixtools1.1.0 by jianfei since 2020.6.8

import numpy as np
from scipy.stats import gamma
from nlmMinimizer import nlmMinimizer

import sys

class MixedGammaResult:

    def __init__(self):
        self.pai = None  # pai is another name of lambda, which is a keyword of python
        self.shape = None
        self.scale = None
        self.loglik = None
        self.posterior = None
        self.all_loglik = None

    def display(self):
        print('pai: ', self.pai)
        print('shape: ', self.shape)
        print('scale: ', self.scale)
        print('the last loglik: ', self.loglik)
        print('all logliks: ', self.all_loglik)
        print('the last posterior z: ', self.posterior)


class MixedGamma:

    def __init__(self, epsilon=1e-08, maxIter=1000, maxRestarts=20, verb=False):
        self.setConfig(epsilon, maxIter, maxRestarts, verb)

        self.minimizer = nlmMinimizer()

    def setConfig(self, epsilon=1e-08, maxIter=1000, maxRestarts=20, verb=False):
        self.epsilon = 1e-08
        self.maxIter = 1000
        self.maxRestarts = 20
        self.verb = verb

    def _estMoment1_2(self):
        if self.k == 1:
            Ex = np.mean(self.xVec)  # compute mean of the vector x
            Ex2 = np.mean(self.xVec ** 2)  # compute squares of the elements of x
            return Ex, Ex2

        # k >= 2
        # cumsum(pai) is like 0.3333333 0.6666667 1.0000000
        # ind is like 200 400 600
        ind = np.floor(self.n * np.cumsum(self.pai))
        ind = ind.astype(int)
        x_sort = np.sort(self.xVec)

        print('DEB: ind: ', ind)
        # print('x_sort: ', x_sort)
        # print('x_sort[0:ind[0] + 1]: ', x_sort[0:ind[0] + 1])

        x_part = []
        x_part.append(x_sort[0:ind[0] + 1])  # get [0, ind[0]] first ind[0]+1 elements (201 elements)
        for j in range(1, len(ind)):  # for each in [1, len(ind)-1]
            x_part.append(x_sort[ind[j - 1] - 1: ind[j]])  # [199, 399], [399, 599]

        Ex = np.zeros(shape=(self.k,))
        Ex2 = np.zeros(shape=(self.k,))
        for i, p in enumerate(x_part):
            # print('DEB:...x_part_i=', i, p)
            Ex[i] = np.mean(p)  # 1.948985=mean[0, 200]  30.556443=mean[199,399] 325.343651
            Ex2[i] = np.mean(p ** 2)

        print('DEB...before return Ex, Ex2: ', Ex, Ex2)
        return Ex, Ex2

    def _estInitParameters(self):
        # gammamix.init <- function(x, lambda = NULL, alpha = NULL, beta = NULL, k = 2)

        #  compute pai, k
        if self.pai is None:
            # [0.48496047 0.05261586 0.27536281]
            u = np.random.uniform(0, 1, self.k)  # R version: runif(k)
            self.pai = u / sum(u)  # normalized to 1  0.3333333 0.3333333 0.3333333
        else:
            self.k = len(self.pai)

        #  compute shape, scale
        Ex, Ex2 = self._estMoment1_2()
        Ex_2 = Ex ** 2
        Varx = Ex2 - Ex_2

        if not self.shape:
            self.shape = Ex ** 2 / Varx  # 0.3571404  6.1872099 26.8053743

        if not self.scale:
            self.scale = Varx / Ex  # 5.457196  4.938647 12.137255

        self.xSS = np.concatenate((self.shape, self.scale))

    def _update_pdfBuffer(self, pdfBuffer, pai, shape, scale):
        # dens <- function(lambda, theta, k)

        for i in range(self.k):
            pdfBuffer[:, i] = gamma.pdf(self.xVec,
                                             a=shape[i],
                                             scale=scale[i])
            # print('xVec: ', self.xVec)
            # print('shape: ', shape[i])
            # print('scale: ', scale[i])
            # print('gamma.pdf: i=', i, pdfBuffer[:, i])

        pdfBuffer *= pai  # weighted prob. density values: lambda * pdfBuffer

        # print('DEB..._update_pdfBuffer...', pdfBuffer)

    def _sumLogLik(self, pai, shape, scale):
        # return a scalar
        # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
        self._update_pdfBuffer(self.pdfBuffer, pai, shape, scale)
        np.sum(self.pdfBuffer, axis=1, out=self.nVecBuffer)
        np.log(self.nVecBuffer, out=self.nVecBuffer)

        return np.sum(self.nVecBuffer)

    def _lossFun(self, xSS):
        # gamma.ll < - function(theta, z, lambda , k) - sum(z * log(dens(lambda , theta, k)))
        # x is theta,

        print('\t\t\t\t\tDEB--------------> xSS=', xSS)

        if not (np.isfinite(xSS).all() and (xSS > 0).all()):
            print('MixedGamma..WARNING..found invalid xSS: ', xSS)
            return sys.float_info.max

        shape = xSS[0: self.k]
        scale = xSS[self.k: self.k + self.k]

        print('DEB--------------> posiInfi...')
        # for new pai, z, searching xSS
        self._update_pdfBuffer(self.pdfBuffer, self.pai, shape, scale)

        print('pdfBuff.zeros_inds: ', np.argwhere(self.pdfBuffer==0))
        print('pdfBuff.nan: ', np.argwhere(np.isnan(self.pdfBuffer)))
        print('DEB...<------------- posiInfi...')

        # print('\t\t\t_lossFun..np.log..')
        np.log(self.pdfBuffer, out=self.pdfBuffer)

        print('================> DEB...end...np.log(posiInfi)...')
        nanw = np.argwhere(np.isinf(self.pdfBuffer))    # pdfBuffer: n-by-k
        # print('log(pdfBuff).inf_ind: ', nanw)
        if nanw.shape[0] != 0:
            print('log(pdfBuff).nan inds: ', nanw)
            print('log(pdfBuff)[0,0].inf_ele: ', self.pdfBuffer[nanw[0,0], nanw[0,1]])
            print('xVec.ele: ', self.xVec[nanw[0,0]])
        print('zBuff.nan: ', np.argwhere(np.isnan(self.zBuffer)))

        self.pdfBuffer *= self.zBuffer

        print('==+=+===+===+==> DEB...end...pdfBuff*zBuff...')
        print('(pdfB * zBuff).nan_inds: ', np.argwhere(np.isnan(self.pdfBuffer)))

        res = -1.0 * np.sum(self.pdfBuffer)

        print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t*********** loss: ', res)

        return res

    def _update_pai_z(self):

        self._update_pdfBuffer(self.zBuffer, self.pai, self.shape, self.scale)

        weight = self.zBuffer.sum(axis=1)

        self.zBuffer /= weight[:, np.newaxis]       # z is 600-by-3
        self.pai = np.mean(self.zBuffer, axis=0)    # pai_hat is 1-by-3

    def estimateResult(self, xVec: np.ndarray, pai=None, shape=None, scale=None, k=2):

        self.xVec = xVec  # 1-dimension data supposed to follow a mixture of k gamma distributions
        self.n = self.xVec.shape[0]
        self.k = k  # number of clusters

        self.pai = pai  # lambda parameter of gamma distribution
        self.shape = shape
        self.scale = scale
        self.xSS = None   # np.concatenate((self.shape, self.scale))

        self.pdfBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)  # pdfBuffer is n-by-k
        self.zBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)  # zBuffer for computing z
        self.nVecBuffer = np.zeros(shape=(self.n,), dtype=np.float64)

        # initize (lambda = lambda, alpha = alpha, beta = beta, k = k)
        self._estInitParameters()

        print('after...estInitParameters...')

        iter = 0
        mr = 0
        diff = self.epsilon + 1

        # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
        old_obs_ll = self._sumLogLik(self.pai, self.shape, self.scale)

        print('DEB...old_obs_ll: ', old_obs_ll)

        ll = [old_obs_ll]
        new_obs_ll = None

        while diff > self.epsilon and iter < self.maxIter:

            # update z and pai(hat) to obtain a new self._lossFun
            # self.pai is pai_hat
            self._update_pai_z()

            print('DEB...after _update_pai_z')

            retValue = self.minimizer(self.xSS, self._lossFun)
            # out = try(suppressWarnings(nlm(gamma.ll, p = theta~xSS, lambda = lambda.hat, k = k, z = z)),

            print('DEB...after self.minimizer...retValue: ', retValue)

            if retValue is not None:
                self.xSS = self.minimizer.getXSolution()
                self.shape = self.xSS[0: self.k]
                self.scale = self.xSS[self.k: self.k + self.k]

                new_obs_ll = self._sumLogLik(self.pai, self.shape, self.scale)
                diff = new_obs_ll - old_obs_ll

                old_obs_ll = new_obs_ll
                ll.append(new_obs_ll)           # add new likelihood into list

                iter += 1
                if self.verb:
                    print('iteration = ', iter, 'log-lik diff = ', diff,
                          'log-lik = ', new_obs_ll)
                continue

            # retValue is None:
            print('Note: Choosing new starting values.')
            if mr >= self.maxRestarts:
                print('FAILED: maxRestarts=', mr, ' is reached')
                return None
            mr += 1
            iter = 0
            diff = self.epsilon + 1

            self._estInitParameters()
            old_obs_ll = self._sumLogLik(self.pai, self.shape, self.scale)
            ll = [old_obs_ll]

        # end of while
        if (iter >= self.maxIter):
            print('WARNING! NOT CONVERGENT!')
        print('number of iterations=', iter)

        # return result
        r = MixedGammaResult()
        r.pai = self.pai
        r.shape = self.shape
        r.scale = self.scale
        r.loglik = new_obs_ll
        r.posterior = self.zBuffer
        r.all_loglik = ll
        return r
