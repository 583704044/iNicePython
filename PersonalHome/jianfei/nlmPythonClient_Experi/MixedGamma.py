# Adpated from mixtools1.1.0 by jianfei since 2020.6.8

import sys
import numpy as np
from scipy.stats import gamma

from nlmMinimizer import nlmMinimizer


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

        assert (Varx > 0).all(), 'Varx must > 0'

        if self.shape is None:
            self.shape = Ex ** 2 / Varx  # 0.3571404  6.1872099 26.8053743

        if self.scale is None:
            self.scale = Varx / Ex  # 5.457196  4.938647 12.137255

        self.xSS = np.concatenate((self.shape, self.scale))

        # update self.pdfBuffer
        self._positiveInference(self.pai, self.shape, self.scale, self.pdfBuffer)

    def _positiveInference(self, pai, shape, scale, outBuffer):
        # dens <- function(lambda, theta, k)
        # outBuffer[i, j]= p(x_i | z_i = j) p(z_i = j)
        # = gamma.pdf(x_i, shape_j, scale_j) pai_j
        for j in range(self.k):
            outBuffer[:, j] = gamma.pdf(self.xVec, a=shape[j], scale=scale[j])  # likelihood
            print('outBuff.zeros: ', outBuffer[outBuffer==0])

        # remove zeros
        # outBuffer[outBuffer< sys.float_info.min] = sys.float_info.min

        outBuffer *= pai  # product rule of probability, i.e. un-normalized posterior probability
        print('pai * outBuff.zeros: ', np.argwhere(outBuffer == 0))

    def _update_zBuffer_pai(self):
        # zBuffer: n-by-k, the latent membership prob for each i-th data, i in [0,n)

        print('_zBuffer_pai.zBuff.nan: ', np.argwhere(np.isnan(self.pdfBuffer)))
        self._positiveInference(self.pai, self.shape, self.scale, self.zBuffer)

        print('_zBuffer_pai.posInf..zBuff.nan: ', np.argwhere(np.isnan(self.pdfBuffer)))

        # normalize z_Buffer
        self.zBuffer.sum(axis=1, out=self.nVecBuffer)
        self.zBuffer /= self.nVecBuffer[:, np.newaxis]  # z is 600-by-3

        print('_zBuffer_pai.divide/sum..zBuff.nan: ', np.argwhere(np.isnan(self.pdfBuffer)))

        # pai_hat_j = 1/n sum_i..n p(Z_i =j| x_i, theta)
        self.pai = np.mean(self.zBuffer, axis=0)        # pai_hat is 1-by-3

    def _update_shape_scale(self, xSS):

        if np.isfinite(xSS).all() and (xSS > 0).all():
            self.shape = xSS[0: self.k]
            self.scale = xSS[self.k: self.k + self.k]
            # DONT SET self.ssErrorFree = True,
            # because the nlm_simple may return valid solution,
            # even it accesses bad solution during the single solving process.
            self.xSS = xSS
            return True
        return False


    def _lossFun(self, xSS):
        # gamma.ll < - function(theta, z, lambda , k) - sum(z * log(dens(lambda , theta, k)))
        # x is theta=(shape, scale)

        print('\t\t\t\t\tDEB--------------> xSS=', xSS)

        if not (np.isfinite(xSS).all() and (xSS > 0).all()):
            print('MixedGamma..WARNING..found invalid xSS: ', xSS)
            return sys.float_info.max

        shape = xSS[0: self.k]
        scale = xSS[self.k: self.k + self.k]

        print('DEB--------------> posiInfi...')
        # given new zBuffer=> new pai,
        # compute loss for the searching point xSS
        self._positiveInference(self.pai, shape, scale, self.pdfBuffer)

        print('pdfBuff.zeros: ', np.argwhere(self.pdfBuffer==0))
        print('pdfBuff.nan: ', np.argwhere(np.isnan(self.pdfBuffer)))

        print('DEB...<------------- posiInfi...')

        np.log(self.pdfBuffer, out=self.pdfBuffer)

        print('================> DEB...end...np.log(posiInfi)...', self.pdfBuffer)
        nanw = np.argwhere(np.isinf(self.pdfBuffer))
        print('log(pdfBuff).inf_ind: ', nanw)
        if nanw.shape[0] != 0:
            print('log(pdfBuff).inf_ele: ', self.pdfBuffer[nanw[0,0], nanw[0,1]])
            print('xVec.ele: ', self.xVec[nanw[0,0]])
        print('zBuff.nan: ', np.argwhere(np.isnan(self.zBuffer)))

        self.pdfBuffer *= self.zBuffer

        print('==+=+===+===+==> DEB...end...pdfBuff*zBuff...', self.pdfBuffer)
        print('(pdfB * zBuff).nan: ', np.argwhere(np.isnan(self.pdfBuffer)))

        res = -1.0 * np.sum(self.pdfBuffer)

        print('\t\t\t\t*********** loss: ', res)

        return res

    def _sumLogLik(self):
        # return a scalar
        # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))

        # already calcuated in self._lossFun, self._estInitParameters
        self._positiveInference(self.pai, self.shape, self.scale, self.pdfBuffer)

        np.sum(self.pdfBuffer, axis=1, out=self.nVecBuffer)
        np.log(self.nVecBuffer, out=self.nVecBuffer)

        return np.sum(self.nVecBuffer)

    def _checkInitCall(self, xVec, pai, shape, scale, k):
        assert (xVec > 0).all() and np.isfinite(xVec).all()
        assert pai is None or (pai > 0).all()
        assert shape is None or (shape > 0).all()
        assert scale is None or (scale > 0).all()
        assert k > 0


    def estimateResult(self, xVec: np.ndarray, pai=None, shape=None, scale=None, k=2):

        self._checkInitCall(xVec, pai, shape, scale, k)

        self.xVec = xVec  # 1-dimension data supposed to follow a mixture of k gamma distributions
        self.n = self.xVec.shape[0]
        self.k = k  # number of clusters

        self.pai = pai  # lambda parameter of gamma distribution
        self.shape = shape
        self.scale = scale
        self.xSS = None   # np.concatenate((self.shape, self.scale))

        self.pdfBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)  # pdfBuffer is n-by-k
        self.zBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)    # zBuffer for computing z
        self.nVecBuffer = np.zeros(shape=(self.n,), dtype=np.float64)        # used in computing z, sumLogLik

        # initize (lambda = lambda, alpha = alpha, beta = beta, k = k)
        self._estInitParameters()

        print('after...estInitParameters...')

        iter = 0
        mr = 0                      # number of max restarts
        diff = self.epsilon + 1

        # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
        old_obs_ll = self._sumLogLik()

        print('DEB...old_obs_ll: ', old_obs_ll)

        ll = [old_obs_ll]
        new_obs_ll = None

        while diff > self.epsilon and iter < self.maxIter:

            # update membership (z and pai(hat)) to obtain a new self._lossFun
            # self.pai is pai_hat
            self._update_zBuffer_pai()

            print('DEB...after _update_pai_z')

            retValue = self.minimizer(self.xSS, self._lossFun)
            # out = try(suppressWarnings(nlm(gamma.ll, p = theta~xSS, lambda = lambda.hat, k = k, z = z)),

            print('DEB...after self.minimizer...retValue: ', retValue)
            xSS = self.minimizer.getXSolution()
            xOK = self._update_shape_scale(xSS)

            if xOK and retValue is not None:

                new_obs_ll = self._sumLogLik()  # self.pdfBuffer is right
                diff = new_obs_ll - old_obs_ll

                old_obs_ll = new_obs_ll
                ll.append(new_obs_ll)           # add new likelihood into list

                iter += 1
                if self.verb:
                    print('iteration = ', iter, 'log-lik diff = ', diff,
                          'log-lik = ', new_obs_ll)
                continue

            # !xOK or !self.ssErrorFree or retValue None
            print('Note: Choosing new starting values.')
            if mr >= self.maxRestarts:
                print('FAILED: maxRestarts=', mr, ' is reached')
                return None
            mr += 1
            iter = 0
            diff = self.epsilon + 1

            self._estInitParameters()
            old_obs_ll = self._sumLogLik()  # must be called after _estInitParameters
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
