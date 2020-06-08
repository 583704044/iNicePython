
# Adpated from mixtools1.1.0 by jianfei since 2020.6.8


import numpy as np
from scipy.stats import gamma

from nlmMinimizer import nlmMinimizer


class MixedGammaResult:

    def __init__(self):
        self.pai = None       # pai is another name of lambda, which is a keyword of python
        self.shape = None
        self.scale = None
        self.loglik = None
        self.posterior = None
        self.all_loglik = None

class MixedGamma:

        def __init__(self, epsilon=1e-08, maxIter=1000, maxRestarts=20, verb=False):
            self.setConfig(epsilon, maxIter, maxRestarts, verb)

            self.minimizer = nlmMinimizer()

        def setConfig(self, epsilon=1e-08, maxIter=1000, maxRestarts=20, verb=False):
            self.epsilon        = 1e-08
            self.maxIter        = 1000
            self.maxRestarts    = 20
            self.verb           = verb

        def _estMoment1_2(self):

            if self.k == 1:
                Ex = np.mean(self.xVec)          # compute mean of the vector x
                Ex2 = np.mean(self.xVec ** 2)    # compute squares of the elements of x

                return Ex, Ex2

            # k >= 2
            # cumsum(pai) is like 0.3333333 0.6666667 1.0000000
            # ind is like 200 400 600
            ind = np.floor(self.n * np.cumsum(self.pai))

            x_sort = np.sort(self.xVec)

            x_part = []
            x_part.append(x_sort[0:ind[0]+1])  # get [0, ind[0]] first ind[0]+1 elements (201 elements)
            for j in range(1, len(ind)):       # for each in [1, len(ind)-1]
                x_part.append(x_sort[ind[j-1]-1: ind[j]])  # [199, 399], [399, 599]

            Ex = np.zeros(shape=(self.k,))
            Ex2 = np.zeros(shape=(self.k,))
            for i, p in enumerate(x_part):
                Ex[i] = np.mean(p)            # 1.948985=mean[0, 200]  30.556443=mean[199,399] 325.343651
                Ex2[i] = np.mean(p ** 2)

            return Ex, Ex2

        def _estInitParameters(self):
            # gammamix.init <- function(x, lambda = NULL, alpha = NULL, beta = NULL, k = 2)

            #
            #  compute pai, k
            #
            if not self.pai:
                u = np.random.uniform(0, 1, self.k)  # R version: runif(k)
                self.pai = u/sum(u)         # normalized to 1  0.3333333 0.3333333 0.3333333
            else:
                self.k = len(self.pai)

            #
            #  compute shape, scale
            #
            Ex, Ex2 = self._estMoment1_2()
            Ex_2 = Ex ** 2
            Varx = Ex2 - Ex_2

            if not self.shape:
                self.shape = Ex ** 2/Varx     # 0.3571404  6.1872099 26.8053743

            if not self.scale:
                self.scale = Varx/Ex          # 5.457196  4.938647 12.137255

        def _update_pdfBuffer(self):
            # dens <- function(lambda, theta, k)

            # x: quantiles
            # a as a shape parameter for gamma dist.
            # gamma.pdf(x, a, loc, scale) is identically equivalent to
            # gamma.pdf(y, a) / scale with y = (x - loc) / scale.
            # return pdf : ndarray: Pobability density function evaluated at x
            for i in range(self.k):
                self.pdfBuffer[:, i] = gamma.pdf(self.xVec,
                                             a=self.shape[i],
                                             scale=self.scale[i])

            self.pdfBuffer *= self.pai   # weighted prob. density values: lambda * pdfBuffer

        def _sumLogLik(self):
            # return a scalar
            # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
            self._update_pdfBuffer()
            np.sum(self.pdfBuffer, axis=1, out=self.nVecBuffer)
            np.log(self.nVecBuffer, out=self.nVecBuffer)

            return np.sum(self.nVecBuffer)

        def _lossFun(self, x):
            # gamma.ll < - function(theta, z, lambda , k) - sum(z * log(dens(lambda , theta, k)))
            # x is theta,
            shape = x[0: self.k]
            scale = x[self.k: self.k + self.k]

            for i in range(self.k):
                self.pdfBuffer[:, i] = gamma.pdf(self.xVec,
                                             a=shape[i],
                                             scale=scale[i])
            self.pdfBuffer *= self.pai  # weighted prob. density values: lambda * pdfBuffer

            np.log(self.pdfBuffer, out=self.pdfBuffer)
            self.pdfBuffer *= self.zBuffer

            return -1.0 * np.sum(self.pdfBuffer)

        def _update_pai_z(self):

            #
            # compute new pai, i.e., pai_hat and z
            # from old self.pai, self.shape, self.scale
            #
            for i in range(self.k):
                self.zBuffer[:, i] = gamma.pdf(self.xVec,
                                             a=self.shape[i],
                                             scale=self.scale[i])

            self.zBuffer *= self.pai   # weighted prob. density values: lambda * pdfBuffer

            weight = self.zBuffer.sum(axis=1)
            self.zBuffer /= weight[:, np.newaxis]     # z is 600-by-3

            self.pai = np.mean(self.zBuffer, axis=0)  # pai_hat is 1-by-3

        def estimateResult(self, xVec: np.ndarray, pai=None, shape=None, scale=None, k=2):

            self.xVec   = xVec          # 1-dimension data supposed to follow a mixture of k gamma distributions
            self.n      = self.xVec.shape[0]
            self.k      = k             # number of clusters

            self.pai    = pai           # lambda parameter of gamma distribution
            self.shape  = shape
            self.scale  = scale

            self.pdfBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)  # pdfBuffer is n-by-k
            self.zBuffer = np.zeros(shape=(self.n, self.k), dtype=np.float64)  # zBuffer for computing z
            self.nVecBuffer = np.zeros(shape=(self.n, ), dtype=np.float64)

            # initize (lambda = lambda, alpha = alpha, beta = beta, k = k)
            self._estInitParameters()

            iter = 0
            mr = 0
            diff = self.epsilon + 1

            # old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
            old_obs_ll = self._sumLogLik()
            ll = old_obs_ll

            x = np.concatenate((self.shape, self.scale))
            while diff > self.epsilon and iter < self.maxIter:

                # update z and pai
                self._update_pai_z()

                ok, retValue = self.minimizer(x, self._lossFun)
                # out = try(suppressWarnings(nlm(gamma.ll, p = theta, lambda = lambda.hat, k = k, z = z)),

                if ok:
                    x = self.minimizer.getXSolution()

                #         theta.hat = out$estimate
                #         alpha.hat = theta.hat[1:k]
                #         beta.hat = theta.hat[(k + 1):(2 * k)]
                #         # Debug> theta.hat
                #         #[1]  0.3571404  6.1872099 26.8053743  5.4571960  4.9386467 12.1372546
                #         new.obs.ll <- sum(log(apply(dens(lambda.hat, theta.hat, k), 1, sum)))
                #         diff <- new.obs.ll - old.obs.ll
                #         old.obs.ll <- new.obs.ll
                #         ll <- c(ll, old.obs.ll)
                #         # Debug> ll
                #         #[1] -2539.123 -2538.997
                #         lambda = lambda.hat
                #         theta = theta.hat
                #         alpha = alpha.hat
                #         beta = beta.hat
                #         iter = iter + 1
                #         if (verb) {
                #           cat("iteration =", iter, " log-lik diff =", diff, " log-lik =",
                #               new.obs.ll, "\n")
                #         }