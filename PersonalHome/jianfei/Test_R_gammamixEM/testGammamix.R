library("mixtools")
set.seed(100)

# rgamma(n, shape, rate = 1, scale = 1/rate)
x <- c(rgamma(200, shape = 0.2, scale = 14),
       rgamma(200, shape = 32, scale = 10),
       rgamma(200, shape = 5, scale = 6))
cat('---------------- x is dim=', dim(x), 'nrow=', NROW(x), 'ncol=', NCOL(x), "\n")
# dim=  nrow=600, ncol=1

# function(x, lambda = NULL, alpha = NULL, beta = NULL, k = 2,
# epsilon = 1e-08, maxit = 1000, maxrestarts = 20, verb = FALSE) {
out <- gammamixEM(x, lambda = c(1, 1, 1) / 3, verb = TRUE, maxrestarts = 30)
cat('---------------- x...lambda,gamma.pars' )
print(out$lambda)
print(out$gamma.pars)
print(out$loglik)
print(out$posterior)
print(out$all.loglik)
# a = list(x = x, lambda = lambda, gamma.pars = theta,
# loglik = new.obs.ll, posterior = z,
# all.loglik = ll, ft = "gammamixEM")

# $lambda
#[1] 0.3245213 0.3421453 0.3333334

# $gamma.pars
#         comp.1   comp.2   comp.3
#alpha 0.3571404   6.187210 26.80537
#beta  5.4571960   4.938647 12.13725
