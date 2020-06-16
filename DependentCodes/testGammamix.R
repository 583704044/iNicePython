library("mixtools")
# set.seed(100)
#x <- c(rgamma(200, shape = 0.2, scale = 14), 
#       rgamma(200, shape = 32, scale = 10), 
#       rgamma(200, shape = 5, scale = 6))

# out <- gammamixEM(x, lambda = c(1, 1, 1)/3, verb = TRUE)
# out <- gammamixEM(x, k=3, verb = TRUE)

print(warnings())

cat("..............................")
print(out$lambda)
print(out$gamma.pars)
for ( p in out$gamma.pars) {
  print(p)
}

# mixtools 1.1.0 initial guess for alpha and beta are
# Debug> alpha
# [1]  0.3571404  6.1872099 26.8053743
# Debug> beta
# [1]  5.457196  4.938647 12.137255

# mixtools 1.1.0 running results:
# iteration = 5  log-lik diff = 5.848051e-10  log-lik = -2538.996 
# number of iterations= 5 
# $lambda
# [1] 0.3245213 0.3421453 0.3333334

# $gamma.pars
#       comp.1    comp.2     comp.3
# alpha 0.3571404 6.187210   26.80537
# beta  5.4571960 4.938647   12.13725
# mean: 1.95      30.56      325.34

# mixtools 1.2.0 running result:
# iteration = 547  log-lik diff = 9.681571e-09  log-lik = -2511.448 
# number of iterations= 547 
# $lambda
# [1] 0.3215866 0.3450800 0.3333333
# $gamma.pars
#        comp.1   comp.2   comp.3
# alpha 0.2109625 5.144978 30.55456
# beta  8.9941068 5.766442 10.68708