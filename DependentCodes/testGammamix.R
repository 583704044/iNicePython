library("mixtools")
set.seed(100)
x <- c(rgamma(200, shape = 0.2, scale = 14), 
       rgamma(200, shape = 32, scale = 10), 
       rgamma(200, shape = 5, scale = 6))
out <- gammamixEM(x, lambda = c(1, 1, 1)/3, verb = TRUE)
print(out$lambda)
print(out$gamma.pars)
for ( p in out$gamma.pars) {
  print(p$alpha)
}
  

# mixtools 1.1.0 running results:
# $lambda
# [1] 0.3245213 0.3421453 0.3333334

# $gamma.pars
#       comp.1    comp.2     comp.3
# alpha 0.3571404 6.187210   26.80537
# beta  5.4571960 4.938647   12.13725
# mean: 

# mixtools 1.2.0 running result:
# iteration = 547  log-lik diff = 9.681571e-09  log-lik = -2511.448 
# number of iterations= 547 
# $lambda
# [1] 0.3215866 0.3450800 0.3333333
# $gamma.pars
#        comp.1   comp.2   comp.3
# alpha 0.2109625 5.144978 30.55456
# beta  8.9941068 5.766442 10.68708