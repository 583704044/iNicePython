gammamixEM <-
  function(x, lambda = NULL, alpha = NULL, beta = NULL, k = 2,
           epsilon = 1e-08, maxit = 1000, maxrestarts = 20, verb = FALSE) {
    x <- as.vector(x)
    tmp <- gammamix.init(x = x, lambda = lambda, alpha = alpha, beta = beta, k = k)
    lambda <- tmp$lambda
    # Debug> lambda
    #[1] 0.3333333 0.3333333 0.3333333
    alpha <- tmp$alpha
    beta <- tmp$beta
    # Debug> alpha
    #[1]  0.3571404  6.1872099 26.8053743
    #Debug> beta
    #[1]  5.457196  4.938647 12.137255
    theta <- c(alpha, beta)

    k <- tmp$k
    iter <- 0
    mr <- 0
    diff <- epsilon + 1
    n <- length(x)

	dens <- NULL
    dens <- function(lambda, theta, k) {
      temp <- NULL
      alpha = theta[1:k]
      beta = theta[(k + 1):(2 * k)]
      for (j in 1:k) {
        temp = cbind(temp,
					 dgamma(x, shape = alpha[j], scale = beta[j])
					)
        # dgamma gives the density,
        # dgamma is computed via the Poisson density,
        # using code contributed by Catherine Loader (see dbinom).
        # list("dgamma", <pointer: 0x55c6a1b64ba0>,
        # list("stats", "/usr/lib/R/library/stats/libs/stats.so",
        # FALSE, <pointer: 0x55c6a2130aa0>, <pointer: 0x55c6a184e350>),
        # 4L)
        # Rmath.h
        # # define dgamma		Rf_dgamma
        # https://github.com/coatless/Rmath/blob/master/dgamma.c
        # https://cran.r-project.org/doc/manuals/r-devel/R-admin.html#The-standalone-Rmath-library
	  }
      # Debug> temp
      #               [,1]         [,2]          [,3]
      #  [1,] 1.395551e+00 9.031276e-14  1.349256e-88
      #  [2,] 1.782873e-01 2.866671e-07  7.320547e-56
      # ...
      #  [599,] 3.494986e-07 3.225129e-03  1.345641e-12
      #  [600,] 4.207091e-06 1.189496e-02  6.914517e-15
      # dgamma(x, shape, rate), x is vector of quantiles (sample distribution)
      # > dgamma(4:1, shape=1)
      #[1] 0.01831564 0.04978707 0.13533528 0.36787944  (
      #> dgamma(c(1,3,2,4), shape=1)
      #[1] 0.36787944 0.04978707 0.13533528 0.01831564
      temp = t(lambda * t(temp))  # same as lambda * temp
      # t(temp) is like:
      #              [,1]         [,2]         [,3]         [,4]         [,5]          [,6]         [,7]
      #[1,] 1.395551e+00 1.782873e-01 1.705495e-02 1.861821e-01 1.020244e-01  5.229065e+00 8.371382e-01
      #[2,] 9.031276e-14 2.866671e-07 1.873350e-03 2.207121e-07 5.923245e-06  2.300288e-18 5.004878e-12
      #[3,] 1.349256e-88 7.320547e-56 1.864623e-34 1.899238e-56 5.753433e-49 1.806011e-111 6.766864e-80
      # t(lambda * t(temp))
      #                [,1]         [,2]          [,3]
      #  [1,] 4.651836e-01 3.010425e-14  4.497519e-89
      #  [2,] 5.942909e-02 9.555570e-08  2.440182e-56
      #  [3,] 5.684984e-03 6.244499e-04  6.215409e-35
      #  [4,] 6.206071e-02 7.357068e-08  6.330794e-57
      #  [5,] 3.400813e-02 1.974415e-06  1.917811e-49
      #  [6,] 1.743022e+00 7.667625e-19 6.020036e-112
      #  [7,] 2.790461e-01 1.668293e-12  2.255621e-80
      # Debug> 4.651836e-01/1.395551e+00
      #[1] 0.3333333
      # Debug> 5.942909e-02/1.782873e-01
      #[1] 0.3333333
      # Debug> 2.255621e-80/6.766864e-80
      # [1] 0.3333333
      # Debug> 5.684984e-03/1.705495e-02
      #[1] 0.3333334

      # Debug> lambda * temp
      #               [,1]         [,2]          [,3]
      #  [1,] 4.651836e-01 3.010425e-14  4.497519e-89
      #  [2,] 5.942909e-02 9.555570e-08  2.440182e-56
      #  [3,] 5.684984e-03 6.244499e-04  6.215409e-35
      #  [4,] 6.206071e-02 7.357068e-08  6.330794e-57
      #  [5,] 3.400813e-02 1.974415e-06  1.917811e-49

      temp
    }
    old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
    # Debug> apply(dens(lambda, theta, k), 1, sum) for each row, sum 3 elements, the result is a vector with 600 rows
    #  [1] 4.651836e-01 5.942919e-02 6.309434e-03 6.206078e-02 3.401010e-02 1.743022e+00 2.790461e-01
    # > sum(c(4.651836e-01,3.010425e-14,4.497519e-89))
    #[1] 0.4651836
    # > sum(c(5.942909e-02,9.555570e-08,2.440182e-56))
    #[1] 0.05942919
    # Debug> log(apply(dens(lambda, theta, k), 1, sum))
    #  [1]  -0.76532310  -2.82296979  -5.06570930  -2.77964099  -3.38109763   0.55562024  -1.27637842
    # Debug> log(4.651836e-01)
    #[1] -0.7653231

    # Debug> sum(log(apply(dens(lambda, theta, k), 1, sum)))
    #[1] -2539.123
    # old.obs.ll [1] -2539.123
    ll <- old.obs.ll
    gamma.ll <- function(theta, z, lambda, k) -sum(z * log(dens(lambda, theta, k)))
    # Debug> log(dens(lambda, theta, k))
    #                [,1]        [,2]        [,3]
    #  [1,]   -0.76532310  -31.134110 -203.426547
    #  [2,]   -2.82297140  -16.163556 -128.052692
    # Debug> gamma.ll
    # function (theta, z, lambda, k)
    # -sum(z * log(dens(lambda, theta, k)))
    # <environment: 0x10c43bd0>

    while (diff > epsilon && iter < maxit) {
      # initialization diff <- epsilon + 1
      # epsilon = 1e-08

      dens1 = dens(lambda, theta, k)
      # Debug> dens1 --- first loop
      #               [,1]         [,2]          [,3]
      #  [1,] 4.651836e-01 3.010425e-14  4.497519e-89
      #  [2,] 5.942909e-02 9.555570e-08  2.440182e-56
      #  [3,] 5.684984e-03 6.244499e-04  6.215409e-35
      # ...
      #  [598,] 4.255243e-04 8.678647e-03  7.057811e-25
      #  [599,] 1.164995e-07 1.075043e-03  4.485471e-13
      #  [600,] 1.402364e-06 3.964986e-03  2.304839e-15
      z = dens1 / apply(dens1, 1, sum)  # 600-by-3
      # Debug> apply(dens1, 1, sum) --for each row, sum 3 elements, the result is a vector of 600
      #  [1] 4.651836e-01 5.942919e-02 6.309434e-03 6.206078e-02 3.401010e-02 1.743022e+00 2.790461e-01
      # Debug> z --- each column of dens1 divided by the vector "apply(dens1, 1, sum)"
      #               [,1]         [,2]          [,3]
      #  [1,] 1.000000e+00 6.471478e-14  9.668267e-89
      #  [2,] 9.999984e-01 1.607892e-06  4.106034e-55
      #  [3,] 9.010292e-01 9.897083e-02  9.850977e-33
      # Debug> 3.010425e-14/4.651836e-01
      #[1] 6.471477e-14
      # Debug> 9.555570e-08/5.942919e-02
      #[1] 1.607892e-06

      lambda.hat = apply(z, 2, mean)  # lambda.hat is 1-by-3
      # Debug> lambda.hat
      #[1] 0.3253258 0.3413408 0.3333334
      # Debug> mean(yjfc), where yjfc = z[,1]
      #[1] 0.3253258
      # Debug> yjfc = z[,2]
      #Debug> mean(yjfc)
      #[1] 0.3413408

      out = try(suppressWarnings(nlm(gamma.ll, p = theta, lambda = lambda.hat, k = k, z = z)),
                silent = TRUE)
      # gamma.ll: sum(600-by-3)=1, theta: k+k, lambda.hat: 1-by-3, k: 3, z: 600-by-3
      # gamma.ll <- function(theta, z, lambda, k) -sum(z * log(dens(lambda, theta, k)))
      # element-wise product between z and log(dens(lambda, theta, k)), result is 600-by-3
      if (class(out) == "try-error") {
        # The value of the expression if expr is evaluated without error,
        # but an invisible object of class "try-error" containing the error message,
        # and the error condition as the "condition" attribute, if it fails.
        cat("Note: Choosing new starting values.", "\n")
        if (mr == maxrestarts) stop(paste("Try different number of components?", "\n"))
        mr <- mr + 1
        tmp <- gammamix.init(x = x, k = k)
        lambda <- tmp$lambda
        alpha <- tmp$alpha
        beta <- tmp$beta
        theta <- c(alpha, beta)
        k <- tmp$k
        iter <- 0
        diff <- epsilon + 1
        old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
        ll <- old.obs.ll
      } else {
        theta.hat = out$estimate
        alpha.hat = theta.hat[1:k]
        beta.hat = theta.hat[(k + 1):(2 * k)]
        # Debug> theta.hat
        #[1]  0.3571404  6.1872099 26.8053743  5.4571960  4.9386467 12.1372546
        new.obs.ll <- sum(log(apply(dens(lambda.hat, theta.hat, k), 1, sum)))
        diff <- new.obs.ll - old.obs.ll
        old.obs.ll <- new.obs.ll
        ll <- c(ll, old.obs.ll)
        # Debug> ll
        #[1] -2539.123 -2538.997
        lambda = lambda.hat
        theta = theta.hat
        alpha = alpha.hat
        beta = beta.hat
        iter = iter + 1
        if (verb) {
          cat("iteration =", iter, " log-lik diff =", diff, " log-lik =",
              new.obs.ll, "\n")
        }
      }
    }
    if (iter == maxit) {
      cat("WARNING! NOT CONVERGENT!", "\n")
    }
    cat("number of iterations=", iter, "\n")
    theta = rbind(alpha, beta)
    rownames(theta) = c("alpha", "beta")
    colnames(theta) = c(paste("comp", ".", 1:k, sep = ""))
    a = list(x = x, lambda = lambda, gamma.pars = theta, loglik = new.obs.ll,
             posterior = z, all.loglik = ll, ft = "gammamixEM")
    class(a) = "mixEM"
    a
  }	