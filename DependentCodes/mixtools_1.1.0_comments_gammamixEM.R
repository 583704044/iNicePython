gammamixEM <-
  function(x, lambda = NULL, alpha = NULL, beta = NULL, k = 2,
           epsilon = 1e-08, maxit = 1000, maxrestarts = 20, verb = FALSE) {
    # 5) k: Number of components. Initial value ignored unless alpha and beta are both NULL.
    # 6) epsilon: The convergence criterion.
    # Convergence is declared when the change in the
    # observed data log-likelihood increases (yjf: maximized log-liklihood) by less than epsilon.

    # 7) maxit: The maximum number of iterations.

    # 8) maxrestarts: The maximum number of restarts allowed in case of
    # a problem with the particular starting values chosen (each restart uses randomly chosen starting values).

    # 9) verb: If TRUE, then various updates are printed during each iteration of the algorithm.

    # 10) x: A vector of length n consisting of the data.
    x <- as.vector(x)

    tmp <- gammamix.init(x = x, lambda = lambda, alpha = alpha, beta = beta, k = k)
    #list(lambda = lambda, alpha = alpha, beta = beta, k = k)
    lambda <- tmp$lambda
    # 2) lambda: Initial value of mixing proportions. If NULL,
    # then lambda is random from a uniform Dirichlet distribution
    # (i.e., its entries are uniform random and then it is normalized to sum to 1).

    alpha <- tmp$alpha
    # 3) alpha: Starting value of vector of component shape parameters.
    # If non-NULL, alpha must be of length k if allowng different component shape parameters,
    # or a single value if fix.alpha = TRUE.
    # If NULL, then the initial value is estimated by partitioning the data into k regions
    # (with lambda determining the proportion of values in each region) and
    # then calculating the method of moments estimates.
    beta <- tmp$beta
    # 4) beta: Starting value of vector of component scale parameters. (yjf: actually it is rate parameter)
    # If non-NULL and a vector, k is set to length(beta).
    # If NULL, then the initial value is estimated the same method described for alpha.
    theta <- c(alpha, beta)

    k <- tmp$k

    iter <- 0
    mr <- 0
    diff <- epsilon + 1
    n <- length(x)
    dens <- NULL
    dens <- function(lambda, theta, k) {
      temp <- NULL
      alpha = theta[1:k]  # theta <- c(alpha, beta), the first k alphas and the seond k betas
      beta = theta[(k + 1):(2 * k)]
      for (j in 1:k) {
        #
        temp = cbind(temp, dgamma(x, shape = alpha[j], scale = beta[j])) }
      temp = t(lambda * t(temp))  # t: matrix transpose
      temp
    }
    old.obs.ll <- sum(log(apply(dens(lambda, theta, k), 1, sum)))
    ll <- old.obs.ll
    gamma.ll <- function(theta, z, lambda, k) -sum(z * log(dens(lambda, theta, k)))

    while (diff > epsilon && iter < maxit) {
      dens1 = dens(lambda, theta, k)
      z = dens1 / apply(dens1, 1, sum)
      lambda.hat = apply(z, 2, mean)

      out = try(suppressWarnings(nlm(gamma.ll, p = theta, lambda = lambda.hat, k = k, z = z)),
                silent = TRUE)
      if (class(out) == "try-error") {
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
        new.obs.ll <- sum(log(apply(dens(lambda.hat, theta.hat, k), 1, sum)))
        diff <- new.obs.ll - old.obs.ll
        old.obs.ll <- new.obs.ll
        ll <- c(ll, old.obs.ll)
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

