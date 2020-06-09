import numpy as np
import scipy
from scipy import stats
from nlmMinimizer import nlmMinimizer
from functools import partial


def init(x, lbda=None, alpha=None, beta=None, k=2):
    n = len(x)
    if lbda is None:
        cond = True
        while cond:
            lbda = np.random.uniform(size=k)  # random
            lbda = lbda/np.sum(lbda)
            if np.min(lbda) < 0.05:
                cond = True
            else:
                cond = False
    else:
        k = len(lbda)
    if k == 1:
        x_bar = np.mean(x)
        x2_bar = np.mean(x**2)
    else:
        x_sort = np.sort(x)
        ind = np.floor(n * np.cumsum(lbda))
        ind = np.array([int(i) for i in ind])
        x_part = list()
        x_part.append(x_sort[0:ind[0]+1])
        for j in range(2, k+1):
            x_part.append(x_sort[ind[j-2]-1:ind[j-1]])
        x_bar = np.array([np.mean(i) for i in x_part])
        x2_bar = np.array([np.mean(i) for i in [[i**2 for i in j] for j in x_part]])
    if alpha is None:
        alpha = x_bar**2/(x2_bar - x_bar**2)
    if beta is None:
        beta = (x2_bar - x_bar**2)/x_bar
    return {'lambda': lbda, 'alpha': alpha, 'beta': beta, 'k': k}


def gammamixEM(x, lbda=None, alpha=None, beta=None, k=2, epsilon=1e-08, maxit=1000, maxrestarts=20, verb=False):
    tmp = init(x=x, lbda=lbda, alpha=alpha, beta=beta, k=k)
    lbda = tmp['lambda']
    alpha = tmp['alpha']
    beta = tmp['beta']
    theta = np.concatenate((alpha, beta))
    k = tmp['k']
    iter = 0
    mr = 0
    diff = epsilon + 1
    n = len(x)

    def dens(lbda, theta, k):
        temp = np.empty((len(x), 0))
        alpha = theta[:k]
        beta = theta[k:2*k]
        for j in range(k):
            temp = np.concatenate((temp, scipy.stats.gamma.pdf(x, alpha[j], scale=beta[j])[:, np.newaxis]), axis=1)
        temp = np.array([lbda*i for i in temp])
        return temp

    old_obs_ll = np.sum(np.log(np.sum(dens(lbda, theta, k), axis=1)))
    ll = [old_obs_ll]

    def gamma_ll(theta, z, lbda, k):
        return -np.sum(z * np.log(dens(lbda, theta, k)))

    while diff > epsilon and iter < maxit:
        dens1 = dens(lbda, theta, k)
        z = np.array([i/np.sum(i) for i in dens1])
        lambda_hat = np.mean(z, axis=0)

        try:
            bound_f = partial(gamma_ll, z=z, lbda=lambda_hat, k=k)
            m = nlmMinimizer(theta, bound_f)
            flag, retValue = m()
            estimate = m.getXSolution()

            # print('...........after call...........')
            # print('m call: ok=', flag, ', retVal=', retValue)
            # print('...........summary info...........')
            # m.showResult()
        except:
            print "Note: Choosing new starting values."
            if mr == maxrestarts:
                raise SystemExit("Try different number of components?")
            mr = mr + 1
            tmp = init(x = x, k = k)
            lbda = tmp['lambda']
            alpha = tmp['alpha']
            beta = tmp['beta']
            theta = np.concatenate((alpha, beta))
            k = tmp['k']
            iter = 0
            diff = epsilon + 1
            old_obs_ll = np.sum(np.log(np.sum(dens(lbda, theta, k), axis=1)))
            ll = old_obs_ll
            continue

        theta_hat = estimate
        alpha_hat = theta_hat[:k]
        beta_hat = theta_hat[k:2*k]
        new_obs_ll = np.sum(np.log(np.sum(dens(lambda_hat, theta_hat, k), axis=1)))
        diff = new_obs_ll - old_obs_ll
        old_obs_ll = new_obs_ll
        ll.append(old_obs_ll)
        lbda = lambda_hat
        theta = theta_hat
        alpha = alpha_hat
        beta = beta_hat
        iter = iter + 1
        if verb:
            print "iteration =", iter, " log-lik diff =", diff, " log-lik =", new_obs_ll

    if iter == maxit:
        print "WARNING! NOT CONVERGENT!"
    print "number of iterations=", iter
    theta = np.concatenate((alpha, beta))
    a = {'x': x, 'lambda':lbda, 'gamma_pars': theta, 'loglik': new_obs_ll, 'posterior': z, 'all_loglik': ll,
         'ft': 'gammamixEM'}
    return a

if __name__ == '__main__':
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = gammamixEM(x, lbda=np.array([1, 1, 1])/3.0, verb=True, maxrestarts=30)
    print out['lambda']
    print out['gamma_pars']
