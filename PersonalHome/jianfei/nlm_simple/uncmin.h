//
// Created by jianfei on 20-6-4.
//

#ifndef NLM_SIMPLE_UNCMIN_H
#define NLM_SIMPLE_UNCMIN_H

/* type of pointer to the target and gradient functions */
typedef void (*fcn_p)(int, double *, double *, void *);
/* type of pointer to the hessian functions */
typedef void (*d2fcn_p)(int, int, double *, double *, void *);

void optif9(int nr, int n, double *x, fcn_p fcn, fcn_p d1fcn, d2fcn_p d2fcn,
       void *state, double *typsiz, double fscale, int method,
       int iexp, int *msg, int ndigit, int itnlim, int iagflg, int iahflg,
       double dlt, double gradtl, double stepmx, double steptl,
       double *xpls, double *fpls, double *gpls, int *itrmcd, double *a,
       double *wrk, int *itncnt);

#endif //NLM_SIMPLE_UNCMIN_H
