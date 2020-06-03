//
// Created by jianfei on 20-5-29.
//

#include <stdio.h>
#include "nlm_simple.h"
#include "FunctionInfo.h"

//double nlm_simple(LossFunType f, double* xInit,
//        unsigned long n, double* xOutput, int* resultCode, int* iterCount,
//        double *c_typsize, double c_fscale, int c_print_level,
//        int c_ndigit, double c_gradtol,
//        int c_stepmax, double c_steptol,
//        int c_iterlim, bool c_check_analyticals);
////c_typsize = [1,1,..1], length=length(xInit)
//// c_fscale = 1, c_print_level = 0 in {0,1,2},
//// c_ndigit = 12, c_gradtol = 1e-06,
//// c_stepmax = max(1000 * sqrt( sum((xInit/typsize)^2) ), 1000),
//// c_steptol = 1e-06,
//// c_iterlim = 100,
//// c_check_analyticals = TRUE

double nlm_simple(LossFunType f, double* xInit,
        unsigned long n, double* xOutput, int* resultCode, int* iterCount,
        double *c_typsize, double c_fscale, int c_print_level,
        int c_ndigit, double c_gradtol,
        int c_stepmax, double c_steptol,
        int c_iterlim, bool c_check_analyticals) {
//return function value at the point xOutput

    if (c_print_level < 0) {
        c_print_level = 0;
    }
    else if (c_print_level > 2) {
        c_print_level = 2;
    }
    int msgArr[] = {9, 1, 16};



    print.level <- as.integer(print.level)
    if (print.level < 0 || print.level > 2)
        stop("'print.level' must be in {0,1,2}")
    msg <- (1 + c(8, 0, 16))[1 + print.level]
    if (!check.analyticals)  msg <- msg + (2 + 4)

    FunctionInfo state(f, n);


    //hessian = FALSE
    typsize = rep(1, length(p)),
    fscale = 1, print.level = 0, ndigit = 12, gradtol = 1e-06,
    stepmax = max(1000 * sqrt(sum((p/typsize)^2)), 1000), steptol = 1e-06,
    iterlim = 100, check.analyticals = TRUE

    /* `typsize' : typical size of parameter elements */

    typsiz = fixparam(CAR(args), &n);
    args = CDR(args);

    /* `fscale' : expected function size */

    fscale = asReal(CAR(args));
    if (ISNA(fscale)) error(_("invalid NA value in parameter"));
    args = CDR(args);

    /* `msg' (bit pattern) */
    omsg = msg = asInteger(CAR(args));
    if (msg == NA_INTEGER) error(_("invalid NA value in parameter"));
    args = CDR(args);

    ndigit = asInteger(CAR(args));
    if (ndigit == NA_INTEGER) error(_("invalid NA value in parameter"));
    args = CDR(args);

    gradtl = asReal(CAR(args));
    if (ISNA(gradtl)) error(_("invalid NA value in parameter"));
    args = CDR(args);

    stepmx = asReal(CAR(args));
    if (ISNA(stepmx)) error(_("invalid NA value in parameter"));
    args = CDR(args);

    steptol = asReal(CAR(args));
    if (ISNA(steptol)) error(_("invalid NA value in parameter"));
    args = CDR(args);

    /* `iterlim' (def. 100) */
    itnlim = asInteger(CAR(args));
    if (itnlim == NA_INTEGER) error(_("invalid NA value in parameter"));

    state->R_env = rho;

    /* force one evaluation to check for the gradient and hessian */
    iagflg = 0;			/* No analytic gradient */
    iahflg = 0;			/* No analytic hessian */
    state->have_gradient = 0;
    state->have_hessian = 0;
    R_gradientSymbol = install("gradient");
    R_hessianSymbol = install("hessian");

    v = allocVector(REALSXP, n);
    for (i = 0; i < n; i++) REAL(v)[i] = x[i];  //x: `p' : initial parameter value
    SETCADR(state->R_fcall, v);                 //set initial parameter values for the callback fun
    PROTECT(value = eval(state->R_fcall, state->R_env));

    v = getAttrib(value, R_gradientSymbol);     //check if analysis-gradient is provided
    if (v != R_NilValue) {
        if (LENGTH(v) == n && (isReal(v) || isInteger(v))) {
            iagflg = 1;
            state->have_gradient = 1;                   //set state->have_gradient = 1
            v = getAttrib(value, R_hessianSymbol);

            if (v != R_NilValue) {
                if (LENGTH(v) == (n * n) && (isReal(v) || isInteger(v))) {
                    iahflg = 1;
                    state->have_hessian = 1;            //set state->have_hessian = 1
                } else {
                    warning(_("hessian supplied is of the wrong length or mode, so ignored"));
                }
            }
        } else {
            warning(_("gradient supplied is of the wrong length or mode, so ignored"));
        }
    }
    UNPROTECT(1); /* value: analysis-gradient/hessian */

    // in the body of the R nlm
    //  if (!check.analyticals)
    //    msg <- msg + (2 + 4)
    if (((msg/4) % 2) && !iahflg) { /* skip check of analytic Hessian */
        msg -= 4;   //if analysis-hessian is not provided, msg -= 4
    }
    if (((msg/2) % 2) && !iagflg) { /* skip check of analytic gradient */
        msg -= 2;   //if analysis-hessian is not provided, msg -= 2
    }
    FT_init(n, FT_SIZE, state);  //allocate 5 Ftable in function_info *state
    //    //typedef struct {
    //    //  SEXP R_fcall;	      /* unevaluated call to R function */
    //    //  SEXP R_env;	      /* where to evaluate the calls */
    //
    //    //  int have_gradient;  //default=0
    //    //  int have_hessian;   //default=0
    //
    //    //  int n;	             length of the parameter (x) vector
    //    //  int FT_size;	      /* size of table to store computed function values */ initial=5
    //    //  int FT_last;	      /* Newest entry in the table */, initial=-1
    //    //  ftable *Ftable;
    //    //} function_info;

    /* Plug in the call to the optimizer here */

    method = 1;	/* Line Search */
    // not expensive if analysis-Hessian is provided.
    iexp = iahflg ? 0 : 1; /* Function calls are expensive */
    dlt = 1.0;

    xpls = (double*)R_alloc(n, sizeof(double));
    gpls = (double*)R_alloc(n, sizeof(double));
    a = (double*)R_alloc(n*n, sizeof(double));
    wrk = (double*)R_alloc(8*n, sizeof(double));

    /*
     *	 Dennis + Schnabel Minimizer
     *
     *	  SUBROUTINE OPTIF9(NR,N,X,FCN,D1FCN,D2FCN,TYPSIZ,FSCALE,
     *	 +	   METHOD,IEXP,MSG,NDIGIT,ITNLIM,IAGFLG,IAHFLG,IPR,
     *	 +	   DLT,GRADTL,STEPMX,STEPTOL,
     *	 +	   XPLS,FPLS,GPLS,ITRMCD,A,WRK)
     *
     *
     *	 Note: I have figured out what msg does.
     *	 It is actually a sum of bit flags as follows
     *	   1 = don't check/warn for 1-d problems
     *	   2 = don't check analytic gradients
     *	   4 = don't check analytic hessians
     *	   8 = don't print start and end info
     *	  16 = print at every iteration
     *	 Using msg=9 is absolutely minimal
     *	 I think we always check gradients and hessians
     */

    optif9(n, n, x, (fcn_p) fcn, (fcn_p) Cd1fcn, (d2fcn_p) Cd2fcn,
           state, typsiz, fscale, method, iexp, &msg, ndigit, itnlim,
           iagflg, iahflg, dlt, gradtl, stepmx, steptol, xpls, &fpls,
           gpls, &code, a, wrk, &itncnt);

    //optif9(int nr, int n, double *x, fcn_p fcn, fcn_p d1fcn, d2fcn_p d2fcn,
    //       void *state, double *typsiz, double fscale, int method,
    //       int iexp, int *msg, int ndigit, int itnlim,
    //       int iagflg, int iahflg, double dlt, double gradtl, double stepmx, double steptl,
    //       double *xpls, double *fpls, double *gpls, int *itrmcd, double *a,
    //       double *wrk, int *itncnt)

    if (msg < 0)
        opterror(msg);
    if (code != 0 && (omsg&8) == 0)
        optcode(code);

    if (want_hessian) {
        PROTECT(value = allocVector(VECSXP, 6));
        PROTECT(names = allocVector(STRSXP, 6));
        fdhess(n, xpls, fpls, (fcn_p) fcn, state, a, n, &wrk[0], &wrk[n],
               ndigit, typsiz);
        for (i = 0; i < n; i++)
            for (j = 0; j < i; j++)
                a[i + j * n] = a[j + i * n];
    }
    else {
        PROTECT(value = allocVector(VECSXP, 5));
        PROTECT(names = allocVector(STRSXP, 5));
    }
    k = 0;

    SET_STRING_ELT(names, k, mkChar("minimum"));
    SET_VECTOR_ELT(value, k, ScalarReal(fpls));         //double: should return to python-end
    k++;

    //out$estimate
    SET_STRING_ELT(names, k, mkChar("estimate"));      //array[1-by-2k]: must return to python-end
    SET_VECTOR_ELT(value, k, allocVector(REALSXP, n));

    //xpls = (double*)R_alloc(n, sizeof(double));   //the memory of the return values is allocated in C-end
    for (i = 0; i < n; i++)
        REAL(VECTOR_ELT(value, k))[i] = xpls[i];  //xpls(n)	    <--> on exit:  xpls is local minimum
    k++;

    SET_STRING_ELT(names, k, mkChar("gradient"));
    SET_VECTOR_ELT(value, k, allocVector(REALSXP, n));
    for (i = 0; i < n; i++)
        REAL(VECTOR_ELT(value, k))[i] = gpls[i];
    k++;

    if (want_hessian) {
        SET_STRING_ELT(names, k, mkChar("hessian"));
        SET_VECTOR_ELT(value, k, allocMatrix(REALSXP, n, n));
        for (i = 0; i < n * n; i++)
            REAL(VECTOR_ELT(value, k))[i] = a[i];
        k++;
    }

    SET_STRING_ELT(names, k, mkChar("code"));
    SET_VECTOR_ELT(value, k, allocVector(INTSXP, 1));
    INTEGER(VECTOR_ELT(value, k))[0] = code;        //should return to python-end
    k++;

    /* added by Jim K Lindsey */
    SET_STRING_ELT(names, k, mkChar("iterations"));
    SET_VECTOR_ELT(value, k, allocVector(INTSXP, 1));
    INTEGER(VECTOR_ELT(value, k))[0] = itncnt;     //should return to python-end
    k++;

    setAttrib(value, R_NamesSymbol, names);
    UNPROTECT(3);
    return value;

    //print input initial data
    int i=0;
    for(; i<n; ++i) {
        printf("nlm_simple.so: x_init[%d] is %f \n", i, xInit[i]);
    }

    //test if we can call into the python-end with toy data
    i=0;
    for(; i<n; ++i) {
        xOutput[i] = 1000 + i;           //toy solution
    }

    *code = 111;
    *iterCount = 101;

    double loss = (*f)(xOutput, n);
    printf("nlm_simple.so: python loss is %f \n", loss);

    return loss;
}