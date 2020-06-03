

// #ifdef HAVE_CONFIG_H
#include <config.h>
// #endif

#define NO_NLS
#include <Defn.h>
#include <float.h>		/* for DBL_MAX */
#include <R_ext/Applic.h>	/* for optif9, fdhess */
#include <R_ext/RS.h>	       	/* for Memcpy */

#include "statsR.h"
#include "stats.h" // R_zeroin2

#undef _
#ifdef ENABLE_NLS
#include <libintl.h>
#define _(String) dgettext ("stats", String)
#else
#define _(String) (String)
#endif





/* Fatal errors - we don't deliver an answer */

static void NORET opterror(int nerr)
{
    switch(nerr) {
        case -1:
            error(_("non-positive number of parameters in nlm"));
        case -2:
            error(_("nlm is inefficient for 1-d problems"));
        case -3:
            error(_("invalid gradient tolerance in nlm"));
        case -4:
            error(_("invalid iteration limit in nlm"));
        case -5:
            error(_("minimization function has no good digits in nlm"));
        case -6:
            error(_("no analytic gradient to check in nlm!"));
        case -7:
            error(_("no analytic Hessian to check in nlm!"));
        case -21:
            error(_("probable coding error in analytic gradient"));
        case -22:
            error(_("probable coding error in analytic Hessian"));
        default:
            error(_("*** unknown error message (msg = %d) in nlm()\n*** should not happen!"), nerr);
    }
}

/* Warnings - we return a value, but print a warning */

static void optcode(int code)
{
    switch(code) {
        case 1:
            Rprintf(_("Relative gradient close to zero.\n"));
            Rprintf(_("Current iterate is probably solution.\n"));
            break;
        case 2:
            Rprintf(_("Successive iterates within tolerance.\n"));
            Rprintf(_("Current iterate is probably solution.\n"));
            break;
        case 3:
            Rprintf(_("Last global step failed to locate a point lower than x.\n"));
            Rprintf(_("Either x is an approximate local minimum of the function,\n\
the function is too non-linear for this algorithm,\n\
or steptol is too large.\n"));
            break;
        case 4:
            Rprintf(_("Iteration limit exceeded.  Algorithm failed.\n"));
            break;
        case 5:
            Rprintf(_("Maximum step size exceeded 5 consecutive times.\n\
Either the function is unbounded below,\n\
becomes asymptotic to a finite value\n\
from above in some direction,\n"\
"or stepmx is too small.\n"));
            break;
    }
    Rprintf("\n");
}


/* Store an entry in the table of computed function values */
static void FT_store(int n, const double f, const double *x, const double *grad,
                     const double *hess, function_info *state) {
    int ind;

    ind = (++(state->FT_last)) % (state->FT_size);   //keep max 5 caches.
    state->Ftable[ind].fval = f;
    //typedef struct {
    //    double   fval;
    //    double  *x;      //vector of size=n
    //    double  *grad;   //vector of size=n
    //    double  *hess;   //vector of size = n * n
    //} ftable;
    Memcpy(state->Ftable[ind].x, x, n);             //keep values of inputs of parameters
    if (grad) {
        Memcpy(state->Ftable[ind].grad, grad, n);
        if (hess) {
            Memcpy(state->Ftable[ind].hess, hess, n * n);
        }
    }
}

/* Check for stored values in the table of computed function values.
   Returns the index in the table or -1 for failure */

static int FT_lookup(int n, const double *x, function_info *state)
{
    double *ftx;
    int i, j, ind, matched;
    int FT_size, FT_last;
    ftable *Ftable;

    FT_last = state->FT_last;   //initial=-1
    FT_size = state->FT_size;   //default=5
    Ftable = state->Ftable;     //length of Ftable is 5

    for (i = 0; i < FT_size; i++) {
        ind = (FT_last - i) % FT_size;
        /* why can't they define modulus correctly */
        if (ind < 0) ind += FT_size;
        ftx = Ftable[ind].x;
        if (ftx) {
            matched = 1;
            for (j = 0; j < n; j++) {
                if (x[j] != ftx[j]) {       //yjf.WARNING:  equal in double?
                    // typedef struct {
                    //    double   fval;
                    //    double  *x;      //vector of size=n
                    //    double  *grad;   //vector of size=n
                    //    double  *hess;   //vector of size = n * n
                    //} ftable;
                    matched = 0;            //check whether the new input x is the cache of ftx
                    break;
                }
            }
            if (matched) return ind;
        }
    }
    return -1;
}


/* This how the optimizer sees them */

static void fcn(int n, const double x[], double *f, function_info *state) {
    SEXP s, R_fcall;
    ftable *Ftable;
    double *g = (double *) 0, *h = (double *) 0;
    int i;

    R_fcall = state->R_fcall;
    Ftable = state->Ftable;
    if ((i = FT_lookup(n, x, state)) >= 0) {
        *f = Ftable[i].fval;                    //found f's value in the cache
        return;
    }
    /* calculate for a new value of x */
    s = allocVector(REALSXP, n);
    SETCADR(R_fcall, s);
    for (i = 0; i < n; i++) {
        //check if the searching-point x is finite value
        if (!R_FINITE(x[i])) error(_("non-finite value supplied by 'nlm'"));
        REAL(s)[i] = x[i];
    }

    //run the objective function R_fcall stored in state object
    //R_fcall will also compute gradient/hessian if analysis-gradient/hessian are provided
    s = PROTECT(eval(state->R_fcall, state->R_env));

    switch(TYPEOF(s)) {
        case INTSXP:
            if (length(s) != 1) goto badvalue;
            if (INTEGER(s)[0] == NA_INTEGER) {
                warning(_("NA replaced by maximum positive value"));
                *f = DBL_MAX;   //check the overflow of returned integer
            }
            else *f = INTEGER(s)[0];
            break;
        case REALSXP:
            if (length(s) != 1) goto badvalue;
            if (!R_FINITE(REAL(s)[0])) {
                warning(_("NA/Inf replaced by maximum positive value"));
                *f = DBL_MAX; //check the overflow of returned real
            }
            else *f = REAL(s)[0];
            break;
        default:
            goto badvalue;
    }
    if (state->have_gradient) {
        //get the analysis-gradient attribute-value in the evaluation of R_fcall
        g = REAL(PROTECT(coerceVector(getAttrib(s, install("gradient")), REALSXP)));

        if (state->have_hessian) {
            h = REAL(PROTECT(coerceVector(getAttrib(s, install("hessian")), REALSXP)));
        }
    }

    FT_store(n, *f, x, g, h, state);


    UNPROTECT(1 + state->have_gradient + state->have_hessian);
    return;

    badvalue:
    error(_("invalid function value in 'nlm' optimizer"));
}

static void Cd1fcn(int n, const double x[], double *g, function_info *state)
{
    int ind;

    if ((ind = FT_lookup(n, x, state)) < 0) {	/* shouldn't happen */
        //fcn should be invoked before Cd1fcn
        //fcn(int n, const double x[], double *f, function_info *state), *f is the returned objectfun's value
        fcn(n, x, g, state);
        //g is the return-value of the objectfun,
        // the analysis-gradient/hessian are automatically computed in the body of fcn

        if ((ind = FT_lookup(n, x, state)) < 0) {
            error(_("function value caching for optimization is seriously confused"));
        }
    }

    Memcpy(g, state->Ftable[ind].grad, n);  //return the cached analysis-gradient
}


static void Cd2fcn(int nr, int n, const double x[], double *h,
                   function_info *state)
{
    int j, ind;

    if ((ind = FT_lookup(n, x, state)) < 0) {	/* shouldn't happen */
        fcn(n, x, h, state);
        if ((ind = FT_lookup(n, x, state)) < 0) {
            error(_("function value caching for optimization is seriously confused"));
        }
    }
    for (j = 0; j < n; j++) {  /* fill in lower triangle only */
        //j is the column-index in [1, n]
        //the Hessian matrix is store in column order
        Memcpy( h + j*(n + 1), state->Ftable[ind].hess + j*(n + 1), n - j);
        // h + j*n + j <----  (hess + j*n + j) [0, n-j)
        // j * (n+1) = j*n + j(pass some locations to the current j-th diagonal element),
    }
}


/* NOTE: The actual Dennis-Schnabel algorithm `optif9' is in uncmin.c */

/* Rinternals:  typedef struct SEXPREC *SEXP; */
/* /* The standard node structure consists of a header followed by the
   node data. */
/*
typedef struct SEXPREC {
    SEXPREC_HEADER;
    union {
        struct primsxp_struct primsxp;
        struct symsxp_struct symsxp;
        struct listsxp_struct listsxp;
        struct envsxp_struct envsxp;
        struct closxp_struct closxp;
        struct promsxp_struct promsxp;
    } u;
} SEXPREC;
 * */

SEXP nlm(SEXP call, SEXP op, SEXP args, SEXP rho)
{
    SEXP value, names, v, R_gradientSymbol, R_hessianSymbol;

    double *x, *typsiz, fscale, gradtl, stepmx,
            steptol, *xpls, *gpls, fpls, *a, *wrk, dlt;

    int code, i, j, k, itnlim, method, iexp, omsg, msg,
            n, ndigit, iagflg, iahflg, want_hessian, itncnt;


/* .Internal(
 *	nlm(function(x) f(x, ...), p=theta, hessian, typsize, fscale,
 *	    msg, ndigit, gradtol, stepmax, steptol, iterlim)
 */
    function_info *state;


    //typedef struct {
    //    SEXP R_fcall;	      /* unevaluated call to R function */  PROTECT(state->R_fcall = lang2(v, R_NilValue));
    //    SEXP R_env;	      /* where to evaluate the calls */ state->R_env = rho;
    //    int have_gradient;
    //    int have_hessian;
    //    int n;	           length of the parameter (x) vector  //not being set
    //    int FT_size;	     // size of table to store computed function values
    //    int FT_last;	      /* Newest entry in the table */
    //    ftable *Ftable;
    //} function_info;
    state = (function_info *) R_alloc(1, sizeof(function_info));

    /* the function to be minimized */
    // in gammamixEM.R call nlm(gamma.ll, p = theta, lambda = lambda.hat, k = k, z = z)
    //function(x) f(x, ...), where ... is
    // lambda = lambda.hat, k = k, z = z
    v = CAR(args);
    if (!isFunction(v))
        error(_("attempt to minimize non-function"));
    PROTECT(state->R_fcall = lang2(v, R_NilValue));
    args = CDR(args);
    //INLINE_FUN SEXP lang2(SEXP s, SEXP t)
    //{
    //    PROTECT(s);
    //    s = LCONS(s, list1(t));
    //    UNPROTECT(1);
    //    return s;
    //}
    // LibExtern SEXP	R_NilValue;	    /* The nil object */

    /* `p' : inital parameter value */

    n = 0;
    x = fixparam(CAR(args), &n);
    args = CDR(args);

    /* `hessian' : H. required? */

    want_hessian = asLogical(CAR(args));
    if (want_hessian == NA_LOGICAL) want_hessian = 0;
    args = CDR(args);

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
}