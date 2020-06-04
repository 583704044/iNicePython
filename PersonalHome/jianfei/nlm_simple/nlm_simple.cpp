//
// Created by jianfei on 20-5-29.
//

#include "nlm_simple.h"
#include "FunctionInfo.h"
#include <iostream>
using namespace std;


#define FT_SIZE 5		/* default size of table to store computed function values */

static void optcode(int code)
{
    switch(code) {
        case 1:
            cout << "Relative gradient close to zero.\n";
            cout << "urrent iterate is probably solution.\n";
            break;
        case 2:
            cout << "Successive iterates within tolerance.\n";
            cout << "Current iterate is probably solution.\n";
            break;
        case 3:
            cout << "Last global step failed to locate a point lower than x.\n";
            cout << "Either x is an approximate local minimum of the function,\n\
the function is too non-linear for this algorithm,\n\
or steptol is too large.\n";
            break;
        case 4:
            cout << "Iteration limit exceeded.  Algorithm failed.\n";
            break;
        case 5:
            cout << "Maximum step size exceeded 5 consecutive times.\n\
Either the function is unbounded below,\n\
becomes asymptotic to a finite value\n\
from above in some direction,\n"\
"or stepmx is too small.\n";
            break;
    }
    cout << "\n";
}

/* Fatal errors - we don't deliver an answer */

static void opterror(int nerr)
{
    switch(nerr) {
        case -1:
            cout << "ERROR: non-positive number of parameters in nlm" << endl;
            break;

        case -2:
            cout << "ERROR: nlm is inefficient for 1-d problems" << endl;
            break;

        case -3:
            cout << "ERROR: invalid gradient tolerance in nlm" << endl;
            break;

        case -4:
            cout << "ERROR: invalid iteration limit in nlm" << endl;
            break;

        case -5:
            cout << "ERROR: minimization function has no good digits in nlm" << endl;
            break;

        case -6:
            cout << "ERROR: no analytic gradient to check in nlm!" << endl;
            break;

        case -7:
            cout << "ERROR: no analytic Hessian to check in nlm!" << endl;
            break;

        case -21:
            cout << "probable coding error in analytic gradient" << endl;
            break;

        case -22:
            cout << "probable coding error in analytic Hessian" << endl;
            break;

        default:
            cout << "*** unknown error message (msg= " << nerr << ") in nlm_simple\n*** should not happen!" << endl;
            break;
    }
}

static void fcn(int n, const double x[], double *f, FunctionInfo *state) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////
double nlm_simple(LossFunType f, double* xInit,
        unsigned long n, double* xOutput, int* resultCode, int* iterCount,
        double *c_typsize, double c_fscale, int c_print_level,
        int c_ndigit, double c_gradtol,
        int c_stepmax, double c_steptol,
        int c_iterlim, bool c_check_analyticals) {
//return function value at the point xOutput

    //
    //msg
    //
    if (c_print_level < 0) {
        c_print_level = 0;
    }
    else if (c_print_level > 2) {
        c_print_level = 2;
    }
    const int msgArr[] = {9, 1, 17};  //msg <- (1 + c(8, 0, 16))[1 + print.level]
    int msg = msgArr[c_print_level];
    int omsg = msg;                  //keep a copy to used after calling optif9
    if (!c_check_analyticals)
        msg += (2 + 4);

    //
    // analytic gradient, hessian
    //
    int iagflg = 0;
    int iahflg = 0;

    if ((msg/4) % 2 && !iahflg) { /* skip check of analytic Hessian */
        msg -= 4;   //if analysis-hessian is not provided, msg -= 4
    }
    if ((msg/2) % 2 && !iagflg) { /* skip check of analytic gradient */
        msg -= 2;   //if analysis-hessian is not provided, msg -= 2
    }

    //
    // state
    //
    FunctionInfo state(f, n);
    state.createFTable(FT_SIZE);

    /* Plug in the call to the optimizer here */

    int method = 1;	/* Line Search */
    // not expensive if analysis-Hessian is provided.
    int iexp = 1;   /* Function calls are expensive */
    double dlt = 1.0;

    double* gpls= new double[n];     // <--> on exit:  gradient at solution xpls
    double* a= new double[n * n];    // a(n,n) --> workspace for hessian (or estimate)
                                     // and its cholesky decomposition
    double* wrk = new double[8 * n]; // wrk(n,8)   --> workspace

    //<--> on exit:  function value at solution, xpls
    double fpls = 100000.0;
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

    optif9(n, n, xInit, (fcn_p) fcn, (fcn_p) Cd1fcn, (d2fcn_p) Cd2fcn,
           &state, c_typsize, c_fscale, method, iexp, &msg, c_ndigit, c_iterlim,
           iagflg, iahflg, dlt, c_gradtol, c_stepmax, c_steptol, xOutput, &fpls,
           gpls, resultCode, a, wrk, iterCount);
    /*	provide complete interface to minimization package.
 *	user has full control over options.

 * PARAMETERS :

 *	nr	     --> row dimension of matrix
 *	n	     --> dimension of problem
 *	x(n)	 --> on entry: estimate to a root of fcn
 *	fcn	     --> name of subroutine to evaluate optimization function
 *			     must be declared external in calling routine
 *				   fcn: r(n) --> r(1)
 *	d1fcn	     --> (optional) name of subroutine to evaluate gradient
 *			 of fcn.  must be declared external in calling routine
 *	d2fcn	     --> (optional) name of subroutine to evaluate hessian of
 *			 of fcn.  must be declared external in calling routine
 *	state	    <--> information other than x and n that fcn,
 *			 d1fcn and d2fcn requires.
 *			 state is not modified in optif9 (but can be
 *			 modified by fcn, d1fcn or d2fcn).
 *	typsiz(n)    --> typical size for each component of x
 *	fscale	     --> estimate of scale of objective function
 *	method	     --> algorithm to use to solve minimization problem
 *			   =1 line search
 *			   =2 double dogleg
 *			   =3 more-hebdon
 *	iexp	     --> =1 if optimization function fcn is expensive to
 *			 evaluate, =0 otherwise.  if set then hessian will
 *			 be evaluated by secant update instead of
 *			 analytically or by finite differences
 *	msg	    <--> on input:  ( > 0) to inhibit certain automatic checks
 *			 on output: ( < 0) error code; =0 no error
 *	ndigit	     --> number of good digits in optimization function fcn
 *	itnlim	     --> maximum number of allowable iterations
 *	iagflg	     --> =1 if analytic gradient supplied
 *	iahflg	     --> =1 if analytic hessian supplied
 *	dlt	         --> trust region radius
 *	gradtl	     --> tolerance at which gradient considered close
 *			 enough to zero to terminate algorithm
 *	stepmx	     --> maximum allowable step size
 *	steptl	     --> relative step size at which successive iterates
 *			 considered close enough to terminate algorithm
 *	xpls(n)	    <--> on exit:  xpls is local minimum
 *	fpls	    <--> on exit:  function value at solution, xpls
 *	gpls(n)	    <--> on exit:  gradient at solution xpls
 *	itrmcd	    <--	 termination code (in 0..5 ; 0 is "perfect");
 *			see optcode() in optimize.c for meaning
 *	a(n,n)	     --> workspace for hessian (or estimate)
 *			 and its cholesky decomposition
 *	wrk(n,8)     --> workspace
 *	itncnt	    <--> iteration count
 */


    if (msg < 0)
        opterror(msg);
    if (*resultCode != 0 && (omsg&8) == 0)
        optcode(*resultCode);

    //release memory
    delete[] gpls;     // <--> on exit:  gradient at solution xpls
    delete[] a;         // a(n,n) --> workspace for hessian (or estimate)
    // and its cholesky decomposition
    delete[] wrk; // wrk(n,8)   --> workspace

    return fpls;
}


