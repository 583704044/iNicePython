//
// Created by jianfei on 20-6-3.
//

#ifndef NLM_SIMPLE_FUNCTIONINFO_H
#define NLM_SIMPLE_FUNCTIONINFO_H

#include "nlm_simple.h"

/* General Nonlinear Optimization */

class FunEntry {
public:
    double  fval;
    double  *x;      //vector of size=n

    FunEntry();
    ~FunEntry();
    void copy(double* x1, int n);
};

class FunctionInfo {
public:
    LossFunType fcall;	  //typedef double (*LossFunType)(double *xOutput, unsigned long n);
    int n;	              // length of the parameter (x) vector

    int tableCapacity;	      /* size of table to store computed function values */
    int last;	              /* the last valid element index in ptable */
    FunEntry *ptable;

public:
    FunctionInfo(LossFunType lossFunCallBack, int n, int tableCapacity);
    ~FunctionInfo();

    int lookup(const double *x, double* fval);
    void store(double fval, const double *x);

};


#endif //NLM_SIMPLE_FUNCTIONINFO_H
