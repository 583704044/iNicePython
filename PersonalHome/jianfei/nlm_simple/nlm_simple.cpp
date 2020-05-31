//
// Created by jianfei on 20-5-29.
//

#include <stdio.h>
#include "nlm_simple.h"

//typedef double (*LossFunType)(double *xOutput, unsigned long n);
//double nlm_simple(LossFunType f, double* xInit,
//        unsigned long n, double* xOutput, int* code, int* iterCount);

double nlm_simple(LossFunType f, double* xInit,
        unsigned long n, double* xOutput, int* code, int* iterCount) {
//return function value at the point xOutput

    //print input initial data
    int i=0;
    for(; i<n; ++i) {
        printf("nlm_simple.so: x_init[%d] is %f \n", i, xInit[i]);
    }

    //test if we can call into the python-end with toy data
    i=0;
    for(; i<n; ++i) {
        xOutput[i] = i;           //toy solution
    }

    *code = 11;
    *iterCount = 10;

    double loss = (*f)(xOutput, n);
    printf("nlm_simple.so: python loss is %f \n", loss);

    return loss;
}