//
// Created by jianfei on 20-5-30.
//

#ifndef NLM_SIMPLE_NLM_SIMPLE_H
#define NLM_SIMPLE_NLM_SIMPLE_H

#ifdef __cplusplus
extern "C"
{
#endif

void optcode(int code);

typedef double (*LossFunType)(const double *xOutput, unsigned long n);

double nlm_simple(LossFunType f, double* xInit,
        unsigned long n, double* xOutput, int* resultCode, int* iterCount,
        double *c_typsize, double c_fscale, int c_print_level,
        int c_ndigit, double c_gradtol,
        int c_stepmax, double c_steptol,
        int c_iterlim, bool c_check_analyticals);
//c_typsize = [1,1,..1], length=length(xInit)
// c_fscale = 1, c_print_level = 0 in {0,1,2},
// c_ndigit = 12, c_gradtol = 1e-06,
// c_stepmax = max(1000 * sqrt( sum((xInit/typsize)^2) ), 1000),
// c_steptol = 1e-06,
// c_iterlim = 100,
// c_check_analyticals = TRUE

#ifdef __cplusplus
}
#endif

#endif //NLM_SIMPLE_NLM_SIMPLE_H
