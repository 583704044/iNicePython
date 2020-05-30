//
// Created by jianfei on 20-5-30.
//

#ifndef NLM_SIMPLE_NLM_SIMPLE_H
#define NLM_SIMPLE_NLM_SIMPLE_H

#ifdef __cplusplus
extern "C"
{
#endif
typedef double (*LossFunType)(void* objectPointer);
double nlm_simple(LossFunType f, void* objectPointer, double* xInit,
        unsigned long n, double* xOutput, int* code, int* iterCount);

#ifdef __cplusplus
}
#endif

#endif //NLM_SIMPLE_NLM_SIMPLE_H
