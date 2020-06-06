//
// Created by jianfei on 20-6-5.
//

#ifndef NLM_SIMPLE_NLM_TESTER_H
#define NLM_SIMPLE_NLM_TESTER_H

class nlm_tester {
public:
    static void testSimple();
    static double loss(const double *x, unsigned long n);
    static double lossMatrix(const double *x, unsigned long n);

};


#endif //NLM_SIMPLE_NLM_TESTER_H
