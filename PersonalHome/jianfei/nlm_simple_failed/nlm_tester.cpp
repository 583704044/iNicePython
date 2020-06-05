//
// Created by jianfei on 20-6-5.
//

#include "nlm_tester.h"
#include "nlm_simple.h"
#include <iostream>
using namespace std;

//typedef double (*LossFunType)(const double *xOutput, unsigned long n);
//
//double nlm_simple(LossFunType f, double* xInit,
//        unsigned long n, double* xOutput, int* resultCode, int* iterCount,
//        double *c_typsize, double c_fscale, int c_print_level,
//        int c_ndigit, double c_gradtol,
//        int c_stepmax, double c_steptol,
//        int c_iterlim, bool c_check_analyticals);

double nlm_tester::loss(const double *xOutput, unsigned long n) {

    static  int count = 1;
    ++count;

    cout << "nlm_tester.loss" << endl;
    for(int i=0; i<n; ++i) {
        cout << xOutput[i] << ", ";
    }
    return 102.1/count;
}

void nlm_tester::testSimple() {

    double xInit[] = {1.0,2,3};
    unsigned long n = 3;
    double xOutput[] = {0,0,0};
    int resultCode = 0;
    int interCount = 0;

    double c_typsize[] = {1.0, 1, 1};
    double c_fscale = 1.0;
    int c_print_level = 0;      //in {0,1,2}

    int c_ndigit = 12;
    double c_gradtol = 1e-06;
    double c_steptol = 1e-06;
    int c_iterlim = 100;
    bool c_check_analyticals = true;
    int c_stepmax = 1000;

    cout << "start....call nlm_simple...." << endl;
    double fval = nlm_simple(nlm_tester::loss, xInit, n, xOutput,
               &resultCode, &interCount,
               c_typsize,c_fscale, c_print_level,
               c_ndigit, c_gradtol, c_stepmax,
               c_steptol, c_iterlim, c_check_analyticals);

    cout << "retValue: " << fval << endl;
    cout << "resultCode: " << resultCode << endl;
    cout << "interCount: " << interCount << endl;

    cout << "xOutput: " << endl;
    for(int i=0; i<n; ++i) {
        cout << xOutput[i] << ", ";
    }
    cout << endl;
}

int main() {
    nlm_tester::testSimple();
    return 0;
}