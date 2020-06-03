//
// Created by jianfei on 20-6-3.
//

#ifndef NLM_SIMPLE_FUNCTIONINFO_H
#define NLM_SIMPLE_FUNCTIONINFO_H

#include "nlm_simple.h"

/* General Nonlinear Optimization */

class FTable {
public:
    double  fval;
    double  *x;      //vector of size=n

    FTable();
    ~FTable();
    void createX(int n);
};

class FunctionInfo {
private:
    LossFunType fcall;	  //typedef double (*LossFunType)(double *xOutput, unsigned long n);
    int n;	              // length of the parameter (x) vector

    int FT_size;	      /* size of table to store computed function values */
    int FT_last;	      /* Newest entry in the table */
    FTable *pftable;

public:
    FunctionInfo(LossFunType lossFunCallBack, int n);
    ~FunctionInfo();

    void createFTable(int FT_size=5);

};


#endif //NLM_SIMPLE_FUNCTIONINFO_H
