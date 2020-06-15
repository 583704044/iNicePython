//
// Created by jianfei on 20-6-3.
//

#include "FunctionInfo.h"
#include <float.h>
#include <iostream>
#include <cstring>
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//struct FunEntry {
//    double  fval;
//    double  *x;      //vector of size=n
//};
FunEntry::FunEntry() {
    fval = DBL_MAX;     /* initialize to unlikely parameter values float.h*/
    x=0;                //vector of size=n
}
FunEntry::~FunEntry(){
    if (x) {
//        cout << "FunEntry.x= " << x << " dctor invoked" << endl;
        delete[] x;
        x = 0;
    }
}
void FunEntry::copy(double fvalue, const double *x1, int n) {
    fval = fvalue;
    if (!x) {
        x = new double[n];
    }
    memcpy(x, x1, n * sizeof(double));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//    LossFunType fcall;	  //typedef double (*LossFunType)(double *xOutput, unsigned long n);
//    int n;	              // length of the parameter (x) vector
//
//    int tableCapacity;	      /* size of table to store computed function values */
//    int num;	              /* number of entries in the table */
//    FunEntry *ptable;

FunctionInfo::FunctionInfo(LossFunType lossFunCallBack, int n, int tableCapacity) {
    fcall = lossFunCallBack;
    this->n = n;
    this->tableCapacity = tableCapacity;
    last = -1;
    ptable = new FunEntry[tableCapacity];
}
FunctionInfo::~FunctionInfo() {
    if (ptable) {
//        cout << "FunctionInfo.pftable= " << ptable << " dctor invoked" << endl;
        delete[] ptable;
        ptable = 0;
    }
}

//    int n;	          // length of the parameter (x) vector
//
//    int FT_size;	      /* size of table to store computed function values */
//    int FT_last;	      /* Newest entry in the table */
//    FunEntry *pftable;
int FunctionInfo::lookup(const double *x, double* fval) {
    if (last < 0) {
        return -1;  //the loop-queue is empty
    }

    double *ftx;
    int j;

    int p = last;
    for(int i=0; i<tableCapacity && ptable[p].x; ++i) {
        ftx = ptable[p].x;

        for(j=0; j<n && x[j]==ftx[j]; ++j);
        if (j>=n) {
            *fval = ptable[p].fval;
            return p;
        }

        --p;
        if (p < 0) p += tableCapacity;
    }
    return -1;
}

/* Store an entry in the table of computed function values */
void FunctionInfo::store(double fval, const double *x) {
    ++last;
    if (last >= tableCapacity)
        last = 0;               //keep max 5 caches.

    ptable[last].copy(fval, x, n);

//    memcpy(ptable[last].x, x, n * sizeof(double));
}