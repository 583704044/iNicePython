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
        cout << "FunEntry.x= " << x << " dctor invoked" << endl;
        delete[] x;
        x = 0;
    }
}
void FunEntry::copy(double *x1, int n) {
    if (!x) {
        x = new double[n];
    }
    memcpy(x, x1, n * sizeof(double));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//class FunctionInfo {
//private:
//    LossFunType fcall;	  //typedef double (*LossFunType)(double *xOutput, unsigned long n);
//    int n;	              // length of the parameter (x) vector

//    int FT_size;	      /* size of table to store computed function values */
//    int FT_last;	      /* Newest entry in the table */
//    FunEntry *pftable;

FunctionInfo::FunctionInfo(LossFunType lossFunCallBack, int n) {
    fcall = lossFunCallBack;
    this->n = n;

    FT_size = 0;
    FT_last = -1;
    pftable = 0;
}
FunctionInfo::~FunctionInfo() {

    if (pftable) {
        cout << "FunctionInfo.pftable= " << pftable << " dctor invoked" << endl;
        delete[] pftable;
        pftable = 0;
    }
}



void FunctionInfo::createFTable(int FT_size/*=5*/) {
    this->FT_size = FT_size;

    this->pftable= new FunEntry[FT_size];
    FT_last = -1;
}

//    int n;	          // length of the parameter (x) vector
//
//    int FT_size;	      /* size of table to store computed function values */
//    int FT_last;	      /* Newest entry in the table */
//    FunEntry *pftable;
int FunctionInfo::lookup(const double *x) {

    double *ftx;
    int ind, j;
    for (int i = 0; i < FT_size; i++) {
        ind = (FT_last - i) % FT_size;
        /* why can't they define modulus correctly */
        if (ind < 0) ind += FT_size;

        ftx = pftable[ind].x;
        if (!ftx) continue;

        for(j=0; j<n && x[j]==ftx[j]; ++j);
        if (j>=n) return ind;
    }
    return -1;
}

/* Store an entry in the table of computed function values */
void FunctionInfo::store(double fval, const double *x) {
    int ind = (++(FT_last)) % FT_size;   //keep max 5 caches.

    pftable[ind].fval = fval;
    memcpy(pftable[ind].x, x, n * sizeof(double));
}