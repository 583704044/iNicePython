//
// Created by jianfei on 20-6-3.
//

#include "FunctionInfo.h"
#include <float.h>
#include <iostream>
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//struct FTable {
//    double  fval;
//    double  *x;      //vector of size=n
//};
FTable::FTable() {
    fval = DBL_MAX;     /* initialize to unlikely parameter values float.h*/
    x=0;                //vector of size=n
}
FTable::~FTable(){
    if (x) {
        cout << "FTable.x= " << x << " dctor invoked" << endl;
        delete[] x;
        x = 0;
    }
}
void FTable::createX(int n) {
    if (x) {
        cout << "WARNING: x= " << x << " is not null..unexpected..release its memory." << endl;
        delete[] x;
        x = 0;
    }
    x = new double[n];
    for (int i = 0; i < n; ++i) {
        x[i] = DBL_MAX; /* initialize to unlikely parameter values float.h*/
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//class FunctionInfo {
//private:
//    LossFunType fcall;	  //typedef double (*LossFunType)(double *xOutput, unsigned long n);
//    int n;	              // length of the parameter (x) vector

//    int FT_size;	      /* size of table to store computed function values */
//    int FT_last;	      /* Newest entry in the table */
//    FTable *pftable;

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

    FTable *table= new FTable[FT_size];
    for (int i = 0; i < FT_size; i++) {
        table[i].createX(n);
    }
    this->pftable = table;
    FT_last = -1;
}