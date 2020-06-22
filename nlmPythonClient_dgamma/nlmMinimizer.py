import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import math

class nlmMinimizer:

    def __init__(self, c_print_level=0):
        self.code = ctypes.c_int(0)
        self.iterCount = ctypes.c_int(0)
        self.retValue = 1E10

        # typedef double (*LossFunType)(const double *x, unsigned long n);
        #
        # double nlm_simple(LossFunType f, double* xInit,
        #         unsigned long n, double* xOutput, int* resultCode, int* iterCount,
        #         double *c_typsize, double c_fscale, int c_print_level,
        #         int c_ndigit, double c_gradtol,
        #         int c_stepmax, double c_steptol,
        #         int c_iterlim, bool c_check_analyticals);

        self.c_fscale = 1.0
        self.c_print_level = c_print_level  # must be in {0,1,2}
        self.c_ndigit = 12
        self.c_gradtol = 1e-06
        self.c_steptol = 1e-06
        self.c_iterlim = 100
        self.c_check_analyticals = True
        # c_stepmax = max(1000 * sqrt( sum((xInit/typsize)^2) ), 1000),

        # load dynamic lib
        lib = ctypes.CDLL("./libnlm_simple.so")

        self.fun = lib.nlm_simple
        self.fun.restype = ctypes.c_double
        self.callBackType = ctypes.CFUNCTYPE(ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_ulong)
        self.fun.argtypes = [self.callBackType,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_ulong,                # unsigned long n
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.POINTER(ctypes.c_int),  # int* resultCode
                        ctypes.POINTER(ctypes.c_int),  # int* iterCount
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"), # c_typsize
                        ctypes.c_double,    # double c_fscale
                        ctypes.c_int,       # int c_print_level
                        ctypes.c_int,       # int c_ndigit
                        ctypes.c_double,    # double c_gradtol
                        ctypes.c_int,       # int c_stepmax
                        ctypes.c_double,    # double c_steptol
                        ctypes.c_int,       # int c_iterlim
                        ctypes.c_bool       # bool c_check_analyticals
                        ]

        # void optcode(int code)
        self.optcode = lib.optcode
        self.optcode.argtypes = [ctypes.c_int]


    # typedef double (*LossFunType)(const double *xOutput, unsigned long n);
    def __call__(self, xInit, lossFun):

        self.xInit = xInit
        self.lossFun = lossFun
        self.xOuput = np.ones(xInit.size, dtype=np.float64)
        self.c_typsize = np.ones(xInit.size, dtype=np.float64)

        xs = self.xInit / self.c_typsize
        s = np.sum(xs ** 2)
        self.c_stepmax = max(int(1000.0 * math.sqrt(s)), 1000)

        # double nlm_simple(LossFunType f, double* xInit,
        #         unsigned long n, double* xOutput, int* resultCode, int* iterCount,
        #         double *c_typsize, double c_fscale, int c_print_level,
        #         int c_ndigit, double c_gradtol,
        #         int c_stepmax, double c_steptol,
        #         int c_iterlim, bool c_check_analyticals);
        cb = self.callBackType(lambda x, n:
                self.lossFun(np.array([x[i] for i in range(n)])
                                )
                              )
        self.retValue = None
        # print('DEB: starting call self.fun....')
        try:
            self.retValue = self.fun(cb,
                                     self.xInit,
                                     self.xInit.size,
                                     self.xOuput,
                                     ctypes.byref(self.code),
                                     ctypes.byref(self.iterCount),
                                     self.c_typsize,
                                     self.c_fscale,
                                     self.c_print_level,
                                     self.c_ndigit,
                                     self.c_gradtol,
                                     self.c_stepmax,
                                     self.c_steptol,
                                     self.c_iterlim,
                                     self.c_check_analyticals)

            # print('DEG: end of calling self.fun...retVal=',self.retValue)
            if self.code.value == 1 or self.code.value == 2:
                return self.retValue
            # void optcode(int code)
            # {
            #     switch(code) {
            #         case 1:
            #             cout << "Relative gradient close to zero.\n";
            #             cout << "Current iterate is probably solution.\n";
            #             break;
            #         case 2:
            #             cout << "Successive iterates within tolerance.\n";
            #             cout << "Current iterate is probably solution.\n";
            #             break;
            #         case 3:
            #             cout << "Last global step failed to locate a point lower than x.\n";
            #             cout << "Either x is an approximate local minimum of the function,\n\
            # the function is too non-linear for this algorithm,\n\
            # or steptol is too large.\n";
            #             break;
            #         case 4:
            #             cout << "Iteration limit exceeded.  Algorithm failed.\n";
            #             break;
            #         case 5:
            #             cout << "Maximum step size exceeded 5 consecutive times.\n\
            # Either the function is unbounded below,\n\
            # becomes asymptotic to a finite value\n\
            # from above in some direction,\n"\
            # "or stepmx is too small.\n";
            #             break;
            #     }
            #     cout << "\n";
            # }
            self.optcode(self.code)
            return None
        except Exception as e:
            print('nlm_simple throw an exception: ', e)
            return None

    def showResult(self):
        print('xInit =', self.xInit)
        print('xOuput = ', self.xOuput)
        print('code =', self.code.value)
        self.optcode(self.code)

        print('iterCount =', self.iterCount.value)
        print('retValue = ', self.retValue)

    def getXSolution(self):
        return self.xOuput

    def getRetValue(self):
        return self.retValue

    def getIterCount(self):
        return self.iterCount.value