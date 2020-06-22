
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

class GammaPdf:

    def __init__(self):
        # load dynamic lib
        lib = ctypes.CDLL("./libnlm_simple.so")

        # void dgammas(double *xIn, unsigned long n, double *pdfOut, double shape, double scale, int give_log);
        self.fun = lib.dgammas
        # self.fun.restype = ctypes..c_double
        # self.callBackType = ctypes.CFUNCTYPE(ctypes.c_double,
        #                                     ctypes.POINTER(ctypes.c_double),
        #                                     ctypes.c_ulong)
        self.fun.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_ulong,  # unsigned long n
                             ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_double,  # double shape
                             ctypes.c_double,  # scale
                             ctypes.c_int,  # int give_log
                             ]

    def __call__(self, xIn, shape, scale, pdfOut= None, give_log = False):
        n = xIn.shape[0]
        out = pdfOut
        if out is None:
            out = np.zeros(xIn.shape, dtype=np.float64)

        # print('xIn.shape: ', xIn.shape)
        # print('out: ', out)

        ngiveLog = 1 if give_log else 0
        try:
            # void dgammas(double *xIn, unsigned long n, double *pdfOut, double shape, double scale, int give_log);
            self.fun(xIn, n, out, shape, scale, ngiveLog)

            return out
        except Exception as e:
            print('nlm_simple.dgammas..throw an exception: ', e)
            return None