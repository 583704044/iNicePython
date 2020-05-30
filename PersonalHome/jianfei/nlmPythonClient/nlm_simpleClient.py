import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

class nlmMinimizer:

    def __init__(self, xInit):
        self.xInit = xInit
        self.xOuput = np.ones(self.xInit.size)
        self.code = ctypes.c_int(0)
        self.iterCount = ctypes.c_int(0)
        self.retValue = 0

        # double nlm_simple(LossFunType f, void* objectPointer, double* xInit,
        #         unsigned long n, double* xOutput, int* code, int* iterCount)
        lib = ctypes.CDLL("./libnlm_simple.so")
        self.fun = lib.nlm_simple
        self.fun.restype = ctypes.c_double
        funType = ctypes.CFUNCTYPE(ctypes.c_double,
                                   ctypes.c_void_p)
        self.fun.argtypes = [funType,
                        ctypes.c_void_p,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_ulong,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int)]

    # typedef double (*LossFunType)(void* objectPointer);
    @staticmethod
    def loss(SELF):
        # x = np.ctypeslib.as_array(x)
        # print("loss: double* x=", nlmClient.xOuput)
        a = ctypes.cast(SELF, ctypes.POINTER(nlmMinimizer))
        print('nlmMinimizer.xOutput: ', a.xOuput)
        return 100


    def __call__(self):
        # double nlm_simple(LossFunType f, void* objectPointer, double* xInit,
        # unsigned long n, double* xOutput, int* code, int* iterCount)
        cb = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)(nlmMinimizer.loss)
        self.retValue = self.fun(cb, ctypes.cast(self, ctypes.c_void_p),
                                 self.xInit,
                                 self.xInit.size,
                                 self.xOuput,
                                 ctypes.byref(self.code),
                                 ctypes.byref(self.iterCount))
    def show(self):
        print('xInit =', self.xInit)
        print('xOuput = ', self.xInit)
        print('code =', self.code)
        print('iterCount =', self.iterCount)
        print('retValue = ', self.retValue)


    @staticmethod
    def test():
        m = nlmMinimizer(np.array([1,2,3,4]))
        m()


if __name__ == '__main__':
    nlmMinimizer.test()
