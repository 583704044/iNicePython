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

        # typedef double (*LossFunType)(double *xOutput, unsigned long n);
        # double nlm_simple(LossFunType f, double* xInit,
        #         unsigned long n, double* xOutput, int* code, int* iterCount);
        lib = ctypes.CDLL("./libnlm_simple.so")
        self.fun = lib.nlm_simple
        self.fun.restype = ctypes.c_double
        self.callBackType = ctypes.CFUNCTYPE(ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_ulong)
        self.fun.argtypes = [self.callBackType,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.c_ulong,
                        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                        ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int)]

    # typedef double (*LossFunType)(double *xOutput, unsigned long n);
    @staticmethod
    def loss(xOutput, n):
        print('nlmMinimizer.xOutput: ', xOutput, 'n: ', n)

        lst = [xOutput[i] for i in range(n)]
        for i in range(n):
            print('xOutput['+str(i)+']: ', xOutput[i])

        # xnp = np.ctypeslib.as_array(xOutput, shape=(n))
        # print('np.array from ndpointer: ', xnp)
        xnp = np.array(lst)
        print('np.array from ndpointer: ', xnp)

        return 2000


    def __call__(self):
        # double nlm_simple(LossFunType f, void* objectPointer, double* xInit,
        # unsigned long n, double* xOutput, int* code, int* iterCount)
        cb = self.callBackType (nlmMinimizer.loss)
        self.retValue = self.fun(cb,
                                 self.xInit,
                                 self.xInit.size,
                                 self.xOuput,
                                 ctypes.byref(self.code),
                                 ctypes.byref(self.iterCount))
        return self.retValue

    def show(self):
        print('xInit =', self.xInit)
        print('xOuput = ', self.xOuput)
        print('code =', self.code.value)
        print('iterCount =', self.iterCount.value)
        print('retValue = ', self.retValue)

    def getXSolution(self):
        return self.xOuput

    def getRetValue(self):
        return self.retValue

    def getIterCount(self):
        return self.iterCount.value


    @staticmethod
    def test():
        m = nlmMinimizer(np.array([1.0,2,3,4]))
        print('m call', m())
        print('...........after call...........')
        m.show()


if __name__ == '__main__':
    nlmMinimizer.test()
