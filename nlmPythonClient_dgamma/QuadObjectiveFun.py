
import numpy as np

from nlmMinimizer import nlmMinimizer

class QuadObjectiveFun:
# An example of customer classes that use nlmMinimizer

    sA = np.array([[5.0, 5.0], [5.0, 12.5]])  # A is positive definite matrix
    sb = np.array([2.0, 2.0])

    def __init__(self):
        self.A = np.array([[5.0, 5.0], [5.0, 12.5]])  # A is positive definite matrix
        self.b = np.array([2.0, 2.0])

    def lossFun(self, x):
        # define the loss function which will be callbacked by nlmMinimizer
        loss = 0.5 * (self.A.dot(x)).dot(x) + self.b.dot(x)

        print('debugInfo...GMT.loss: ', loss)
        return loss

    @staticmethod
    def stLossFun(x):
        # define the loss function which will be callbacked by nlmMinimizer
        loss = 0.5 * (QuadObjectiveFun.sA.dot(x)).dot(x) + QuadObjectiveFun.sb.dot(x)

        print('debugInfo_stdLossFun...GMT.loss: ', loss)
        return loss

    @staticmethod
    def test():
        q = QuadObjectiveFun()

        m = nlmMinimizer()
        retValue = m(np.array([1.0, 2.0]), q.lossFun)  # ok.
        # retValue = m(np.array([1.0, 2.0]), QuadObjectiveFun.stLossFun) # also ok.

        print('...........after call...........')
        print('m call: ok=', retValue != None, ', retVal=', retValue)
        print('m call: xSolution=', m.getXSolution())

        print('...........summary info...........')
        m.showResult()

if __name__ == '__main__':
    QuadObjectiveFun.test()