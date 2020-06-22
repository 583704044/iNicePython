
import numpy as np

class GammaMixedGen:

    def __init__(self, numOfComponents, shapeRange, scaleRange):

        self.k = numOfComponents
        self.shapeRange = shapeRange
        self.scaleRange = scaleRange

    def genSamples(self):


