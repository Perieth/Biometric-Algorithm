import numpy as np
from deap import base


class neuralNetwork(object):
        def __init__(self):
                #Hyperparameters
                self.inputLayerSize = 8
                self.outputLayerSize = 5
                self.hiddenLayerSize = 7

                #weights
                self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
                self.w2 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)


        def forward(self, X):
                #propagate inputs through network
                self.z2 = np.dot(X, self.W1)
                self.a2 = self.sigmoid(self.z2)
                self.z3 = np.dot(self.a2, self.W2)
                yHat = self.sigmoid(self.z3)

        def sigmoid(self, z):
                #apply sigmoid activation function
                return 1/(1+np.exp(-z))




def Main():
        with open('bigFaceFile.txt','r') as f:
            content = f.readlines()
        faceTests = [x.strip() for x in content] 
        print( faceTests[2])
Main()
