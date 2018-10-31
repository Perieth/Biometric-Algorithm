import numpy as np
np.set_printoptions(suppress=True, precision=5)
score = 0

neuronNumber = 7 #Best set to 7

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class neuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],neuronNumber) 
        self.weights2   = np.random.rand(neuronNumber,5)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))


        self.weights1 += d_weights1
        self.weights2 += d_weights2

             
if __name__ == "__main__":

    #Insert Inputs
    X = np.array([  [0, 47, 5, 9, 32, 17, 10, 29],
    [10, 62, 4, 7, 28, 17, 6, 29],
    [-1, 46, 5, 7, 31, 16, 9, 24],
    [23, 46, 4, 9, 29, 16, 9, 22],
    [-1, 42, 3, 7, 22, 12, 7, 26],
    [2, 61, 7, 11, 29, 18, 9, 30],
    [14, 74, 5, 10, 26, 19, 5, 31],
    [1, 60, 7, 11, 31, 18, 8, 30],
    [26, 61, 6, 12, 36, 20, 11, 36],
    [-5, 55, 5, 9, 26, 16, 8, 31],
    [-1, 67, 8, 10, 40, 25, 10, 32],
    [14, 89, 7, 11, 35, 20, 7, 35],
    [0, 65, 4, 10, 29, 16, 9, 29],
    [29, 61, 6, 12, 40, 22, 12, 33],
    [-1, 71, 4, 10, 25, 8, 13, 27],
    [-2, 65, 3, 8, 29, 14, 11, 37],
    [4, 72, 3, 10, 29, 15, 10, 35],
    [-2, 65, 5, 9, 31, 15, 11, 38],
    [17, 67, 5, 11, 34, 19, 10, 37],
    [-2, 65, 4, 9, 28, 17, 10, 38],
    [-1, 56, 7, 10, 31, 13, 11, 32],
    [12, 72, 5, 9, 25, 12, 5, 32],
    [-1, 57, 5, 10, 35, 17, 9, 32],
    [4, 60, 4, 8, 24, 12, 8, 30],
    [-10, 60, 6, 11, 24, 11, 9, 28],
    [-1, 62, 4, 8, 25, 12, 9, 22],
    [29, 51, 8, 14, 38, 20, 14, 32],
    [-1, 65, 8, 11, 36, 16, 12, 28],
    [14, 79, 6, 10, 32, 16, 9, 31],
    [2, 52, 9, 11, 35, 17, 12, 27]]
                    )

    X = X/np.amax(X, axis=0)


    #insert Expected outputs
    y = np.array([  [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0]])
    nn = neuralNetwork(X,y)


    for i in range(10000):
        nn.feedforward()
        nn.backprop()
           
    print(nn.output)


    predictedFace   = np.argmax(nn.output, axis = 1)
    trueFace        = np.argmax(y, axis = 1)
    
    print(predictedFace)
    print(trueFace)

    for i in range(len(trueFace)):
        if predictedFace[i] == trueFace[i]:
            print("test " + str(i) + " is successful")
            score += 1
        else:
            print("test " + str(i) + " is unsuccessful")

    print("score is: " + str(score/len(trueFace)))





















            
        


