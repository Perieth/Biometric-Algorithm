import numpy as np
import tkinter
import tkinter.messagebox
np.set_printoptions(suppress=True, precision = 7) 
score = 0

neuronNumber = 7 #Best set to 7

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

class neuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(8,neuronNumber) 
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

##class programGUI:
##
##    def __init__(self):
##         
##        self.main_window = tkinter.Tk()
##        self.main_window.title('Enter Landmark Data')
##        self.main_window.geometry("500x120")
##        self.main_window.resizable(0,0)
##
##
##        self.top_frame = tkinter.Frame(self.main_window)
##        self.mid_frame = tkinter.Frame(self.main_window)
##        self.bottom_frame = tkinter.Frame(self.main_window)
##
##        self.answerEntry = tkinter.Entry(self.mid_frame, width=50)
##        self.sumbitButton = tkinter.Button(self.bottom_frame, text='Submit', command = self.userInput)
##
##
##        self.sumbitButton.pack(side='right')
##         
##        #Pack entry
##        self.answerEntry.pack(side='right', padx=5)
##         
##        self.top_frame.pack(pady=10)
##        self.mid_frame.pack(pady=5)
##        self.bottom_frame.pack()
##
##    def userInput(self):
##        rawUserInput = self.answerEntry.get()
##        
##        print(rawUserInput)
##        userInputCrunch(rawUserInput)
##        
       

             
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
    [2, 52, 9, 11, 35, 17, 12, 27],
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
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0]])
    
    neuralResults = []
    scoreList = []
    weight1results = []
    weight2results = []
    
    for i in range(1):        
        nn = neuralNetwork(X,y)
        print("loading: " + str(i*10))
        nn.feedforward()
        for z in range(10000):
            nn.backprop()
            nn.feedforward()
        neuralResults.append(nn.output)
        weight1results.append(nn.weights1) 
        weight2results.append(nn.weights2)

        predictedFace   = np.argmax(nn.output, axis = 1)
        trueFace        = np.argmax(y, axis = 1)
        score = 0
        for z in range(len(trueFace)):
            if predictedFace[z] == trueFace[z]:
                score = score + 1
            
        scoreList.append(str(score/len(trueFace)))
    bestIndex = int(scoreList.index(max(scoreList)))
    print(scoreList)


    print(str(neuralResults[bestIndex]))
    print("score is: " + str(scoreList[bestIndex]))


    while True:
        
        userInput = [int(userInput) for userInput in input("Input landmark data: ").split()] # Input data as " 2 52 9 11 35 17 12 27 "
        InputArray = np.array([[0, 47, 5, 9, 32, 17, 10, 29],
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
    [2, 52, 9, 11, 35, 17, 12, 27],
    [2, 52, 9, 11, 35, 17, 12, 27], userInput])

        InputArray = InputArray/np.amax(InputArray, axis=0)
        layer1test = sigmoid(np.dot(InputArray, weight1results[bestIndex]))
        userOutput = sigmoid(np.dot(layer1test , weight2results[bestIndex]))
        print("  Neutral   Joyful      Sad    Surprised   Angry")
    
        print(str(userOutput[(len(userOutput)-1)]))





    






   # gui = programGUI()


















            
        


