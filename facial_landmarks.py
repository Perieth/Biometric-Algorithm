# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
 
 
import tkinter
import tkinter.messagebox
from tkinter import filedialog
 
 
import json
 
import matplotlib.pyplot as plt
 
class ProgramGUI:
 
 
    def __init__(self):
         
        self.main_window = tkinter.Tk()
        self.main_window.title('Enter image')
        self.main_window.geometry("500x120")
        self.main_window.resizable(0,0)
 
 
        #Frame creation
        self.top_frame = tkinter.Frame(self.main_window)
        self.mid_frame = tkinter.Frame(self.main_window)
        self.bottom_frame = tkinter.Frame(self.main_window)
      
        #Entry creation
        self.answerEntry = tkinter.Entry(self.mid_frame, width=50)
 
        #Button creation
        self.sumbitButton = tkinter.Button(self.bottom_frame, text='Submit', command=self.setImage)
        self.chooseButton = tkinter.Button(self.mid_frame, text='Browse', command=self.chooseFile)
 
        #Pack buttons
        self.sumbitButton.pack(side='right')
        self.chooseButton.pack(side='right')
         
        #Pack entry
        self.answerEntry.pack(side='right', padx=5)
         
        self.top_frame.pack(pady=10)
        self.mid_frame.pack(pady=5)
        self.bottom_frame.pack()
 
        tkinter.mainloop()
              
    def chooseFile(self):
        self.main_window.sourceFile = ''
        self.main_window.sourceFile = filedialog.askopenfilename(parent=self.main_window, initialdir= os.getcwd(), title='Select a file to analyse:')
        self.answerEntry.delete(0, 'end')
        self.answerEntry.insert(0,self.main_window.sourceFile)
 
             
    def main(self,image_variable):
             
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
 
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print ("[*] Successfully loaded predictor file")
 
        #load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_variable)
        print ("[*] Successfully obtained image")
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
        # detect faces in the grayscale image
        rects = detector(gray, 2)
    
        # loop over the face detections
        # rects = data structure containing the cords for each face box.
        # i = the index of the current face
        # rect = cords for current face
        for (i, rect) in enumerate(rects): #For each face in the picture
            # determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = self.getFaceLandmarks(gray,rect,predictor) #This extracts faces from the image and stores in 'shape'
 
            facialFeatures = self.facialDataParser(shape)
             
            self.drawImage(shape,rect, image, i) #This draws image (with facial data) to screen
 


            mouthDeets = facialFeatures["mouth"]
            p63 = mouthDeets[63]
            p67 = mouthDeets[67]
            p55 = mouthDeets[55]
            p49 = mouthDeets[49]
            p52 = mouthDeets[52]
            p58 = mouthDeets[58]
            mouthOpening = ( p67[1] - p63[1])
            mouthWidth = ( p55[0] - p49[0])
            topLipThickness = (p63[1] - p52[1])
            bottomLipThickness = (p58[1] - p67[1])

            #IMPORTANT for some reason points for eye and eyebrow data are wrong i.e p27 in p22, p18 in p23 
            eyes = {}
            eyes = facialFeatures['left_eyebrow'].copy()   
            eyes.update(facialFeatures['right_eyebrow'])    
            eyes.update(facialFeatures['right_eye'])
            eyes.update(facialFeatures['left_eye'])  ## Combines all eye dict into one dict

            p42 = eyes[42]
            p20 = eyes[20]
            p25 = eyes[25]
            p47 = eyes[47]
            p21 = eyes[21]
            p39 = eyes[39]
            p45 = eyes[45]
            p39 = eyes[39]
            p41 = eyes[41]
            p44 = eyes[44]
            p48 = eyes[48]
            p22 = eyes[27] # to counteract flipped data
            p23 = eyes[18] # this as well
            eyebrowRaiseDistance = (p42[1] - p20[1])
            upperEyelidToEyebrow = (p39[1] - p21[1])
            upperEyelidToLowerEyelid = (p41[1] - p39[1])
            eyebrowDistance = (p23[0] - p22[0])

            facialExpression = [image_variable[17]] #assuming picture is located in I:/FacesDB/ and using FacesDB naming system
                                                 #FaceDB naming system e.g. s001-0Y
                                                 #Values for Y 0 = neutral, 1 = happy, 2 = sad, 3 = surprised, 4 = angry

            

            usefulData = [ mouthOpening, mouthWidth, topLipThickness, bottomLipThickness,
                           eyebrowRaiseDistance, upperEyelidToEyebrow, upperEyelidToLowerEyelid,
                           eyebrowDistance]

            
            usefulFile = open("bigFaceFile.txt", 'a')                       
            usefulFile.write(str(usefulData))           
            usefulFile.write(",\n")
            usefulFile.close()

            print(facialExpression)
            otherUsefulFile = open("awnserSheet.txt", 'a') 
            otherUsefulFile.write(str(facialExpression))
            otherUsefulFile.write(",\n")            
            otherUsefulFile.close



            
             
            #self.showGraph(facialFeatures, "mouth")
 
 
    # get image path from the user
    def setImage(self):
        self.image_variable = self.answerEntry.get()
        print("[*] Successfully selected image")
        self.main(self.image_variable)
 
    def getFaceLandmarks(self, gray, rect,predictor):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape
     
    def drawImage(self, shape, rect, image, i):
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
 
        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in self.getComponent(shape, "mouth"):
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        for (x, y) in self.getComponent(shape, "right_eyebrow"):
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in self.getComponent(shape, "left_eyebrow"):
                cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        for (x, y) in self.getComponent(shape, "right_eye"):
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in self.getComponent(shape, "left_eye"):
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in self.getComponent(shape, "nose"):
                cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
        for (x, y) in self.getComponent(shape, "jaw"):
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
 
        # show the output image with the face detections + facial landmarks
        print ("[*] Successfully analysed image \n[*] Now displaying results \n-------------------------------------")
        cv2.imshow("Output", image)
         
    def facialDataParser(self, shape):
        ######## Example formatting: facialFeatures = {"mouth": [[1, 3], [1, 4], [3, 6]], "right_eyebrow": [[54, 32], [21, 31], [32, 76]]}
        faceComponents = ["jaw","left_eyebrow","right_eyebrow","nose","left_eye","right_eye","mouth"]
        facialFeatures = {}
        index = 0
        for component in faceComponents:
            coordList = []
            cordDict = {}
            for (x, y) in self.getComponent(shape, component):
 
                 
                index = index + 1
 
                cordDict[index] = coord = [x.tolist(), y.tolist()]
                    
            facialFeatures[str(component)] = cordDict
        return facialFeatures

    def getFullFace(self, shape):
        face = shape
        return shape
     
    def getComponent(self, shape, component):
        (j, k) = face_utils.FACIAL_LANDMARKS_68_IDXS[component]
        data = shape[j:k]
        return data
 
    def showGraph(self, facialFeatures, feature):
        xList = []
        yList = []
        (s, e) = face_utils.FACIAL_LANDMARKS_68_IDXS[feature]
        s += 1
        for i in range (s, e):
            xList.append(-facialFeatures[feature][i][0]/100)
            yList.append(-facialFeatures[feature][i][1]/100)
                 
        print('x: ' + str(xList))
        print('y: ' + str(yList))
 
        plt.plot(xList, yList, 'ro')
 
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph')
        plt.grid(False)
        plt.show()
         
print ("This program is adapted from the original program @ www.pyimagesearch.com/")
print ()
print ("Debug:")
gui = ProgramGUI()


