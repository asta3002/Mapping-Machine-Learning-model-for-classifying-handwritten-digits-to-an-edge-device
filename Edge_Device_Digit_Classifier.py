from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
# load and prepare the image

print("Welcome!")
print("Camera Starting Read")
#path = input("Please Enter File Name\n")    # To read image from file

#Reading image

#img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # To process image read from file
cam = cv2.VideoCapture('http://192.168.137.11:81/stream') # Setup camera
result, imag = cam.read()   # Read image from cam
print('Captured')
img = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)    # To convert the image from cam to single channel image
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)

#Pre process for image irrespective of how its inputted
img = np.array(img)
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')
img = img / 255.0

#Load Interpreter for tflite
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

#Allocate variables to get input and outputs
input_details = interpreter.get_input_details()  # return list in which each item is a dictionary with details about the input tensor. 
output_details = interpreter.get_output_details() # return list in which each item is a dictionary with details about the output tensor.



#Set input tensor to image and invoke the interpreter
interpreter.set_tensor(input_details[0]['index'],img)
interpreter.invoke()

#Get output from outpu_details and return index of max probability
output_data = interpreter.get_tensor(output_details[0]['index'])
digit = np.argmax(output_data[0])

print("Image predicted as: ",digit)
print("Thank You!")






