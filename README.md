# Mapping Machine Learning Model for Classifying Handwritten Digits to an Edge Device

This repository contains a machine learning model for classifying handwritten digits and deploying it to an edge device.

## Usage

1. Run the "Digit_Classifier_Training.py" file to train and save the model as "model.tflite".

2. Then, run the "Edge_Device_Digit_Classifier.py" on your edge device to load and run the model.

   - Ensure that your edge device is connected to a camera. If not, you may need to change the arguments in the code for reading images using OpenCV (cv2).

Feel free to explore the code and customize it for your specific use case.
