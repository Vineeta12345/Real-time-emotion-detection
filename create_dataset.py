# -*- coding: utf-8 -*-
"""
Created on Wed Sept 30 00:40:47 2020

@author: Vineeta
"""


import numpy as np
import imutils
import time
import cv2
import os
import math

#system libraries
import os
import sys
from threading import Timer
import shutil
import time


def create_dataset_folders(dataset_path,labels):
    for label in labels:
        dataset_folder = dataset_path+"\\"+label
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
            
def detect_face(frame, faceNet,threshold=0.5):
	# grab the dimensions of the frame and then construct a blob
	# from it
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	locs = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
			# add the face and bounding boxes to their respective
			# lists
			locs.append((startX, startY, endX, endY))

	return (locs)

def capture_face_expression(face_expression,label,dataset_path):
    if len(face_expression)!=0:
        dataset_folder = dataset_path+"\\"+label
        number_files = len(os.listdir(dataset_folder)) # dir is your directory path  
        image_path  = "%s\\%s_%d.jpg"%(dataset_folder,label,number_files)      
        cv2.imwrite(image_path, face_expression)
    
# define constant
dataset_path=os.getcwd()+"\\dataset"
face_model_path=os.getcwd()+"\\face_detector" 
labels = ["neutral","happy","sad","angry"]

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
weightsPath = os.path.sep.join([face_model_path,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


print("[INFO] Creating dataset folders...")
create_dataset_folders(dataset_path,labels)

cap = cv2.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert into gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    locs = detect_face(frame, faceNet,threshold=0.5)
    face_expression = None
    for box in locs:
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        face_expression= gray[startY:endY,startX:endX].copy()
        cv2.rectangle(gray, (startX, startY), (endX, endY),(255,255,255), 2)
        
    
    # show video stream
   # cv2.putText(gray, "N - Neutral", (10, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
    #cv2.putText(gray, "H - Happy", (10, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
    #cv2.putText(gray, "S - Sad", (10, 55),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
    #cv2.putText(gray, "Q - Quit", (10, 75),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
    #cv2.putText(gray, "A - Angry", (10, 95),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)

    cv2.imshow('EMOLY',gray)
    
    # wait for key press
    key = cv2.waitKey(1)
    if  key == ord('q'):
        break
    elif key == ord('n'):
        capture_face_expression(face_expression,"neutral",dataset_path)
        print("Neutral")
    elif key == ord('h'):
        capture_face_expression(face_expression,"happy",dataset_path)
        print("Happy")
    elif key == ord('s'):
        capture_face_expression(face_expression,"sad",dataset_path)
        print("Sad")
    elif key == ord('a'):
        capture_face_expression(face_expression,"angry",dataset_path)
        print("angry")
        

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()