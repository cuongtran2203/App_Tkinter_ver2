import os 
import cv2 
from torch_mtcnn import detect_faces
from PIL import Image
from arcface import ArcFace
import numpy as np
import sys
from .face_detector import *
import time
face_rec=ArcFace.ArcFace()
face_detect=FaceDetector("./coreAI/face_rec_models/ulffd_landmark.tflite",1080,1920)
count=0
emb1=[]
def face_detection(frame):
    img = cv2.resize(frame, (320, 240))  # resize the images
    lb=[]
    faces_list=[]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') #flipped[...,::-1].copy().astype('float32') #
    img = (img / 255) - 0.5  # normalization
    pred_bbox_pixel, pred_ldmk_pixel, pred_prob = face_detect.detect_face(img)
    for box,landmark in zip(pred_bbox_pixel, pred_ldmk_pixel):
        c=0
        # faces =frame[ int(box[1]) :int(box[3]), int(box[0]) : int(box[2])]
            # Obtain frame size and resized bounding box positions
        frame_height, frame_width = frame.shape[:2]
        
        x_min, x_max = [int(position * face_detect.resize_factors[0]) for position in box[0::2]]
        y_min, y_max = [int(position * face_detect.resize_factors[1]) for position in box[1::2]]  

        # Ensure box stays within the frame
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)
        faces=frame[y_min:y_max, x_min:x_max]
        faces_list.append([y_min,y_max,x_min,x_max])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    return frame,faces_list
        