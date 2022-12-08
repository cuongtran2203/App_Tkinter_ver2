import cv2 
import numpy as np
from torch_mtcnn import detect_faces
from PIL import Image
from arcface import ArcFace
import os
import gtts
import time
from core_faceRECOG import *

def camera_ID():
    cam=cv2.VideoCapture("rtsp://admin:ZTLBTF@192.168.1.21:554")
    width  = cam.get(3)  # float `width`
    height = cam.get(4)  # float `height`
    count=0
    model=Face_recognition(720,1280)
    while True :
        ret,frame=cam.read()

        if ret:
            frame=cv2.resize(frame,(1280,720))
            count+=1
            if count%30==0:
                t1=time.time()
                model.run(frame)
                print("Time processed {}".format(time.time()-t1))
                FPS=1/(time.time()-t1)
                cv2.putText(frame,"FPS :{:.3f}".format(FPS),(10,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,25,120),1)
                print("FPS :{:.3f}".format(FPS))
            cv2.imshow("Camera",frame)
            if cv2.waitKey(1) & 0xff==ord("q"):
                break

if __name__ == "__main__":
   camera_ID()