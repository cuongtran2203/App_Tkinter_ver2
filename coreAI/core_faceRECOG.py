from .face_detector import *
from .check_Fakeface import *
from arcface import ArcFace
from multiprocessing import Process
from .play_sound import *
import sys
import datetime



class Face_recognition():
    def __init__(self,height,width):
        self.face_rec=ArcFace.ArcFace()
        self.height=height
        self.width=width
        self.face_detect=FaceDetector("./coreAI/face_rec_models/ulffd_landmark.tflite",self.height,self.width)
        self.sound=Sound()
        self.check_Face=Face_Anti_Spoofing()

    def compare_2_emb(self,emb2):
        root="DB"
        list_dir=os.listdir(root)
        emb_vector=[]
        dist_min=0
        label=[]
        for file in list_dir:
            with open(os.path.join(root,file),"rb") as f :
                emb1=np.load(f)
            for emb in emb1:
                emb_vector.append([emb,file.split('.')[0]])
        min=9
        for emb_labels in emb_vector:
            
            dist_min=self.face_rec.get_distance_embeddings(emb_labels[0],emb2)
            
            if dist_min<min:
                min=dist_min
                label=emb_labels[1]
        if min>0.6:
            print(min)
            label="None"
        return label

    def run(self,frame):
        img = cv2.resize(frame, (320, 240))  # resize the images
        lb=[]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') #flipped[...,::-1].copy().astype('float32') #
        img = (img / 255) - 0.5  # normalization
        pred_bbox_pixel, pred_ldmk_pixel, pred_prob = self.face_detect.detect_face(img)
        for box,landmark in zip(pred_bbox_pixel, pred_ldmk_pixel):
            c=0
            # faces =frame[ int(box[1]) :int(box[3]), int(box[0]) : int(box[2])]
              # Obtain frame size and resized bounding box positions
            frame_height, frame_width = frame.shape[:2]
            
            x_min, x_max = [int(position * self.face_detect.resize_factors[0]) for position in box[0::2]]
            y_min, y_max = [int(position * self.face_detect.resize_factors[1]) for position in box[1::2]]  

            # Ensure box stays within the frame
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame_width, x_max), min(frame_height, y_max)
            faces=frame[y_min:y_max, x_min:x_max]
            res=self.check_Face.run(faces)
            if res < 0.5 :
            # Draw bounding box around detected object
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
                cv2.putText(frame,"Fake",(x_min, y_min-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
                print("Fake")
            else : 
            # Face Recognition
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                cv2.putText(frame,"Real",(x_min, y_min-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),1)
                print("Real")
                emb1=self.face_rec.calc_emb(faces)
                labels=self.compare_2_emb(emb1)
                if labels!="None":
                    lb.append(labels)
                    
        return frame,lb
        

        