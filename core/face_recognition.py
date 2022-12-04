from .face_detect import *
import torch 
from arcface import ArcFace
face_rec = ArcFace.ArcFace()
cfg = cfg_mnet
model_path="C:/Users/cuong/Desktop/APP_tkinter/core/weights/mobilenet0.25_Final.pth"
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, model_path, True)
net.eval()
print('Finished loading model!')
device ="cpu"
net = net.to(device)

def get_emb(img):
    img_pro,faces_list=process_images(net,cfg,img)
    emb_list=[]
    for face in faces_list:
        emb=face_rec.calc_emb(face)
        emb_list.append(emb)
    return emb_list