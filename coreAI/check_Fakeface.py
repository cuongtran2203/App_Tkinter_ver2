import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from .model.Model import DeePixBiS
class Face_Anti_Spoofing(object):
    def __init__(self):
        self.model = DeePixBiS()
        self.model.load_state_dict(torch.load('/home/cuong/tienanh/Face_Recognition_APP-main/coreAI/face_rec_models/DeePixBiS.pth'))
        self.model.eval()

        self.tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def run(self,face_img):
        faceRegion = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
        # cv.imshow('Test', faceRegion)

        faceRegion = self.tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)
        mask, binary = self.model.forward(faceRegion)
        res = torch.mean(mask).item()
        return res

