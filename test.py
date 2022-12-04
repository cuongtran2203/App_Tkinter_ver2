from core.face_detect import *
import cv2
cam=cv2.VideoCapture(0)
cfg = cfg_mnet
model_path="./core/weights/mobilenet0.25_Final.pth"
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, model_path, True)
net.eval()
print('Finished loading model!')
device ="cpu"
net = net.to(device)
cam=cv2.VideoCapture(0)
    # testing begin
while True:
        _,img_raw=cam.read()
        img,_=process_images(net,cfg,img_raw)
        cv2.imshow("d",img)
        if cv2.waitKey(1) & 0xff==ord("q"):
            break
