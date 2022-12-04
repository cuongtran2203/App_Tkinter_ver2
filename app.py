from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from core.face_detect import *
import cv2
import numpy as np
from datetime import datetime
from tkinter import ttk
import os
from arcface import ArcFace
from tkinter import messagebox
face_rec=ArcFace.ArcFace()
#define model face_detect
cfg = cfg_mnet
model_path="C:/Users/cuong/Desktop/APP_tkinter/core/weights/mobilenet0.25_Final.pth"
# net and model
net = RetinaFace(cfg=cfg, phase = 'test')
net = load_model(net, model_path, True)
net.eval()
print('Finished loading model!')
device ="cpu"
net = net.to(device)
cam=cv2.VideoCapture(0)
def get_emb(img):
    img_pro,faces_list=process_images(net,cfg,img)
    emb_list=[]
    for face in faces_list:
        emb=face_rec.calc_emb(face)
        emb_list.append(emb)
    return emb_list
def calc_distance(faces,emb1):
    emb = face_rec.calc_emb(faces)    
    dis=face_rec.get_distance_embeddings(emb, emb1)
    return dis

def save_db():
    label=label_name_1.get()
    team=label_Team_name_1.get()
    if label=="":
        messagebox.showerror("Infor","Please enter  ID")
    else:
        try :
            img=cv2.imread("face.jpg")
            emb=get_emb(img)
            with open("DB/"+label+"."+team+".npy","wb") as f :
                np.save(f,emb)
            messagebox.showinfo("Infor","Added to database")  
        except :
            print("watiiing")
            messagebox.showerror("Infor","Don't capture image")  
        os.remove("face.jpg")
def hide():
    notebook.hide(1)
def show():
    notebook.add(frame2,text="Tab2")
def select_apply():
    notebook.select(1)
def select_back():
    notebook.select(0)
def predict(faces):
    MIN=9
    label=''
    team=''
    list_face=os.listdir("DB")
    for file in list_face:
        file_path=os.path.join("DB",file)
        with open(file_path,"rb") as f:
            emb_list=np.load(f)
        dis=calc_distance(faces,emb_list[0])
        if dis<MIN:
            MIN=dis
            label=file.split(".")[0]
            team=file.split(".")[1]
            
    if MIN>0.6 :
        label="None"
        team="None"
    return label,team

   
faces_=None
def snapshot():
    global faces_
    # Get a frame from the video source
    ret, frame = cam.read()
    if ret:
        img,face_list=process_images(net,cfg,frame)
        cv2.imwrite("face.jpg",img)
        faces=cv2.resize(face_list[0],(320,240))
        faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
        face_ = ImageTk.PhotoImage(Image.fromarray(faces))
        canvas2_2.create_image(0,0, image = face_, anchor=NW)
        messagebox.showinfo("Infor","Capture Image Successfully")

def update_frame1():
    global canvas1,canvas2,photo,count,canvas4,label_Time1
    # Doc tu camera
    ret, frame = cam.read()
    count+=1
    Min=9
    # Ressize
    
    frame = cv2.resize(frame,(640,480))

    img,face_list=process_images(net,cfg,frame)
    if len(face_list)>0:
        faces=face_list[0]
        try:
            label,team=predict(faces)
            if label is not None  :
                label_name.configure(text=label,font=('Helvetica 15 '))
                cv2.imwrite("face.png",faces)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                label_Team_name.configure(text=team,font=('Helvetica 15 '))
                canvas4.itemconfig(text_id,text="ID :{}\nTime : {}".format(label,current_time))
                label_Time1.configure(text=str(current_time),font=('Helvetica 15'))
        except :
            pass
        # img2=ImageTk.PhotoImage(Image.fromarray(im_face))   
    else :
        canvas4.itemconfig(text_id,text="ID: None")
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))

    # Show
    canvas1.create_image(0,0, image = photo, anchor=NW)
   
    frame1.after(15, update_frame1)
def update_frame2():
    global canvas1_2,canvas2_2,photo_2,canvas4_2
    # Doc tu camera
    ret, frame = cam.read()
    # Ressize
    frame = cv2.resize(frame,(640,480))
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    photo_2 = ImageTk.PhotoImage(image=Image.fromarray(frame))
    # Show
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    canvas1_2.create_image(0,0, image = photo_2, anchor=NW)
    canvas4_2.itemconfig(text_id,text="Time : {}".format(current_time))
    frame2.after(15, update_frame2)

if __name__=="__main__":
    count=0
    #APP
    root = Tk()
    root.geometry("940x540")
    root.title("Face Recogniton System")
    # root.attributes('-fullscreen',True)
    notebook=ttk.Notebook(root)
    notebook.place(x=0,y=0)
    frame1=Frame(notebook,width=940,height=540,bg="#0062ff")
  
    frame1.pack(fill="both",expand=1)
    canvas1 = Canvas(frame1, width = 640, height = 480,bg="red")
    canvas1.place(x=0,y=0)
    canvas2=Canvas(frame1,width=300,height=240,bg="#0062ff")
    canvas2.place(x=640,y=0)
    img1 = ImageTk.PhotoImage(Image.open('app.jpg').resize((320,240)))
    canvas2.create_image(0,0,image = img1, anchor=NW)
    canvas3=Canvas(frame1,width=300,height=300,bg="#0062ff")
    label_ID=Label(canvas3,text="ID",fg="black",font=('Helvetica 15'))
    label_ID.pack(fill="both",expand=1,padx=60,pady=9)
    label_name=Label(canvas3,fg="black")
    label_name.pack(fill="both",expand=1,padx=60,pady=9)
    label_Team=Label(canvas3,text="Team",fg="black",font=('Helvetica 15') )
    label_Team.pack(fill="both",expand=1,padx=60,pady=9)
    label_Team_name=Label(canvas3,text="",fg="black")
    label_Team_name.pack(fill="both",expand=1,padx=60,pady=9)
    label_Time=Label(canvas3,text="Time",fg="black",font=('Helvetica 15'))
    label_Time.pack(fill="both",expand=1,padx=60,pady=8)
    now1 = datetime.now()
    current_time1 = now1.strftime("%H:%M:%S")
    label_Time1=Label(canvas3,fg="black")
    label_Time1.pack(fill="both",expand=1,padx=60,pady=10)
    canvas3.place(x=640,y=240)
    canvas4=Canvas(frame1,width=640,height=40,bg="white")
    canvas4.place(x=0,y=480)
    text_id=canvas4.create_text(110,18,text="", fill="black", font=('Helvetica 12 bold'))
    ####TAB2
    frame2=Frame(notebook,width=940,height=540,bg="red")
    frame2.pack(fill="both",expand=1,padx=100,pady=10)
    canvas1_2 = Canvas(frame2, width = 640, height = 480,bg="red")
    canvas1_2.place(x=0,y=0)
    canvas2_2=Canvas(frame2,width=300,height=240,bg="blue")
    canvas2_2.place(x=640,y=0)
    canvas3_2=Canvas(frame2,width=300,height=300,bg="green")
    canvas3_2.place(x=640,y=240)
    canvas4_2=Canvas(frame2,width=640,height=40,bg="white")
    canvas4_2.place(x=0,y=480)
    
    times_str=canvas4_2.create_text(80,18,text="",fill="black", font=('Helvetica 12 bold'))
    label_ID_1=Label(canvas3_2,text="ID",fg="black")
    label_ID_1.pack(fill="both",expand=1,padx=105,pady=10)
    label_name_1=Entry(canvas3_2,fg="black")
    label_name_1.pack(fill="both",expand=1,padx=105,pady=10)
    label_Team_1=Label(canvas3_2,text="Team",fg="black" )
    label_Team_1.pack(fill="both",expand=1,padx=105,pady=10)
    label_Team_name_1=Entry(canvas3_2,fg="black")
    label_Team_name_1.pack(fill="both",expand=1,padx=105,pady=10)
    notebook.add(frame1,text="Recogniton")
    notebook.add(frame2,text="Manager")
    count=0
    # button = tk.Button(frame1, text="Apply User", bg='White', fg='Black',
    #                             command=select_apply)
    # button.place(x=850,y=495)
    button2=Button(frame2, text="Back",bg="white", fg="black",command=select_back)
    button2.place(x=850,y=450)
    button3=Button(frame2, text="Add User",bg="white", fg="black",command=save_db)
    button3.place(x=750,y=450)
    button4=Button(frame2, text="Capture Image",bg="white", fg="black",command=snapshot)
    button4.place(x=650,y=450)

    

     
   
   
    update_frame1()
    update_frame2()
    root.mainloop()
        

