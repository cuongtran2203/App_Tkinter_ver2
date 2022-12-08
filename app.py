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
from save_db import put_data
from threading import Thread
from core.play_sound import *
from multiprocessing import Process
from coreAI.core_app import face_detection
from config_name import NAME
from coreAI.create_db import create_user
face_rec=ArcFace.ArcFace()
#define model face_detect
sound=Sound()
def get_emb(img):
    img_pro,faces_list=face_detection(img)
    emb_list=[]
    for face in faces_list:
        emb=face_rec.calc_emb(face)
        emb_list.append(emb)
    return emb_list
def calc_distance(faces,emb1):
    emb = face_rec.calc_emb(faces)    
    dis=face_rec.get_distance_embeddings(emb, emb1)
    return dis
cam=cv2.VideoCapture('rtsp://admin:XEJVQU@192.168.1.3:554')
def save_db():
    label=label_name_1.get()
    team=label_Team_name_1.get()
    if label=="" or team =="":
        messagebox.showerror("Infor","Please enter  ID")
    else:
        try :
            img=cv2.imread("face.jpg")
            ID=label+"."+team
            create_user(img,ID)
            messagebox.showinfo("Infor","Added to database")  
        except :
            print("watiiing")
            messagebox.showerror("Infor","Don't capture image")  
        os.remove("face.jpg")
        label_name_1.configure(text="")
        label_Team_name_1.configure(text="")
def hide():
    notebook.hide(1)
def show():
    notebook.add(frame2,text="Tab2")
def select_apply():
    notebook.select(1)
    update_frame2()
    frame1.after_cancel(update_frame1)
def select_back():
    notebook.select(0)

   
def snapshot():
    # Get a frame from the video source
    ret, frame = cam.read()
    label = label_name_1.get()
    team = label_Team_name_1.get()
    if label =="" and team=="":
        messagebox.showwarning("Infor","Please enter a team name or ID user")
    else :
        if ret:
            img,face_list=face_detection(frame)
            cv2.imwrite("face.jpg",img)
            faces=cv2.resize(face_list[0],(320,240))
            faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
            face_ = ImageTk.PhotoImage(Image.fromarray(faces))
            canvas2_2.create_image(0,0, image = face_, anchor=NW)
            messagebox.showinfo("Infor","Capture Image Successfully")
def update_sound():
    name=label_name.cget("text")
    if name!="" or name!="None" or name!=" ":
        if name in NAME.keys():
            names=NAME[name]
            thread=Thread(target=sound.sound,args=(names,))
            thread.start()
    else :
        pass
    frame1.after(2500,update_sound)
def update_frame1():
    global canvas1,photo,count,label_name,label_Time1
    # Doc tu camera
    ret, frame = cam.read()
    count+=1
    Min=9
    try:
        img = cv2.resize(frame,(1920,1080))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        # Show
        canvas1.create_image(0,0, image = photo, anchor=NW) 
        if count%10==0:
            thread1=Thread(target=face_rc,args=(frame,))
            
            thread1.start()
        
            

        frame1.after(25, update_frame1)
    except :
        print("Error")


def update_frame2():
    global canvas1_2,canvas2_2,photo_2
    # Doc tu camera
    ret, frame = cam.read()
    # Ressize
    frame = cv2.resize(frame,(1920,1080))
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    photo_2 = ImageTk.PhotoImage(image=Image.fromarray(frame))
    # Show
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    canvas1_2.create_image(0,0, image = photo_2, anchor=NW)
    # canvas4_2.itemconfig(text_id,text="Time : {}".format(current_time))
    frame2.after(500, update_frame2)
    # frame1.after_cancel(update_frame1)


def predict(faces):
    MIN=9
    label=''
    team=''
    emb_vector=[]
    list_face=os.listdir("DB")
    for file in list_face:
        file_path=os.path.join("DB",file)
        with open(file_path,"rb") as f:
            emb_list=np.load(f)
        for emb in emb_list:
            emb_vector.append([emb,file.split(".")[0],file.split(".")[1]])
    for emb_labels in emb_vector:
        dis=calc_distance(faces,emb_labels[0])
        if dis<MIN:
            MIN=dis
            label=emb_labels[1]
            team=emb_labels[2]    
    if MIN>0.65 :
        label="None"
        team="NA"
    return label,team
def face_rc(frame):
    frame = cv2.resize(frame,(640,480))
    img,face_list=face_detection(frame)
    if len(face_list)>0:
        for face in face_list:
            try:
                if face.shape[0]>50 and face.shape[1]>50:

                    label,team=predict(face)
                    if label is not None  :
                        label_name.configure(text=label)
                        if label !="None":
                            put_data(label)
                        # cv2.imwrite("face.png",face)
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        label_Team_name.configure(text=team)
                        # canvas4.itemconfig(text_id,text="ID :{}\nTime : {}".format(label,current_time))
                        label_Time1.configure(text=str(current_time))
                    else :
                        label_name.configure(text="")
                        label_Team_name.configure(text="")
            except :
                pass

    # else :
    #     canvas4.itemconfig(text_id,text="ID: None")
if __name__=="__main__":
    count=0
    #APP
    root = Tk()
    root.geometry("1920x1080")
    root.title("Face Recogniton System")
    root.attributes('-fullscreen',True)
    notebook=ttk.Notebook(root)
    notebook.place(x=0,y=0)
    frame1=Frame(notebook,width=1920,height=1080,bg="white")
    frame1.place(x=0,y=0)
    
    canvas1 = Canvas(frame1, width = 1920, height = 1080, borderwidth=1, relief="solid")
    # canvas1.pack(side=LEFT, fill="both", expand=1)
    canvas1.place(x=0, y=0)

    
    
    canvas3=Canvas(frame1,width=384,height=200,bg="white")
    # canvas3.pack(side=BOTTOM, anchor=SE, fill="x", expand=0)
    canvas3.place(x=1536,y=0)

    # Id area
    label_ID=Label(canvas3,text="ID",fg="black",font=('Helvetica 20'),bg="white")
    label_ID.place(x=10,y=20)

    label_name=Label(canvas3,fg="black", font=("Helvetica 20"), bg="white")
    label_name.place(x=80,y=20)

    # Team work area
    label_Team=Label(canvas3,text="Team",fg="black",font=('Helvetica 20'),bg="white")
    label_Team.place(x=10, y=50)

    label_Team_name=Label(canvas3,text="",fg="black", font=('Helvetica 20'),bg="white")
    label_Team_name.place(x=80, y=50)

    # Time show area
    label_Time=Label(canvas3,text="Time",fg="black",font=('Helvetica 20'),bg="white")
    label_Time.place(x=10, y=80)

    now1 = datetime.now()
    current_time1 = now1.strftime("%H:%M:%S")
    label_Time1=Label(canvas3,fg="black", font=("Helvetica 20"), bg="white")
    label_Time1.place(x=80, y=80)

    button1=Button(canvas3, text="Add User",bg="white", fg="black",command=select_apply, width=30, font=('Helvetica 15 bold'))
    button1.place(x=20, y=150)

   


    ####TAB2
    frame2=Frame(notebook,width=1920,height=1080,bg="white")
    frame2.pack(fill="both",expand=1,padx=100,pady=10)
    # Camera canvas
    canvas1_2 = Canvas(frame2, width = 1920, height = 1080, bg="red")
    canvas1_2.place(x=0,y=0)
    # Face capture area
    canvas2_2=Canvas(frame2,width=384,height=300,bg="blue")
    canvas2_2.place(x=1536,y=0)

    canvas3_2=Canvas(frame2,width=384,height=300,bg="white")
    canvas3_2.place(x=1536,y=300)
    
    # times_str=canvas4_2.create_text(80,18,text="",fill="black", font=('Helvetica 12 bold'))
    label_ID_1=Label(canvas3_2,text="ID",fg="black", bg="white", font=('Helvetica 12 bold'))
    label_ID_1.place(relx=0.5,rely=0.1,anchor=CENTER)
    label_name_1=Entry(canvas3_2,fg="black")
    label_name_1.place(relx=0.5,rely=0.2,anchor=CENTER)

    label_Team_1=Label(canvas3_2,text="Team",fg="black", bg="white", font=('Helvetica 12 bold'))
    label_Team_1.place(relx=0.5,rely=0.3,anchor=CENTER)
    label_Team_name_1=Entry(canvas3_2,fg="black")
    label_Team_name_1.place(relx=0.5,rely=0.4,anchor=CENTER)

    button2=Button(canvas3_2, text="Capture Image",bg="white", fg="black",command=snapshot, width=15, font=('Helvetica 10 bold'))
    button2.place(relx=0.5,rely=0.5,anchor=CENTER)

    button3=Button(canvas3_2, text="Add User",bg="white", fg="black",command=save_db, width=15, font=('Helvetica 10 bold'))
    button3.place(relx=0.5,rely=0.6,anchor=CENTER)

    button4=Button(canvas3_2, text="Back",bg="white", fg="black",command=select_back, width=15, font=('Helvetica 10 bold'))
    button4.place(relx=0.5,rely=0.7,anchor=CENTER)
    

    notebook.add(frame1,text="Recognition")
    notebook.add(frame2,text="Manager")
    count=0


    

     
   
   
    update_frame1()
    update_sound()
    # update_frame2()
    root.mainloop()
        

