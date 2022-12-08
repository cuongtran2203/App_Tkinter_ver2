import os
import gtts
import time
from multiprocessing import Process
from threading import Thread
import sys
class Sound(object):
    def __init__(self) :
        self.v=1
        
    def sound(self,label):
        try:
            tts=gtts.gTTS("xin chào"+label,lang="vi")
            tts.save("hello.mp3")
            os.system("mpg123 hello.mp3")
            # os.remove("hello.mp3")
        except Exception as e:
            print("file is deleted")
    def run(self,label):
        process=Thread(target=self.sound, args=(label,))
        process.start()
        if 0xff==ord("q"):
            sys.exit()