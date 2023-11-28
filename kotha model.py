import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from normalization import word_normalization, letter_normalization
from page import*
from words import* 
from helpers import implt, resize
from tfhelpers import Model
import prediction_helper

#from ocr.datahelpers2 import idx2word
from segment2 import segment
from PIL import Image as imae
from PIL import ImageTk 
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter.messagebox import showinfo
from tkinter import filedialog
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

windo = Tk()
windo.configure(background='white')
windo.title("Letter Recognition")
model = load_model('model12.h5')
windo.geometry('420x620')

windo.resizable(0,0)
load = imae.open("Telugu.jpg")
load = load.resize((300, 300))
render = ImageTk.PhotoImage(load)
labe = Label(windo,image=render)
labe.place(x=60, y=90)

ent1=Entry(font=('Aerial',18),fg="white", bg="midnightblue")

ent1.place(x= 60,y=27,height=45,width=300)

def openFile():
    global loc
    loc = ""
    ent1.delete(0,END)

    filepath = filedialog.askopenfilename(filetypes=(("", "*.png"),("", "*.jpeg") ,("","*.jpg")))

    fole = ""
    if filepath != "":
        fole = open(filepath)

    else:
        messagebox.showerror("ERROR", "PLEASE SELECT A FILE")

    if filepath != "":

        fole = str(fole)
        fole = fole.split("=", 1)[1]

        fole = fole.split("mode" , 1)[0]

        fole=fole[0:-1]
        panel5.configure(state=NORMAL)

        x = ""
        for i in fole:
            if i == "/":
                x = x + "\\"
            elif i == "'":
                x = x + ""
            else:
                x = x + i
               
        loc = x
        ent1.insert(END,x)
        
        poiu=loc[::-1]
        x1 = poiu.split("\\", 1)
        poiu=x1[1]
        poiu=poiu[::-1]
        x1[0]=x1[0][::-1]
        os.chdir(poiu)
        load1=imae.open(x1[0])
        load1 = load1.resize((300, 300))
        img2=ImageTk.PhotoImage(load1)
        labe.configure(image=img2)
        labe.image=img2
        
        windo.update_idletasks()
               

def destroy_widget(widget):
    widget.destroy()

def recognise(img):
    """Recognition using character model"""
    # Pre-processing the word
    img = word_normalization(
        img,
        60,
        border=False,
        tilt=True,
        hyst_norm=True)
    
    # Separate letters
    img = cv2.copyMakeBorder(
        img,
        0, 0, 30, 30,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0])
    #gaps1 = characters.segment(img, RNN=True)4
    implt(img)
    hay= np.array(img)
    from PIL import Image
    im = Image.fromarray(hay)
    im.save("your_file.jpeg")
    img1=image.load_img("your_file.jpeg",target_size=(120,120))
    
    x=image.img_to_array(img1)
    x=x/255
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)
    res =model.predict(img_data)
    pred =  np.argmax(res)
    letterq= {0: 'అ', 1: 'ఆ', 2: 'ఇ', 3: 'ఈ', 4: 'ఉ', 5: 'ఊ', 6: 'ఋ', 7: 'ఎ', 8: 'ఏ', 9: 'ఐ', 10: 'ఒ', 11: 'ఓ', 12: 'ఔ', 13: 'ౠ'}
    
    return str(letterq[pred])

def pred_digit():
    global loc
    image = cv2.cvtColor(cv2.imread(loc), cv2.COLOR_BGR2RGB)
    from page import detection
    crop = detection(image)
    implt(crop)
    from words import detection
    boxes = detection(image)
    lines = sort_words(boxes)
    p=''
    s=''
    for line in lines:
        s = s + " ".join([recognise(crop[y1:y2, x1:x2]) for (x1, y1, x2, y2) in line])
    if (len(pred2(x))!=0):
        s=len(pred2(x))
    no = tk.Label(windo, text='Predicted Word is: '+str(s), width=35, height=1,
                  fg="white", bg="midnightblue",
                  font=('times', 16, ' bold '))
    
    no.place(x=60, y=550)

panel5 = Button(windo,text = 'Predict Letter',state=DISABLED,command = pred_digit,width = 15,borderwidth=0,bg = 'midnightblue',fg = 'white',font = ('times',18,'bold'))
panel5.place(x=90, y=455)

panel6 = Button(windo,text = 'Browse...',width = 15,borderwidth=0,command = openFile,bg ='red',fg = 'white',font = ('times',18,'bold'))
panel6.place(x=90, y=405)


windo.mainloop()