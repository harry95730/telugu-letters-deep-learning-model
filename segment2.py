import cv2
import numpy as np 
import sys
import matplotlib.pyplot as plt
def segment(img): 
    i_list=[]
    j_list=[]
    height,width=img.shape
    s=[]
    for i in range(width):
        k=0
        for j in range(height):
            k=k+img[j,i]
        s.append(k)
    i=0
    b=[]
    while(i<width):
        while(i<width and s[i]<2):
            i+=1
        b.append(i)
        threshold=s[i]
        i+=1
        while(i <width and s[i]==threshold):
            i+=1
        while(i<width and s[i]>10):
            i+=1
        b.append(i)
        threshold=s[i]
        i+=1
        while(i <width and s[i]==threshold):
            i+=1
    fin=[]
    fin.append(b[0])
    for i in range(1,len(b)-2,2):
        fin.append(int((b[i]+b[i+1])/2))
    fin.append(b[len(b)-1])
        
    return fin
        
            
        