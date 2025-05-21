# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:11:46 2021

@author: tiger
"""

import cv2 
import socket
from threading import Thread
import json
import time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
#server_address = ('localhost', 999)
server_address = ('10.1.89.17', 999)
print('connecting to %s port %s' % server_address)
sock.connect(server_address)
typeOf=""


def SendData(typeOf, x1, y1, w, h,socket,mapW,mapH):             
        #input("awaiting key")
        dictData=json.dumps(({'X':x1,'Y':y1,'Type':typeOf,'Width':w,'Height':h,'Xlimit':mapW, 'Ylimit':mapH}))
        socket.sendall(bytes(dictData,encoding="utf-8"))
        time.sleep(0.1)
        
        
# Enable we
# '0' is default ID for builtin web cam
# for external web cam ID can be 1 or -1
imcap = cv2.VideoCapture(0)
imcap.set(3, 600) # set width as 640
imcap.set(4, 400) # set height as 480
# importing cascade

bead10umCascade = cv2.CascadeClassifier("C:\\Users\\tiger\\OneDrive - University of Arkansas\\Machine Learning For Mechanical Engineers\\Project\\10um Beads\\classifier\\cascade.xml")
bead6umCascade = cv2.CascadeClassifier("C:\\Users\\tiger\\OneDrive - University of Arkansas\\Machine Learning For Mechanical Engineers\\Project\\6um Beads\\classifier\\cascade.xml")

padding=0
top=70
bottom=95
left=380
right=168
close=False
    
def listenThread():
    global top
    global bottom
    global right
    global left
    global close
    top=70
    bottom=95
    left=380
    right=168
    close=False
    while ~close:
        newBound=input("New Bounds T:B:L:R  ")      
        if newBound!="" and newBound!="q" and ~close:
            data=newBound.split(":")
            print(data)
            top=int(data[0])
            bottom=int(data[1])
            right=int(data[3])
            left=int(data[2])


x = Thread(target=listenThread,args=())
x.start()
try:
    while True:
        success, img = imcap.read() # capture frame from video    
        # converting image from color to grayscale 
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Xlimit=imgGray.shape[1]
        Ylimit=imgGray.shape[0]
        
        #cropping image
        newBottom= Ylimit-bottom
        newRight=Xlimit-right
        croppedImage=imgGray[top:newBottom,left:newRight]
        
        #getting bounds of new image
        Xlimit=croppedImage.shape[1]
        Ylimit=croppedImage.shape[0]
        #print(imgGray.size)
        # Getting corners around the face
        # 1.3 = scale factor, 5 = minimum neighbor can be detected
        #cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) 
        bead6um = bead6umCascade.detectMultiScale(croppedImage, 1.3,1,maxSize=(65,65))
               
        bead10 = bead10umCascade.detectMultiScale(croppedImage, 1.3,1,minSize=(70,70))
    
        # drawing bounding box around face
        if(len(bead6um)!=0):
            for (x, y, w, h) in bead6um:
                if y<padding or y>(Ylimit-padding) or x<padding or x>(Xlimit-padding):
                    continue
                croppedImage = cv2.rectangle(croppedImage, (x, y), (x + w, y + h), (255, 0,   255), 3)
                #print("("+x.astype(str)+","+y.astype(str)+")" "("+w.astype(str)+","+h.astype(str)+")")
                #SendData("6um",int(x),int(y),int(w),int(h),sock,Xlimit,Ylimit)
        if(len(bead10)!=0):
            for (x, y, w, h) in bead10:
                if y<padding or y>(Ylimit-padding) or x<padding or x>(Xlimit-padding):
                    continue
                croppedImage = cv2.rectangle(croppedImage, (x, y), (x + w, y + h), (0, 255,   0), 3)
                #print("("+x.astype(str)+","+y.astype(str)+")" "("+w.astype(str)+","+h.astype(str)+")")
                #SendData("10um",int(x),int(y),int(w),int(h),sock,Xlimit,Ylimit)
        #displaying image with bounding box
       
        cv2.imshow('face_detect', croppedImage)
        # loop will be broken when 'q' is pressed on the keyboard
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    imcap.release()
    cv2.destroyWindow('face_detect')
    sock.close()
finally:
    print('closing socket')
    cv2.destroyWindow('face_detect')
    #sock.close()
        