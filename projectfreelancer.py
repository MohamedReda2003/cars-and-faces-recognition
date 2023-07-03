import tkinter as tk
import cv2
import numpy as np

def recognizing():
    face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye= cv2.CascadeClassifier('haarcascade_eye.xml')
    car = cv2.CascadeClassifier('haarcascade_car.xml')
    cap=cv2.VideoCapture(0)
    i=0
    while True:
        _,img=cap.read()
        
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(gray)
        eyes=eye.detectMultiScale(gray)
        cars=car.detectMultiScale(gray)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
            cv2.imwrite(f"faces/face{i}.png", img)
        
        for (x,y,w,h) in eyes:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(35, 200, 120), thickness=1)
            
        for (x,y,w,h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
            cv2.imwrite(f"cars/car{i}.png", img)
        
        i+=1
            
            
        cv2.imshow('img',img)
        if cv2.waitKey(1)==ord('q'):
            break
      

    cap.release()
    cv2.destroyAllWindows()


app = tk.Tk()
app.title("OpenCV Function App")

open_button = tk.Button(app, text="Run the recognition system ", command=recognizing)
open_button.pack(pady=10)

app.mainloop()
