import keras
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import base64
from io import BytesIO
import json
import cv2
from keras.models import load_model
import numpy as np


# load the model
model = load_model('Final_Model_Face.h5')
# laod the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img , 1.3 , 5)
    
    if faces is ():
        return None
    
    # crop the faces
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,255) , 2)
        cropped_face = img[y:y+h , x:x+w]
        
    return cropped_face

# face recognition wiht the cam 

cam = cv2.VideoCapture(0)
while True:
    success , frame = cam.read()
    
    
    
    face = face_extractor(frame)
    
    if type(face) is np.ndarray:
        face = cv2.resize(face , (224,224))
        img = Image.fromarray(face , 'RGB')
        
        img_array = np.array(img)
        
        img_array = np.expand_dims(img_array , axis = 0)
        pred = model.predict(img_array)
        print(pred)
        
        none = "None Matching"
        
        if(pred[0][0]>0.3):
            name = 'Yuvan'
        if(pred[0][1]>0.3):
            name = "Bhubesh"
        cv2.putText(frame , name , (50,50) , cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(frame , "No Face Found" , (50,50) , cv2.FONT_HERSHEY_COMPLEX , 1, (255,0,255) , 2)
    cv2.imshow('Face_Detect' , frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



