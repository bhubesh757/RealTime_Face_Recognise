# import the requirements

import cv2
import numpy as np

# load the haar cascde classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load the function or create the function

def face_extractor(img):
    
    faces = face_classifier.detectMultiScale(img , 1.3 , 5)  # https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
    
    if faces is ():
        return None
    
    # crop all faces found:
    
    for(x,y,w,h) in faces:
        x = x-10
        y = y-10
        cropped_face = img[y:y+h+50 , x:x+w+50]
        
    return cropped_face

# then initilize the webcam

cam = cv2.VideoCapture(0) # the 0 referrd here is the webcam of the laptop
count = 0

# to collect the samples of the faces from the webcam input

while True:
    
    success , frame = cam.read()
    
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame) , (400 , 400))
        
        # to save the file which has been captured
        
        file_name_path = "./bhubesh/" + str(count) +".jpg"
        cv2.imwrite(file_name_path,face)
        
        # put the text how uch images has been captured
        
        cv2.putText(face , str(count) , (50 , 50) , cv2.FONT_HERSHEY_COMPLEX , 1 , (255,255,0) , 2)
        cv2.imshow("cropped Face" , face)
        
    else:
        print("Face not Found!!")
        pass
    
    
    if cv2.waitKey(1) == 13 or count == 100:
        break
cam.release()
cv2.destroyAllWindows()
print(' collecting samples completed')





