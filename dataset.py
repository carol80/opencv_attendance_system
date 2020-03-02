import cv2
import os
import numpy as np

face_classifier = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+2*h-1, x:x+2*w-1]

    return cropped_face,x,y,w,h

# def image_size(img):

#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     # faces = face_classifier.detectMultiScale(gray,1.3,5)

#     if faces is():
#         return None

#     for(x,y,w,h) in faces:
#         cropped_face = img[y:y+h, x:x+w]

#     return x,y,w,h


cap = cv2.VideoCapture(0)
count = 0
section = 1		#increment while adding every new dataset

roll = input("Enter student's Roll Number or Name : ")
# Id = roll[-2:]

folderName = roll                                                       # creating the person or user folder
folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images/"+folderName)
if not os.path.exists(folderPath):
    os.makedirs(folderPath)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            # X, Y, W, H = image_size(frame)
            cropped_face,x,y,w,h = face_extractor(frame)
            face = cv2.resize(cropped_face,(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # file_name_path = './dataset/user'+str(count)+'.jpg'
            # cv2.imwrite(file_name_path,face)
            cv2.imwrite(folderPath + "/" + str(count) + ".jpg",face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
            # print()

            # f = open(folderPath + "/" + roll + "_" + str(count) + ".txt","w+")            #for yolov3 .txt file generation 
            # # str1 = "" + section + " " + x + " " + y + " " + w + " " + h + ""
            # f.write("%d %f %f %f %f" % (section, x, y, w, h))
            # f.close()
            # print("%d %f %f %f %f" % (section, x, y, w, h))
        else:
            print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==120:
            break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
