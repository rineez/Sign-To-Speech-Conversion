import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
import cv2
import numpy as np
import tensorflow as tf
import keras

model = keras.models.load_model("asl_classifier.keras")

LABEL_CLASS_MAP = { 0:'', 1:'A', 2:'B', 3:'C', 4:'D', 5:'E', 6:'F', 7:'G', 8:'H', 9:'I',10:'J',
                11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P', 17:"Q", 18:'R', 19:'S', 20:'T',
                21:'U', 22:'V', 23:'W', 24:'X', 25:'Y', 26:'Z' }
COLOR_GREEN=(0,255,0)
x=0
y=0
w=128
h=127

img_size=128
minValue = 70
source=cv2.VideoCapture(0)
count = 0
string = " "
prev = " "
prev_val = 0
while(True):
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.rectangle(img,(x,y),(x+w,y+h),color_dict,2)
    cv2.rectangle(img,(24,24),(310 , 310),COLOR_GREEN,2)
    crop_img=gray[24:310,24:310]
    count = count + 1
    if(count % 100 == 0):
        prev_val = count
    cv2.putText(img, str(prev_val//100), (300, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2) 
    blur = cv2.GaussianBlur(crop_img,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    resized=cv2.resize(res,(img_size,img_size))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,img_size,img_size,1))
    result = model.predict(reshaped)
    print("Predicted label: ", result)
    print("ArgMax=", np.argmax(result,axis=1))
    label=np.argmax(result, axis=1)[0]
    prev = LABEL_CLASS_MAP[label] 
    if(label != 0):
            string = string + prev
            #if(len(string)==1 or string[len(string)] != " "):
    
    cv2.putText(img, prev, (24, 14),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
    cv2.putText(img, string, (275, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
    cv2.imshow("Preprocessed",res)    
    cv2.imshow('LIVE',img)
    
    key=cv2.waitKey(10)
    if(key==27):#press Esc. to exit
        break

print(string)
cv2.destroyAllWindows()
source.release()

cv2.destroyAllWindows()

from gtts import gTTS
from IPython.display import Audio

# Language in which you want to convert 
language = 'en'
# Passing the text and language to the engine,  
# here we have marked slow=False. Which tells  
# the module that the converted audio should  
# have a high speed 
tts = gTTS(text=string, lang=language, slow=False) 
  
# Saving the converted audio in a mp3 file tts_out.mp3 
sound_file = os.path.join('out','tts_out.mp3')
tts.save(sound_file)

Audio(sound_file,autoplay=True)