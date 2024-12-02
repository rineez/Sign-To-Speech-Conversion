import cv2
import numpy as np

X = 0
Y = 1
train_label = 'A' # input("Which label to train? : ")

source = cv2.VideoCapture(0)
color_green = (0,255,0)
win_x, win_y, win_w, win_h = (24, 24, 310, 310)
img_size=128
visual_threshold = 70
gaussian_window = 11
fine_tune_c = 2

ret,img = source.read()
img_h, img_w = img.shape[:2]
while(True):
    ret,img = source.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    frame_p1 = (win_x, win_y)
    frame_p2 = (win_x + win_w , win_y + win_h)
    cv2.rectangle(img,frame_p1,frame_p2,color_green,2)
    crop_img = gray[frame_p1[Y]:frame_p2[Y], frame_p1[X]:frame_p2[X]]
    cv2.putText(img, "Training Frame", (win_x, win_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

    # Clean up the image data for training
    blur = cv2.GaussianBlur(crop_img,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,gaussian_window,fine_tune_c)
    ret, res = cv2.threshold(th3, visual_threshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    resized=cv2.resize(res,(img_size,img_size))
    normalized=resized/255.0
    reshaped=np.reshape(normalized,(1,img_size,img_size,1))

    cv2.imshow('Live View',img)
    cv2.imshow("Training Input",res)

    key=cv2.waitKeyEx(1000)
    if(key==27):#press Esc. to exit
        break
    elif key == 2490368:  # Up arrow key
        win_y = win_y - 1 if win_y>0 else 0
    elif key == 2621440:  # Down arrow key
        win_y = win_y + 1 if win_y<(img_h-win_h) else img_h-win_h
    elif key == 2424832:  # Left arrow key
        win_x = win_x - 1 if win_x>0 else 0
    elif key == 2555904:  # Right arrow key
        win_x = win_x + 1 if win_x<(img_w-win_w) else img_w-win_w

print("Trained Label:", train_label)
cv2.destroyAllWindows()
source.release()
