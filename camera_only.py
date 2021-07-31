# -*- coding: UTF-8 -*-
import cv2
import os
import time
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

cascade_path = ('/usr/local/lib/python3.7/dist-packages/cv2/data/haarcascade_frontalface_default.xml')#ここのパスは必要に応じて変更してください
cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
color = (255, 255, 255)

pwm.set_pwm(0, 0, 375)
time.sleep(1)
pwm.set_pwm(1, 0, 375)
time.sleep(1)

now_degree_x, now_degree_y, move_degree_x, move_degree_y = 375, 375, 0, 0

while(True):
    ret, frame = cap.read()
    facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

    for rect in facerect:
        img_x = rect[0]+rect[2]/2
        img_y = rect[1]+rect[3]/2
        print(img_x, img_y)
        move_degree_x = now_degree_x - (img_x-160)*0.04 #ここの数値も(*-.-)必要に応じて変更してください
        move_degree_y = now_degree_y + (img_y-120)*0.04 #ここの数値も(*-.-)必要に応じて変更してください
        print('deg: ', move_degree_x , move_degree_y)
        pwm.set_pwm(0, 0, int(move_degree_x))#pan
        pwm.set_pwm(1, 0, int(move_degree_y))#tilt
        #time.sleep(0.1)
        now_degree_x = move_degree_x
        now_degree_y = move_degree_y
        cv2.circle(frame, (int(img_x), int(img_y)), 10, (255,255,255), -1)
        cv2.rectangle(frame, tuple(rect[0:2]),tuple(rect[0:2] + rect[2:4]), color, thickness=3)

    cv2.imshow("Show FLAME Image", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()