#inferenceをgoogleドライブのutilsフォルダにあるファイルと入れ替えてください
#forcusがターゲットとなる人の座標[x,y]です

from statistics import mode
import sys
import cv2
from keras.models import load_model
import numpy as np
import time
sys.path.append("/bin/face_classification/src/utils")
from datasets import get_labels
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
from inference import load_detection_model
from inference import draw_emotion
from preprocessor import preprocess_input

import cv2
import os
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO

sys.path.append("/bin/face_classification/src")
sys.path.append("/bin/face_classification/trained_models/emotion_models")

#parameters for loading data and images
detection_model_path = "/bin/face_classification/trained_models/detection_models/haarcascade_frontalface_default.xml"
#haarcascade_frontalface_default.xml...顔 (正面)

emotion_model_path = "/bin/face_classification/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
emotion_labels = get_labels('fer2013')

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)

# GPIOピン番号の指示方法
GPIO.setmode(GPIO.BOARD)

# 前進最高速、停止、後進最高速のPWM値(右)
R_pwm_hi_top_forward = 480
R_pwm_top_forward = 440
R_pwm_forward = 380
R_pwm_stop = 350
R_pwm_back = 320
R_pwm_top_back = 285
R_pwm_hi_top_back = 240
 
# 前進最高速、停止、後進最高速のPWM値(左)
L_pwm_hi_top_forward = 230
L_pwm_top_forward = 256
L_pwm_forward = 328
L_pwm_stop = 355
L_pwm_back = 383
L_pwm_top_back = 430
L_pwm_hi_top_back = 450



# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models(学習済みモデルの読込)
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
#cv2.namedWindow('window_frame')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#for take shot counter
ts = time.time()#撮影用の時間

#shot number counter
picno=0#画像保存の連番初期値

label=["Angry:","Disgust:","Fear:","Happy:","Sad:","Surpris:","Neutral:"] #emotion label 

forcus=[320,240] #deflt target position for camera

#カメラの向きを初期値に戻す(pan軸)
pwm.set_pwm(0, 0, 375)

#1秒待つ
time.sleep(1)

#カメラの向きを初期値に戻す(tilt軸)
pwm.set_pwm(1, 0, 375)

#1秒待つ
time.sleep(1)


now_degree_x, now_degree_y, move_degree_x, move_degree_y = 375, 375, 0, 0


while True:#感情認識の処理　中断されるまでひたすら繰り返す

    ret, frame = cap.read()#[1]
    #read()..1コマ分のキャプチャ画像データを読み込み

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cvtColor(画像ファイル, 変換の指定)..画像の変換
    #グレースケールの画像に変換

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #BGRをRGBに変換

    faces = detect_faces(face_detection, gray_image)
    
    #detected number of person
    number = 1 
    #感情解析の結果を取り出すために、何人いるかをカウントする。初期値として１

    # whether detect person or not
    person = 0  
    #後のエラー回避のため顔が認識されているかどうかをフラグとする 初期値は０

    #flog for taking photo
    shot=0 
    #撮影用のフラグ　初期値は０で撮影しない。1になった場合にループの最後で画像を保存

    for face_coordinates in faces:
    #顔が認識された場合のみこのforが回る。左上にいる人から順番に一人ずつ感情を解析する

        person=1
        #まずは1人目の処理、顔の範囲を切り抜き gray_faceに挿入

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    
        gray_face = gray_image[y1:y2, x1:x2]
    
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

#切り抜いた顔の画像を感情解析にかける、中身はあまり読まなくても大丈夫です。
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0) 
            #.pop(0)インデックス(0)を削除

        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

#感情の最大値に合わせて顔を囲う線の色を変える。
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

#顔を四角の枠で囲い、最も感情が大きかった項目名をテキストで表示
        draw_bounding_box(face_coordinates, frame, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,color, 0, -45, 1, 1)

        if faces.shape[0]>=1:
           draw_emotion(emotion_prediction,rgb_image,number,label)
           number = number+1
        else:
           continue

#Jedgement for takeing shot　感情分析の結果から撮影する条件を指定、shot=1が撮影するフラグ
#emotion_predictionが感情認識結果　解析結果は1人分で1人ずつ順番に解析70行目のfor分で繰り返す
#emotion_prediction[0,3]　の3の位置の数字が感情の種類に対応、処理しやすいために100倍
#0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surpris,6=Neutral
        if emotion_prediction[0,3]*100>80:
        #ここの条件で撮影条件が変わります。今は複数の人の誰かが８０％以上happyなら撮影
        
            shot=1
        else:pass

#calucuration camera target position　認識された複数の顔座標の中央を計算。
#facesが[1人目の顔の左端x座標、上のy座標、W幅、H高さ]
#     　[2人目の顔の左端x座標、上のy座標、W幅、H高さ]
#     　[3人目・・・]
#と出されます。1人も認識できない場合は空
    if person==1:
        aver=np.average(faces,axis=0)    #とりあえず平均
        forcus[0]=int(aver[0]+aver[2]/2) #ｘ座標の平均＋Ｗ幅の平均
        forcus[1]=int(aver[1]+aver[3]/2) #ｙ座標の平均＋Ｈ幅の平均
        cv2.circle(rgb_image,(forcus[0],forcus[1]),30,(0,0,255),5) #わかりやすいように〇を描く

        move_degree_x = now_degree_x - (forcus[0]-320)*0.04
        move_degree_y = now_degree_y + (forcus[1]-240)*0.04
        print('deg: ', move_degree_x , move_degree_y)

       # 首の向き変更 (パン軸)
        pwm.set_pwm(0, 0, int(move_degree_x))
        # 首の向き変更 (チルト軸)
        pwm.set_pwm(1, 0, int(move_degree_y))
        
        now_degree_x = move_degree_x
        now_degree_y = move_degree_y

        # 体の向き変更    
        if move_degree_x >= 450 and move_degree_x <= 640:
            print("Turn Right")
            pwm.set_pwm(14, 0, R_pwm_forward)
            pwm.set_pwm(15, 0, L_pwm_back)

        elif move_degree_x <= 330 and move_degree_x >= 170:
            print("Turn Left")
            pwm.set_pwm(14, 0, R_pwm_back)
            pwm.set_pwm(15, 0, L_pwm_forward)

        else:
            print("Center")
            pwm.set_pwm(14, 0, R_pwm_stop)
            pwm.set_pwm(15, 0, L_pwm_stop)     


    else:pass
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#画像をカラーに戻す
    cv2.imshow('window_frame', frame)#画像を描画

    

##take shot
    tn= time.time()
    tcount=tn-ts
    if shot==1 and tcount>10:
        picno=picno+1
        picname="happy" + str(picno) + ".png"
        cv2.imwrite(picname,frame)
        ts= time.time()
        print("----------------Photo------------------")
        
    else:pass
    print(tcount)
    print(forcus)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()