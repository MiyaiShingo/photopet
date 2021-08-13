# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:03:27 2021

@author: user
"""
import time
import numpy as np
import cv2
import pigpio
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import draw_emotion
from utils.preprocessor import preprocess_input

#顔の座標と感情に応じて写真を撮るかどうかの判別実行
def camera_processing(forcus,cap,face_detection,emotion_offsets,emotion_labels,frame_window,emotion_classifier,emotion_target_size,emotion_window,label):
#     emotion_window=[]
#     # getting input model shapes for inference
    ret, frame = cap.read()#[1]
    #read()..1コマ分のキャプチャ画像データを読み込み
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cvtColor(画像ファイル, 変換の指定)..画像の変換
    #グレースケールの画像に変換
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #BGRをRGBに変換
    faces = detect_faces(face_detection, gray_image)
    number = 1 #感情解析の結果を取り出すために、何人いるかをカウントする。初期値として１
    person = 0    #後のエラー回避のため顔が認識されているかどうかをフラグとする 初期値は０
    shot=0 #撮影用のフラグ　初期値は０で撮影しない。1になった場合にループの最後で画像を保存
    
    for face_coordinates in faces:
    #顔が認識された場合のみこのforが回る。左上にいる人から順番に一人ずつ感情を解析する
        person=1#まずは1人目の処理、顔の範囲を切り抜き gray_faceに挿入
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
#             print("test")
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

        #感情分析の結果から撮影する条件を指定、shot=1が撮影するフラグ
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
    else:pass
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)#画像をカラーに戻す
    cv2.imshow('window_frame', frame)#画像を描画
    print(forcus)  
    return shot,forcus,frame

#カメラ台の制御
def move_camera(now_degree_x, now_degree_y, forcus,pwm):
    #カメラ台を動かす量の算出
    move_degree_x = now_degree_x - (forcus[0]-320)*0.04
    move_degree_y = now_degree_y + (forcus[1]-240)*0.04
    print('deg: ', move_degree_x , move_degree_y)
    # 首の向き変更 (パン軸)
    pwm.set_pwm(0, 0, int(move_degree_x))
    # 首の向き変更 (チルト軸)
    pwm.set_pwm(1, 0, int(move_degree_y))
    #カメラ台の現在値を更新    
    now_degree_x = move_degree_x
    now_degree_y = move_degree_y
    return now_degree_x,now_degree_y,move_degree_x

#スピコンによる本体の制御（酒井さん方式）
def move_tires_1(move_degree_x,pwm):
    # 前進最高速、停止、後進最高速のPWM値(右)
    R_pwm_forward = 380
    R_pwm_stop = 350
    R_pwm_back = 320
     # 前進最高速、停止、後進最高速のPWM値(左)
    L_pwm_forward = 328
    L_pwm_stop = 355
    L_pwm_back = 383
    
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

#DRV8833による本体の制御（土井さん方式）
def move_tires_2(move_degree_x,pwm):
    pass

#TB6612による本体制御（宮井方式）
def move_tires_3(move_degree_x,pi,pwm,ini_degree_x):
    PWM_A=17
    AIN1=22
    AIN2=27
    PWM_B=13
    BIN1=26
    BIN2=19
    
    FREQ=50
    SPEED=50
    move_time=0.2
    pi.set_mode(AIN1,pigpio.OUTPUT)
    pi.set_mode(AIN2,pigpio.OUTPUT)
    pi.set_mode(BIN1,pigpio.OUTPUT)
    pi.set_mode(BIN2,pigpio.OUTPUT)
    pi.set_mode(PWM_A,pigpio.OUTPUT)
    pi.set_mode(PWM_B,pigpio.OUTPUT)
    pi.set_PWM_frequency(PWM_A,FREQ)
    pi.set_PWM_frequency(PWM_B,FREQ)
    pi.set_PWM_range(PWM_A, 100 )
    pi.set_PWM_range(PWM_B, 100 )
  
    if move_degree_x >= 450 and move_degree_x <= 640:
        print("Turn Right")
        pi.write( AIN2, pigpio.HIGH )
        pi.write( AIN1, pigpio.LOW )
        pi.write( BIN2, pigpio.HIGH )
        pi.write( BIN1, pigpio.LOW )
        pi.set_PWM_dutycycle(PWM_A,SPEED)
        pi.set_PWM_dutycycle(PWM_B,SPEED)
        time.sleep(move_time)
        pwm.set_pwm(0, 0, ini_degree_x)  
    elif move_degree_x <= 330 and move_degree_x >= 170:
        print("Turn Left")
        pi.write( AIN1, pigpio.HIGH )
        pi.write( AIN2, pigpio.LOW )
        pi.write( BIN1, pigpio.HIGH )
        pi.write( BIN2, pigpio.LOW )
        pi.set_PWM_dutycycle(PWM_A,SPEED)
        pi.set_PWM_dutycycle(PWM_B,SPEED)
        time.sleep(move_time)
        pwm.set_pwm(0, 0, ini_degree_x)
   
    else:
        print("Center")
        pi.set_PWM_dutycycle(PWM_A,0)
        pi.set_PWM_dutycycle(PWM_B,0)
        
def save_image(shot,picno,ts,frame):
    tn= time.time()
    tcount=tn-ts
    if shot==1 and tcount>5:
        picno=picno+1
        picname="happy" + str(picno) + ".png"
        cv2.imwrite(picname,frame)
        ts= time.time()
        print("----------------Photo------------------")        
    else:pass
    print(tcount)
    return picno,ts
  
    