#inferenceをgoogleドライブのutilsフォルダにあるファイルと入れ替えてください
#forcusがターゲットとなる人の座標[x,y]です

from statistics import mode
import sys
import cv2
from keras.models import load_model
import numpy as np
import time
import Adafruit_PCA9685
import RPi.GPIO as GPIO
import pigpio

import functions

#プログラムの置き場を指定して、モジュールを読み込む"pi"または"bin"
path="pi"
if path=="bin":
# /binに置いた場合
    sys.path.append("/bin/face_classification/src/utils")
    from datasets import get_labels
    from preprocessor import preprocess_input
    from inference import detect_faces
    from inference import draw_text
    from inference import draw_bounding_box
    from inference import apply_offsets
    from inference import load_detection_model
    from inference import draw_emotion
    #parameters for loading data and images
    #haarcascade_frontalface_default.xml...顔 (正面)
    sys.path.append("/bin/face_classification/src")
    sys.path.append("/bin/face_classification/trained_models/emotion_models")
    detection_model_path = "/bin/face_classification/trained_models/detection_models/haarcascade_frontalface_default.xml"
    emotion_model_path = "/bin/face_classification/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
elif path=="pi":
    #/piに置いた場合
    from utils.datasets import get_labels
    from utils.inference import detect_faces
    from utils.inference import draw_text
    from utils.inference import draw_bounding_box
    from utils.inference import apply_offsets
    from utils.inference import load_detection_model
    from utils.inference import draw_emotion
    from utils.preprocessor import preprocess_input
    #parameters for loading data and images
    #haarcascade_frontalface_default.xml...顔 (正面)
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

#↓以下は置き場によらず、共通
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
GPIO.setmode(GPIO.BOARD)# GPIOピン番号の指示方法

pi=pigpio.pi()

#ｶﾒﾗ関係の設定・モデルの読み込み
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
forcus=[320,240] #座標の初期値：画面の縦横の中心
# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)
# loading models(学習済みモデルの読込)
emotion_labels = get_labels('fer2013')
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []
# starting video streaming
#cv2.namedWindow('window_frame')


ts = time.time()#撮影用の時間
picno=0#画像保存の連番初期値
label=["Angry:","Disgust:","Fear:","Happy:","Sad:","Surpris:","Neutral:"] #emotion label 

#カメラ台を初期位置に動かす(pan軸)
ini_degree_x, ini_degree_y=375,450
pwm.set_pwm(0, 0, ini_degree_x)
time.sleep(1)#1秒待つ
#カメラの向きを初期値に戻す(tilt軸)
pwm.set_pwm(1, 0, ini_degree_y)
time.sleep(1)#1秒待つ

#ｶﾒﾗ台関係の初期値
now_degree_x, now_degree_y, move_degree_x, move_degree_y = ini_degree_x, ini_degree_y, 0, 0

while True:#感情認識の処理　中断されるまでひたすら繰り返す

    #顔検出と感情認識の実行。shot:感情認識結果から画像を保存するか否かのフラグ、forcus：顔検出結果より、顔のx,y座標
    shot,forcus,frame=functions.camera_processing(forcus,cap,face_detection,emotion_offsets,emotion_labels,frame_window,emotion_classifier,emotion_target_size,emotion_window,label)
    
     #顔の座標を受け取ってカメラ台を実際に動かした後、ｶﾒﾗの現在位置：now_degreeを更新。   
    now_degree_x,now_degree_y,move_degree_x=functions.move_camera(now_degree_x, now_degree_y, forcus,pwm)
  
    #カメラ台の動きに合わせて本体を移動する
    #制御方式の選択　1：スピコン（酒井さん方式）、2：DRV 8833 （土井さん方式）、3:TB6612（宮井方式）,0：本体なしカメラ台のみ
    control_method=3
    if control_method==1:
        functions.move_tires_1(move_degree_x,pwm)        
    elif control_method==2:
        functions.move_tires_2(move_degree_x)
    elif control_method==3:
        functions.move_tires_3(move_degree_x,pi,pwm,ini_degree_x)  
    elif control_method==0:
        continue
    
    #実行フラグ：shotの状態に応じて、連番:picnoを付けて画像を保存する。
    picno,ts=functions.save_image(shot,picno,ts,frame)

    #終了条件：Qキーを押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#for emerjency of miyai-method
# PWM_A=17
# PWM_B=13
# pi.set_PWM_dutycycle(PWM_A,0)
# pi.set_PWM_dutycycle(PWM_B,0)