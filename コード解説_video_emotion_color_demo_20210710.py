#下記の事前準備が必要です。チーム３のスレッドおよび、Tensoflow3インストールまでを参照してください。
#主要な所は face_classificationの学習済みモデルは公開リポジトリから取得します.
#　git clone https://github.com/oarriaga/face_classification
#　/face_classification/src/utils/preprocessor.py書換
#次の1行をコメントアウト(or 削除)
#from scipy.misc import imread, imresize
#ソース変更
#return imread(image_name)　－＞　return imageio.imread(image_name)
#return imresize(image_array, size)　－＞　return imageio.imresize(image_array, size)
#inferenceをgoogleドライブのutilsフォルダにあるファイルと入れ替えてください。
#video_emotion_color_demo.pyをロボット作成用に修正を加えたのが本プログラムです。
#
#forcusがターゲットとなる人の座標[x,y]です

#中央リポジトリにプッシュしたい（宮井）
#土井が書いて見る

from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import time

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import draw_emotion
from utils.preprocessor import preprocess_input


# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

ts = time.time()  #撮影用の時間
picno=0 #画像保存の連番初期値
label=["Angry:","Disgust:","Fear:","Happy:","Sad:","Surpris:","Neutral:"] #emotion label 

forcus=[350,350] #カメラ追跡用の座標、人が認識されるまでの「ｘ、ｙ」の初期値。カメラの中央のピクセルを入れてください。

while True: #感情認識の処理　中断されるまでひたすら繰り返す

    bgr_image = video_capture.read()[1] #カメラから画像取得
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) #感情解析用にグレーに変更
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image) #顔を識別
    number = 1 #感情解析の結果を取り出すために、何人いるかをカウントする。初期値として１
    person = 0  #後のエラー回避のため顔が認識されているかどうかをフラグとする 初期値は０
    shot=0 #撮影用のフラグ　初期値は０で撮影しない。1になった場合にループの最後で画像を保存
    for face_coordinates in faces: #顔が認識された場合のみこのforが回る。左上にいる人から順番に一人ずつ感情を解析する
        person=1 #まずは1人目の処理、顔の範囲を切り抜き gray_faceに挿入
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
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,color, 0, -45, 1, 1)

#感情の数字を棒グラフと数値で表示、上から順番に1人目、2人目・・・、
        if faces.shape[0]>=1:
           draw_emotion(emotion_prediction,rgb_image,number,label)
           number = number+1
        else:
           continue
 
#Jedgement for takeing shot　感情分析の結果から撮影する条件を指定、shot=1が撮影するフラグ
#emotion_predictionが感情認識結果　解析結果は1人分で1人ずつ順番に解析70行目のfor分で繰り返す
#emotion_prediction[0,3]　の3の位置の数字が感情の種類に対応、処理しやすいために100倍
#0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surpris,6=Neutral
        if emotion_prediction[0,3]*100>80: #ここの条件で撮影条件が変わります。今は複数の人の誰かが８０％以上happyなら撮影
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

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) #画像をカラーに戻す
    cv2.imshow('window_frame', bgr_image) #画像を描画

##take shot　前回撮影から時間が経過＆撮影条件を満たした場合に画像を連番で保存
    tn= time.time() #現在時刻取得
    tcount=tn-ts #前回撮影した時刻と現在時刻の差を計算
    if shot==1 and tcount>10: #撮影条件
       picno=picno+1  #保存する画像の連番
       picname="happy" + str(picno) + ".png" #保存する名前作成
       cv2.imwrite(picname,bgr_image) #画像保存
       ts= time.time() #撮影した時刻をいったん記憶
    else:pass
    print(tcount) #旋回撮影からの経過時間を表示　自分の確認用
    print(forcus) #カメラのフォーカス座標を表示　自分の確認用


    if cv2.waitKey(1) & 0xFF == ord('q'): #修了条件
        break
video_capture.release()
cv2.destroyAllWindows()
