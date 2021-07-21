import cv2

def facepoint(img):
    # 画像をグレーに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 学習ファイルを読み出して顔を認識
    cascade = cv2.CascadeClassifier("trained_models/haarcascade_frontalface_default.xml")
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    # 顔に四角の枠を描画
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), 3)
    # 描画後の画像を保存
    # cv2.imwrite("data/output1.png", img)
    # ウインドウを生成して表示
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img", img)
    cv2.waitKey(0)  # なにかキーを押すまでウインドウを保持
    cv2.destroyAllWindows()
    return img, face


def facepoint2(n):
    print("hello world")
    n= n * 2
    return n

