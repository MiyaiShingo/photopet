import cv2
import numpy as np

from face import facepoint as fcp

# 画像ファイル読み込み
img = cv2.imread("input/img_test.jpg")

# 画像ファイルをface.pyに渡して、顔画像と座標を取り出す。
fcprtn = fcp(img)

