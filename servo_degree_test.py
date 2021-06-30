#変更テスト
from __future__ import division
import Adafruit_PCA9685 as PCA
import time

#PWM Instance
pwm = PCA.PCA9685()
#freq=60Hz
pwm.set_pwm_freq(60)

def move(pan,tilt):
    pan = int(pan * 2.56 + 120)
    tilt = int(tilt * 2.56 + 120)
    pwm.set_pwm(0, 0, pan)
    pwm.set_pwm(1, 0, tilt)

# 0deg=0.5ms, 90deg=1.5ms, 18deg=2.5ms
# 0deg=120step, 90deg=350step, 180deg=580step
# y(deg)=2.56*x+120(step)

if __name__ == '__main__':
    print('move...')
    move(75,75)
    time.sleep(0.5)
    move(105,105)
    time.sleep(0.5)
    move(60,60)
    time.sleep(0.5)
    move(120,120)
    time.sleep(0.5)
    move(45,45)
    time.sleep(0.5)
    move(135,135)
    time.sleep(0.5)
    move(90,90)
    time.sleep(0.5)
    print('stop...')