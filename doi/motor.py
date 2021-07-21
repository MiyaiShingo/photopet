import Adafruit_PCA9685

def motor1(Rmotor, Lmotor):
    pwm = Adafruit_PCA9685.PCA9685()
    pwm.set_pwm_freq(60)

    pwm.set_pwm(0, 0, in1)
    pwm.set_pwm(1, 0, in2)
    pwm.set_pwm(2, 0, in3)
    pwm.set_pwm(3, 0, in4)