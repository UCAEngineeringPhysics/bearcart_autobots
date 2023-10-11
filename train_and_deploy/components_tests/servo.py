from gpiozero import Servo
from time import sleep

servo = Servo(17)

for _ in range(2):
    servo.min()
    print("min")
    sleep(1)
    servo.mid()
    print("mid")
    sleep(1)
    servo.max()
    print("max")
    sleep(1)
    servo.mid()
    print("mid")
    sleep(1)

