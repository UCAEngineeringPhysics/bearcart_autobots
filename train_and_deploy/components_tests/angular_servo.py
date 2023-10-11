"""
Servo test script.
Also calibrates the servo.
Please Log your calibrated angle values below:
    left most angle: -30
    right most angle: 90
    mid angle: 30
"""
from gpiozero import AngularServo
from time import sleep


# -90: left most, 90: right most
ang_servo = AngularServo(pin=17, initial_angle=30, min_angle=90, max_angle=-90)  

for a in range(-45, 91, 5):
    ang_servo.angle = a
    print(f"angle: {a}")
    sleep(.5)
ang_servo.angle = 30
print("CENTER")
sleep(1)
