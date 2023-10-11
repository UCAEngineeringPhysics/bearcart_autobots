from gpiozero import AngularServo
from time import sleep

servo = AngularServo(17, min_angle=0, max_angle=180)

for a in range(181):
    if not a%45:
        servo.angle = a
        print(f"angle: {a}")
        sleep(2)

servo.angle = 90
print("CENTER")
sleep(2)
