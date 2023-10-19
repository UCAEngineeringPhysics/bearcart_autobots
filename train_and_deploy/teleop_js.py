import sys
import os
import numpy as np
import cv2 as cv
import pygame
import time
from gpiozero import LED, AngularServo, PhaseEnableMotor
import json
from time import time


# SETUP
# Init variables
ax0_val, ax4_val = 0., 0.  # left joy med-lat, right joy ant-post
LED_STATUS = False
# Load configs
config_path = os.path.join(sys.path[0], "configs.json")
params_file = open(config_path)
params = json.load(params_file)
STEER_CENTER = params['steer_center']
STEER_RANGE = params['steer_range']
THROTTLE_LIMIT = params['throttle_limit']
# Init head and tail light
head_led = LED(12)
tail_led = LED(16)
# Init servo 
steer = AngularServo(
    pin=params['servo_pin'], 
    initial_angle=params['steer_center'],
    min_angle=params['servo_min_angle'], 
    max_angle=params['servo_max_angle'], 
)
steer.angle = STEER_CENTER #Starting angle
# Init motor 
throttle = PhaseEnableMotor(phase=19, enable=26)
# Init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# Init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):
    if not i % 20:
        print(i/20)  # count down 3, 2, 1 sec
    ret, frame = cap.read()
# Init timer
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.

# MAIN LOOP
try:
    while True:
        ret, frame = cap.read()  # read image
        cv.imshow('camera', cv.resize(frame, (200, 180)))
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYAXISMOTION:
                ax0_val = round((js.get_axis(0)), 2)  # keep 2 decimals
                ax4_val = round((js.get_axis(4)), 2)  
            elif e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(0):
                    throttle.stop()
                    throttle.close()
                    steer.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    print("E-STOP PRESSED. TERMINATE")
                    sys.exit()
                elif js.get_button(5):
                    head_led.toggle()
                    tail_led.toggle()
        # Calaculate steering and throttle value
        act_st = ax0_val  # steer_input: -1: left, 1: right
        act_th = -ax4_val  # throttle input: -1: max forward, 1: max backward
        act_th = np.clip(act_th, -1, 1)
        # Map axis value to angle: steering_center + act_st * steering_range
        ang = STEER_CENTER + act_st * STEER_RANGE
        ang = np.clip(ang, -90, 90)
        # Drive servo and motor
        steer.angle = ang

        if act_th >= 0:
            throttle.forward(min(act_th, THROTTLE_LIMIT))
        else:
            throttle.backward(min(-act_th, THROTTLE_LIMIT))
        # Log action
        action = [act_st, act_th]
        print(f"action: {action}")
        # Log frame rate
        frame_counts += 1
        since_start = time() - start_stamp
        frame_rate = frame_counts / since_start
        print(f"frame rate: {frame_rate}")
        # Press "q" to quit
        if cv.waitKey(1)==ord('q'):
            throttle.stop()
            throttle.close()
            steer.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
# Take care terminate signal (Ctrl-c)
except KeyboardInterrupt:
    throttle.stop()
    throttle.close()
    steer.close()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
