import sys
import os
import numpy as np
import torch
from torchvision import transforms
import cnn_network
import cv2 as cv
import pygame
import time
from gpiozero import LED, AngularServo, PhaseEnableMotor
import json
from time import time


# SETUP
# Init variables
act_st, act_th = 0., 0.
# Load autopilot model
model_datetime = '2023_12_01_14_31'
model_path = os.path.join(
    sys.path[0], 
    'data', 
    model_datetime, 
    'DonkeyNet-15epochs-0.0001lr.pth'
)
autopilot = cnn_network.DonkeyNet()
autopilot.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
to_tensor = transforms.ToTensor()
# Load configs
config_path = os.path.join(sys.path[0], "configs.json")
params_file = open(config_path)
params = json.load(params_file)
STEER_CENTER = params['steer_center']
STEER_RANGE = params['steer_range']
THROTTLE_LIMIT = params['throttle_limit']
# Init head and tail light
head_led = LED(16)  # TODO
tail_led = LED(12)
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
        if ret:
            frame_counts += 1
        for e in pygame.event.get():  # read controller input
            if e.type == pygame.JOYBUTTONDOWN:
                if js.get_button(0):
                    throttle.stop()
                    throttle.close()
                    steer.close()
                    cv.destroyAllWindows()
                    pygame.quit()
                    print("E-STOP PRESSED. TERMINATE")
                    sys.exit()
        # predict steer and throttle
        image = cv.resize(frame, (120, 160))
        img_tensor = to_tensor(image)
        st_pred, th_pred = autopilot(img_tensor[None, :]).squeeze()
        act_st = st_pred.detach().numpy()
        act_th = np.clip(th_pred.detach().numpy(), -1, 1)
        # Map axis value to angle: steering_center + act_st * steering_range
        ang = STEER_CENTER + act_st * STEER_RANGE
        ang = np.clip(ang, -90, 90)
        # Drive servo and motor
        steer.angle = ang
        if act_th >= 0:
            throttle.forward(min(act_th, THROTTLE_LIMIT))
        else:
            throttle.backward(min(-act_th, THROTTLE_LIMIT))
        # Log data
        action = [act_st, act_th]
        print(f"action: {action}")
        # Monitor frame rate, uncomment following 3 lines
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
