#import necessary libraries 
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from gpiozero import PhaseEnableMotor
import sys
import os
import cv2 as cv
from gpiozero import LED
import json
import csv
from datetime import datetime
from gpiozero import LED, AngularServo
from time import time
import torch
from torchvision import transforms
import cnn_network


#Check to make sure robot is in a suitable space for testing
is_lifted = input("Is any tire having contact with the ground or other objects? [yes/no]")
assert is_lifted=="no"
is_ready = input("Are you ready to start motor test? [yes/no]")
assert is_ready=="yes"


def get_numControllers():
    return joystick.get_count()
#Initiate joystick and display
display.init()
joystick.init()
print(f"{get_numControllers()} joystick connected")
#Create an instance of the joystick object
js = joystick.Joystick(0)
#Initiate controller
pygame.display.init()
pygame.joystick.init()
# init LEDs
head_led = LED(16)
tail_led = LED(12)
#Load the model
model_path = os.path.join(sys.path[0], 'models', 'DonkeyNet_15_epochs_lr_1e_3.pth')
to_tensor = transforms.ToTensor()
model = cnn_network.DonkeyNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# create data storage
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')

#Create an instance of the motor and servo objects 
motor = PhaseEnableMotor(phase=19, enable=26)
servo = AngularServo(17, min_angle=-90, max_angle=90)

#Assign default value of 0 for the motor PWM and the steering angle and max throttle limit
throttle = 0
steer = 0
throttle_lim = 0.5
#Set variables initially to zero
is_recording = False
frame_counts = 0
# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i/20)
    ret, frame = cap.read()
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")
try:
    while(True):
        #Check to make sure camera is getting images
        ret, frame = cap.read()
        if ret:
            frame_counts += 1
        else:
            motor.stop()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
        #Main program
        # predict steer and throttle
        image = cv.resize(frame, (120, 160))
        img_tensor = to_tensor(image)
        pred_steer, pred_throttle = model(img_tensor[None, :]).squeeze()
        steer = float(pred_steer)
        throttle = float(pred_throttle)

        #Conditions for throttling the motor 
        if (throttle < 0.05 and throttle > -0.05):
            motor.stop()
        elif (throttle > 0.05):
            motor.forward(throttle * throttle_lim)
        elif (throttle < -0.05):
            motor.backward(-throttle * throttle_lim)
        #Conditions for steering the servo
        if (steer < 0.05 and steer > -0.05):
            servo.angle = -45
        elif (steer > 0.05):
            servo.angle = steer * 90
        elif (steer < -0.05):
            servo.angle = steer * 90
        #Emergency stop
        for e in pygame.event.get():
            if e.type == pygame.JOYBUTTONDOWN:
                if pygame.joystick.Joystick(0).get_button(0):
                    print("Stop")
                    motor.stop()
                    cv.destroyAllWindows()
                    pygame.quit()
                    sys.exit()
       
        # monitor frame rate
        #duration_since_start = time() - start_stamp
        #ave_frame_rate = frame_counts / duration_since_start
        #print(f"frame rate: {ave_frame_rate}")
except KeyboardInterrupt:
    motor.stop()
    motor.close()
    print("Test interrupted!")