# import pygame
from pygame.locals import *
from pygame import event, display, joystick


def get_numControllers():
    return joystick.get_count()

display.init()
joystick.init()
print(f"{get_numControllers()} joystick connected")
js = joystick.Joystick(0)
while True:
    for e in event.get():
        if e.type == JOYAXISMOTION:
            ax0 = js.get_axis(0)
            ax1 = js.get_axis(1)
            ax2 = js.get_axis(2)
            ax3 = js.get_axis(3)
            ax4 = js.get_axis(4)
            ax5 = js.get_axis(5)
            print(f"axis 0: {ax0}")
            print(f"axis 1: {ax1}")
            print(f"axis 2: {ax2}")
            print(f"axis 3: {ax3}")
            print(f"axis 4: {ax4}")
            print(f"axis 5: {ax5}")

