import ss
import dxinput
import cv2
import digits
import time
import keras

pressed = [0 for i in range(5)]
screen_name = "Need for Speed™ Most Wanted"

img = ss.get_screen(screen_name)

def restart_race():
    time.sleep(1)
    dxinput.PressKey(0x1C) # RETURN
    time.sleep(0.2)
    dxinput.ReleaseKey(0x1C) 
    time.sleep(1.5)
    dxinput.PressKey(0x02) # NUMBER 1
    time.sleep(0.2)
    dxinput.ReleaseKey(0x02)
    time.sleep(1.5)
    dxinput.PressKey(0xCB) # LEFT ARROW
    time.sleep(0.2)
    dxinput.ReleaseKey(0xCB)
    time.sleep(1.5)
    dxinput.PressKey(0x1C)
    time.sleep(0.2)
    dxinput.ReleaseKey(0x1C)
    time.sleep(1.0)

def take_action(result):
    for i in range(5):
        if i == 0:
            key = 0xC8 #UP ARROW
        elif i == 1:
            key = 0xD0 #DOWN ARROW
        elif i == 2:
            key = 0xCB # LEFT ARROW
        elif i == 3:
            key = 0xCD # RIGHT ARROW
        else:
            key = 0x38 # LEFT ALT
        if result[i] >= 0.5:
            if pressed[i] == 0:
                pressed[i] = 1
            dxinput.PressKey(key)
        else:
            if pressed[i] == 1:
                pressed[i] = 0
            dxinput.ReleaseKey(key)

def clear_presses():
    for i in range(5):
        if i == 0:
            key = 0xC8
        elif i == 1:
            key = 0xD0
        elif i == 2:
            key = 0xCB
        elif i == 3:
            key = 0xCD
        else:
            key = 0x38
        if pressed[i] == 1:
            dxinput.ReleaseKey(key)
            pressed[i] = 0

def process_image(img):
    img = img.copy()
    img[img<150]=0
    img[img>0]=255
    img[55-2:55+2,75-2:75+2]=100
    img=img[55-32:55+32,75-32:75+32]
    return img

model = keras.models.load_model("cnnv2.h5")

restart_race()
clear_presses()
while True:
    img = ss.get_screen(screen_name)
    speed = digits.get_speed(img)
    img = img[420:-70,80:230,2]
    print(img.shape)
    img = process_image(img)
    img = img.reshape(1,64,64,1)
    result = model.predict([img,speed])
    print(result)
    take_action(result[0])