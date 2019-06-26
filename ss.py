import win32gui
import win32ui
import win32con
import cv2
import numpy as np

def get_screen(screen_name):
    #initialize the required resources
    hwnd = win32gui.FindWindow(None,screen_name)
    win32gui.SetForegroundWindow(hwnd)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, 1000, 600)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(1000, 600) , dcObj, (0,0), win32con.SRCCOPY)
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (600, 1000, 4)
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    return img