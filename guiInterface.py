# Created by rajan at 09/11/19
import cv2
import numpy as np
from enum import Enum

class selection(Enum):

    free_form = 0
    rectangle = 1

contours=[]
pts=[]
mouse_click=False

def getmaskfreeform(event,x,y,flags,param):
    global mouse_click

    if event==cv2.EVENT_LBUTTONDOWN:

        mouse_click=True
        pts.clear()
        contours.clear()

    if event==cv2.EVENT_LBUTTONUP:

        mouse_click=False
        if len(pts)>2:

            contours.append(pts)
            contour_np=np.asarray(contours)

            cv2.drawContours(mask, [contour_np], 0,255, -1)
            cv2.imshow("Mask",mask)

    if mouse_click:
        if len(pts)>2:
            cv2.line(clone,(x,y),pts[len(pts)-1],(0,255,0),2)

        pts.append((x,y))


def guiMaskSelect(img, select):
    global mask
    global clone
    clone=img.copy()
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    cv2.namedWindow("Choose your free form selection")

    if select == selection.free_form:
        cv2.setMouseCallback("Choose your free form selection ", getmaskfreeform)
    elif select == selection.rectangle:
        cv2.setMouseCallback("Choose your rectangle selection", getmaskfreeform)
    else:
        cv2.setMouseCallback("Choose your free form selection", getmaskfreeform)

    while True:
        cv2.imshow("Choose your free form selection", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()

    return mask
