# Created by rajan at 09/11/19
import math

import cv2
import numpy as np
from enum import Enum

class selection(Enum):

    freeform = 0
    rectangle = 1
    ellipse=2

contours=[]
pts=[]
mouse_click=False
count=0
major_axis=0
minor_axis=0
centre=(0,0)

def getLength(p1,p2):

    length=abs(int(math.sqrt(pow((p2[0]-p1[0]),2)+pow((p2[1]-p1[1]),2))))
    return length

def getmaskfreeform(event,x,y,flags,param):
    global mouse_click
    global count

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

def getmaskrectangle(event,x,y,flags,param):
    global mouse_click
    global count

    if event==cv2.EVENT_LBUTTONDOWN:

        mouse_click=True
        pts.clear()
        pts.append((x,y))

    if event==cv2.EVENT_LBUTTONUP:

        mouse_click=False
        pts.append((x,y))

        cv2.rectangle(clone, pts[0], pts[1], (0, 255, 0), 2)
        cv2.rectangle(mask, pts[0], pts[1], 255, -1)
        cv2.imshow("Mask", mask)

def getmaskellipse(event,x,y,flags,params):
    global mouse_click
    global count
    global centre
    global minor_axis
    global major_axis

    if event==cv2.EVENT_LBUTTONDOWN:

        mouse_click=True
        pts.clear()
        pts.append((x,y))

    if event==cv2.EVENT_LBUTTONUP:

        count=count+1
        pts.append((x,y))

        if count==1:
            centre=(int((pts[0][0]+pts[1][0])/2),int((pts[0][1]+pts[1][1])/2))
            minor_axis=getLength(pts[0],pts[1])
            cv2.line(clone, pts[0], pts[1], (0, 255, 0), 2)

        if count==2:
            mouse_click=False
            count=0
            major_axis=getLength(pts[0],pts[1])
            cv2.line(clone, pts[0], pts[1], (0, 255, 0), 2)
            cv2.ellipse(clone,centre,(int(minor_axis/2),int(major_axis/2)),0.0,0.0,360.0,(0,255,0),2)
            cv2.ellipse(mask,centre,(int(minor_axis/2),int(major_axis/2)),0.0,0.0,360.0,255,-1)
            cv2.imshow("Mask", mask)

def guiMaskSelect(img, select):
    global mask
    global clone
    clone=img.copy()
    mask = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    cv2.namedWindow("Choose your selection")

    if select == selection.freeform.value:

        cv2.setMouseCallback("Choose your selection", getmaskfreeform)

    elif select == selection.rectangle.value:

        cv2.setMouseCallback("Choose your selection", getmaskrectangle)
    elif select == selection.ellipse.value:
        cv2.setMouseCallback("Choose your selection", getmaskellipse)
    else:
        cv2.setMouseCallback("Choose your selection", getmaskfreeform)
    while True:
        cv2.imshow("Choose your selection", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()

    return mask
