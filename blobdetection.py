# Created by rajan at 02/12/19

import numpy as np
import cv2
import ImageProcUtils as ip
import math

test_image=cv2.imread('/home/rajan/DIS/HW1/TestImages4Project/butterfly.jpg', -1)
test_image_gray=cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)

initial_scale=(1/math.sqrt(2))
#initial_scale=3
k=math.sqrt(2)
#k=2

output=ip.laplacianscale(test_image_gray, k, initial_scale)