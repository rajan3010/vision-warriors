import math
import matplotlib.pyplot as plt
import time
import cv2
import ImageProcUtils as ip
import numpy as np
intensity = []
global inx
global iny
inx = 0
iny = 0
start_time = 0
end_time = 0

global numlay

sigmaarray = []

def LaplacianScale(g_image, scalenum, initsigma, k):
    global numlay
    global sigma
    dog_list = []
    gauss_list = []
    sigma = initsigma
    sigmaarray.append(sigma)
    gauss_list.append(ip.GaussianSmoothening(g_image, sigma))

    for j in range(1, scalenum):
            sigma = k * sigma
            sigmaarray.append(sigma)
            gauss_list.append(ip.GaussianSmoothening(g_image, sigma))
            dog_list.append(gauss_list[j]-gauss_list[j-1])
    return dog_list


def detect_blob(image_np):
    key_points = []
    global window_size
    for m in range(0, image_np.shape[0]):
        count=0
        window_size = (int(np.ceil(2*sigmaarray[m])-0.5)*2)+1
        dim = window_size//2
        (h, w) = image_np[m].shape
        for i in range(dim, h - dim):
            for j in range(dim, w - dim):
                slice_img = image_np[:, i - 1:i + 2, j - 1:j + 2]
                z, x, y = np.unravel_index(slice_img.argmax(), slice_img.shape)

                if z == m and x == 1 and y == 1:
                    result = np.amax(slice_img)
                    if result >=0.035:
                        #count=count+1
                        key_points.append((i, j, m))
                        image_np[:, i - dim:i + dim+1, j - dim:j + dim+1] = 1

        #print((sigmaarray[m],count))
    return key_points




def printblobcircles(img,scale_num=10, initsigma=0.707106, k=math.sqrt(2)):     #scale_num=number of scales, initsigma=initial scale, k=multiplier
    global start_time
    global end_time
    start_time = time.time()
    key_points = np.array(LaplacianScale(img, scale_num, initsigma, k))
    co_ordinates = list(set(detect_blob(key_points)))
    co_ordinates = np.array(co_ordinates)

    fig, a = plt.subplots()

    a.imshow(image, interpolation='nearest', cmap="gray")

    for blob in co_ordinates:
        x, y, m = blob
        c = plt.Circle((y, x), sigmaarray[m] * math.sqrt(2), color='red', linewidth=1, fill=False)
        a.add_patch(c)

    a.plot()
    end_time = time.time()
    print("total", end_time - start_time)
    plt.show()

# MAIN

#Uncomment (only) the respective pair that you are testing

#image = cv2.imread("butterfly.jpg", 0)
#printblobcircles(image,8,1.0)

#image = cv2.imread("fishes.jpg", 0)
#printblobcircles(image,4,2*math.sqrt(2))

#image = cv2.imread("sunflowers.jpg", 0)
#printblobcircles(image,5,2*math.sqrt(2))

#image = cv2.imread("einstein.jpg", 0)
#printblobcircles(image,6,2*math.sqrt(2))

#image = cv2.imread("cheetah.jpg", 0)
#printblobcircles(image, 9, 1.0)

#image = cv2.imread("ladybug_1.jpg", 0)
#printblobcircles(image,5,2*math.sqrt(2))

#image = cv2.imread("eyes.jpg", 0)
#printblobcircles(image,6,8.0)

#image = cv2.imread("naruto.jpg", 0)
#printblobcircles(image,7,2*math.sqrt(2))
