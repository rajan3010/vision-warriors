# Created by rajan at 11/10/19

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from enum import Enum
import math
import matplotlib.pyplot as plot
import time
#import pyfftw

start_time=0
after_conv_time=0
end_time=0

class Padtype(Enum):
    ZERO=0
    COPY=1
    REFLECT=2
    WRAPAROUND=3

def makepaddingzero(img, pad_width_left,pad_width_right, pad_height_top, pad_height_bottom):
    try:
        img_h, img_w = img.shape[:2]

        if img.ndim == 3:
            padded_image=np.zeros((img_h + pad_height_top + pad_height_bottom, img_w + pad_width_left + pad_width_right, 3),dtype="float32")
        else:
            padded_image=np.zeros((img_h + pad_height_top+ pad_height_bottom, img_w + pad_width_left + pad_width_right),dtype="float32")

        padded_image[pad_height_top:pad_height_top + img_h, pad_width_left:pad_width_left + img_w] = img;
    except:
        print("kernel size exceeded image size")
        exit()

    return padded_image

def makepadding(img, pad_width, pad_height, padtype):

    img_h, img_w = img.shape[:2]

    if img.ndim == 3:
        padded_image=np.zeros((img_h+2*pad_height, img_w+2*pad_width, 3),dtype="uint8")
    else:
        padded_image=np.zeros((img_h+2*pad_height, img_w+2*pad_width),dtype="uint8")

    padded_image[pad_height:pad_height + img_h, pad_width:pad_width + img_w] = img;

    if padtype == Padtype.ZERO:
        pass

    elif padtype == Padtype.COPY:
        for y in range(0,pad_height):
            padded_image[y,pad_width:pad_width + img_w]=padded_image[pad_height,pad_width:pad_width + img_w]           # TOP PADDING
        for y in range(pad_height+img_h, img_h+2*pad_height):
            padded_image[y,pad_width:pad_width + img_w]=padded_image[pad_height +img_h -1,pad_width:pad_width + img_w] # BOTTOM PADDING
        for x in range(0,pad_width):
            padded_image[0:img_h+2*pad_height, x] = padded_image[0:img_h+2*pad_height, pad_width]           # LEFT PADDING
        for x in range(pad_width+img_w,img_w+2*pad_width):
            padded_image[0:img_h+2*pad_height, x] = padded_image[0:img_h+2*pad_height, pad_width+img_w-1]   # RIGHT PADDING

    elif padtype == Padtype.REFLECT:
         padded_image[0:pad_height, pad_width:pad_width+img_w] = padded_image[2*pad_height-1:pad_height-1:-1,pad_width:pad_width+img_w]                           # TOP PADDING
         padded_image[pad_height+img_h:img_h+2*pad_height, pad_width:pad_width+img_w] = padded_image[pad_height+img_h-1:img_h-1:-1, pad_width:pad_width + img_w]   # BOTTOM PADDING
         padded_image[0:img_h+2*pad_height,0:pad_width] = padded_image[0:img_h+2*pad_height, 2*pad_width-1:pad_width-1:-1]                            # LEFT PADDING
         padded_image[0:img_h+2*pad_height, pad_width+img_w:img_w+2*pad_width] = padded_image[0:img_h+2*pad_height, pad_width+img_w-1:img_w-1:-1]    # RIGHT PADDING

    elif padtype == Padtype.WRAPAROUND:
        padded_image[0:pad_height, pad_width:pad_width + img_w] = padded_image[img_h:img_h+pad_height,pad_width:pad_width+img_w]                      # TOP PADDING
        padded_image[pad_height+img_h:img_h+2*pad_height, pad_width:pad_width+img_w] = padded_image[pad_height:2*pad_height,pad_width:pad_width+img_w]               # BOTTOM PADDING
        padded_image[0:img_h+2*pad_height, 0:pad_width] = padded_image[0:img_h+2*pad_height, img_w:img_w+pad_width]                        # LEFT PADDING
        padded_image[0:img_h+2*pad_height, pad_width+img_w:img_w+2*pad_width] = padded_image[0:img_h+2*pad_height, pad_width:2*pad_width]             # RIGHT PADDING

    else:
        pass

    return padded_image


def conv2(f,win,pad):

    #f= f.astype("float32")     #change it to float before convolution

    img_height, img_width = f.shape[:2]
    win_height, win_width = win.shape[:2]
    pad_width= win_width//2
    pad_height=win_height//2

    #image = cv2.copyMakeBorder(f, pad_width, pad_width, pad_width, pad_width,cv2.BORDER_CONSTANT)
    image=makepadding(f, pad_width, pad_height, pad)

    if f.ndim ==3:
        convoluted_image= np.zeros((img_height, img_width,3), dtype="float32" )
    else:
        convoluted_image = np.zeros((img_height, img_width), dtype="float32")

    image=image.astype("float32");

    for h in range(0,img_height):
        for w in range(0,img_width):

            roi=image[h:h + win_height, w:w + win_width]

            if f.ndim==3:
                k = roi * win[:,:,None]; #broadcasting single dimensional filter to 3 layers
                convoluted_image[h,w,0] = k[:,:,0].sum()
                convoluted_image[h, w, 1] = k[:, :, 1].sum()
                convoluted_image[h, w, 2] = k[:, :, 2].sum()
            else:
                k = roi * win;
                convoluted_image[h, w] = k.sum()

    convoluted_image = rescale_intensity(convoluted_image, in_range=(0, 255))
    #convoluted_image = (convoluted_image*255).astype("uint8")

    return convoluted_image

def lineartransform(f, inmax, inmin, outmax, outmin):

    T=(outmax-outmin)/(inmax-inmin)

    img_h, img_w= f.shape[:2]

    output = np.zeros((img_h, img_w))

    for y in range(0,img_h):
        for x in range(0,img_w):

            output[y, x]= f[y, x]*T

    output=output.astype("float32")

    return output

def DFT2(f):
    global img
    global normalizedImg1
    intensity = []
    img = f

    x, y = img.shape[:2]

    normalizedImg1 = np.zeros((x, y))
    ##################################
    for i in range(0, x, 1):
        for j in range(0, y, 1):
            intensity.append((int(img[i, j])))  # storing the intensity values of all the pixels in a list
    c = min(intensity)  # finding minimum intensity value
    d = max(intensity)  # finding maximum intensity value
    #print(c, d)

    a = 0
    b = 1

    for i in range(0, x, 1):
        for j in range(0, y, 1):
            intensityc = int(img[i, j])
            val = ((((intensityc - c) * ((b - a) / (d - c))) + a))  # linear transformation
            normalizedImg1.itemset((i, j), val)  # setting the pixels to the new intensity value
    ##################################

    f1 = np.fft.fft(normalizedImg1, n=None, axis=0, norm=None)
    f2 = np.fft.fft(f1, n=None, axis=1, norm=None)
    f2conj = np.conj(f2)
    return f2conj

def DFT2k(f):
    global img

    # a = np.array(img, dtype ='float32')
    f1 = np.fft.fft(f, n=None, axis=0, norm=None)
    f2 = np.fft.fft(f1, n=None, axis=1, norm=None)
    f2conj = np.conj(f2)
    return f2conj

def IDFT2(f2conj):
    x, y = f2conj.shape[:2]
    f3 = np.fft.fft(f2conj, n=None, axis=0, norm=None)
    f4 = np.fft.fft(f3, n=None, axis=1, norm=None)
    f4conj = np.conj(f4) / ((x - 1) * (-1 + y))
    g = np.real(np.array(f4conj))
    return g



def conv_fast(original_image, kernel):

    sz = original_image.shape
    sz = (sz[0] - kernel.shape[0], sz[1] - kernel.shape[1])  # total amount of padding
    kernel2 = makepaddingzero(np.array(kernel), (sz[1]+1)//2, sz[1]//2, (sz[0]+1)//2, sz[0]//2)
    kernel2 = np.fft.ifftshift(kernel2)
    filtered = np.real(IDFT2(DFT2(original_image) * DFT2k(kernel2)))
    return (filtered)


def downsample(img):

    global down_sampled_image

    img_height, img_width = img.shape[:2]
    #print(img.ndim)
    if img.ndim == 3:
        down_sampled_image = np.zeros((int((img_height + 1) / 2), int((img_width + 1) / 2), 3), dtype="float32")
    else:
        down_sampled_image = np.zeros((int((img_height + 1) / 2), int((img_height + 1) / 2)), dtype="float32")

    down_sampled_image=img[0:img_height:2, 0:img_width:2]

    return down_sampled_image

def upsample(img,size_reference_list,intensity_reference_list, index):

    if img.ndim == 3:
        upsampled_image = np.zeros((size_reference_list[index + 1].shape[0], size_reference_list[index + 1].shape[1], 3), dtype="uint8")
    else:
        upsampled_image = np.zeros((size_reference_list[index + 1].shape[0], size_reference_list[index + 1].shape[1]), dtype="uint8")

    upsampled_image[0:size_reference_list[index + 1].shape[0]:2, 0:size_reference_list[index + 1].shape[1]:2] = intensity_reference_list[index][0:size_reference_list[index].shape[0],0:size_reference_list[index].shape[1]]
    upsampled_image[0:size_reference_list[index + 1].shape[0]:2, 1:size_reference_list[index + 1].shape[1]:2] = upsampled_image[0:upsampled_image.shape[0] - 1:2,0:upsampled_image.shape[1] - 1:2]
    upsampled_image[1:size_reference_list[index + 1].shape[0]:2, 0:size_reference_list[index + 1].shape[1]] = upsampled_image[0:upsampled_image.shape[0] - 1:2,0:upsampled_image.shape[1]]
    upsampled_image = upsampled_image.astype('uint8')

    return upsampled_image

def computeGaussian(img, numlayers, gaussian_list, gaussian_kernel):

    global sampled_image

    if len(gaussian_list) >=numlayers:
        return gaussian_list

    else:

        gaussian_list.append(img)
        img=conv2(img, gaussian_kernel, 0)
        down_sampled_image=downsample(img)
        gaussian_list=computeGaussian(down_sampled_image, numlayers, gaussian_list, gaussian_kernel)

    return gaussian_list

def computeLaplacian(img,numlayers, gaussian_list, gaussian_kernel,channels):

    global gaussian_blur
    gaussian_list = gaussian_list[::-1]
    laplacian_list = [gaussian_list[0]]

    for i in range(0,numlayers-1):

        if channels == 3:
            gaussian_blur=np.zeros((gaussian_list[i+1].shape[0], gaussian_list[i+1].shape[1], 3), dtype="float32")
        else:
            gaussian_blur = np.zeros((gaussian_list[i + 1].shape[0], gaussian_list[i + 1].shape[1]), dtype="float32")

        upsampled_image=upsample(img, gaussian_list,gaussian_list, i)

        gaussian_blur=conv2(upsampled_image,gaussian_kernel,0)
        laplacian_layer=np.subtract(gaussian_list[i+1],gaussian_blur)
        laplacian_list.append(laplacian_layer)

    #laplacian_list= laplacian_list[::-1]

    return laplacian_list



def computePyr(img, numlayers):

    img_h, img_w = img.shape[:2]
    max_layers=1+math.ceil(math.log(math.sqrt(img_h*img_w)/4, 2));

    downscale=2
    sigma = 1
    gauss_kern_size = int(6*sigma-1)
    g_x = cv2.getGaussianKernel(gauss_kern_size, sigma)
    g_y = cv2.getGaussianKernel(gauss_kern_size, sigma)
    g_xy= g_x*g_y.transpose()
    channel=img.ndim

    if numlayers> max_layers:
        print("Invalid number of pyramid layers")
        exit()

    else:
        gaussian_pyramid=[]
        gaussian_pyramid=computeGaussian(img, numlayers, gaussian_pyramid,g_xy)
        laplacian_pyramid=computeLaplacian(img, numlayers,gaussian_pyramid,g_xy,channel)

    return laplacian_pyramid, gaussian_pyramid[::-1]

def lapCollapse(laplacian_pyramid, gauss_kern, num_layers, channels):
    global gaussian_smoothening

    reconstructed_image = [laplacian_pyramid[0]]

    for i in range(0,num_layers-1):

        if channels == 3:
            gaussian_smoothening= np.zeros((laplacian_pyramid[i + 1].shape[0], laplacian_pyramid[i + 1].shape[1], 3),
                                     dtype="uint8")
        else:
            gaussian_smoothening= np.zeros((laplacian_pyramid[i + 1].shape[0], laplacian_pyramid[i + 1].shape[1]),
                                     dtype="uint8")

        upsampled_image=upsample(reconstructed_image[0],laplacian_pyramid,reconstructed_image,i)
        gaussian_smoothening=conv2(upsampled_image,gauss_kern,0)
        reconstructed_layer=np.add(laplacian_pyramid[i+1],gaussian_smoothening)
        reconstructed_image.append(reconstructed_layer)
        '''cv2.imshow('t',laplacian_pyramid[i+1])
        cv2.waitKey(0)'''
        #recontructed_output=reconstructed_image[-1].astype("uint8")

    return reconstructed_image[-1]

def getGaussKernel(sigma):
    # Obtaining a Gaussain Kernel
    global maxdim
    maxdim = int(np.ceil(2 * sigma) - 0.5)
    kernel_temp = cv2.getGaussianKernel((maxdim * 2) + 1, sigma)
    g_kernel = kernel_temp * kernel_temp.transpose()
    return (g_kernel)

def ImageBlend(src_image,src_mask,target_image,target_mask):

    blended_laplacian=[]

    for i in range(0,src_image.numlayers):

        #gaussian_mask=lineartransform(src_mask.gaussian[i],255,0,1,0)
        #gaussian_mask_inverse=lineartransform(target_mask.gaussian[i],255,0,1,0)

        '''result_image=np.add(src_image.laplacian[i]*(src_mask.gaussian[i][:,:,None]//255),target_image.laplacian[i]*(target_mask.gaussian[i][:,:,None]//255))
        result_image=(result_image*255).astype("uint8")'''
        result_image=np.add(cv2.bitwise_and(src_image.laplacian[i],src_image.laplacian[i],mask=src_mask.gaussian[i]), cv2.bitwise_and(target_image.laplacian[i], target_image.laplacian[i],mask=target_mask.gaussian[i]))
        blended_laplacian.append(result_image)

    '''cv2.imshow("test",blended_laplacian[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()'''
    return blended_laplacian

def GaussianSmoothening(img, sigma):
    gker = getGaussKernel(sigma)
    imf = conv_fast(img, gker)
    return (imf)

def laplacianscale(img, k, initial_scale):
    global start_time
    global after_conv_time

    start_time=time.time()
    scale_size=10
    laplacian=[]
    sigma_list=[]

    sigma_init=pow(k,0) * initial_scale
    g_init = getGaussKernel(sigma_init)
    gauss_init=conv_fast(img, g_init)

    for i in range(0, scale_size-1):

        sigma_next=pow(k, i+1) * initial_scale
        g_next = getGaussKernel(sigma_next)
        gauss_next = conv_fast(img, g_next)

        normalized_dog=gauss_next-gauss_init
        '''cv2.imshow("test",normalized_dog)
        cv2.waitKey(0)'''
        laplacian.append(normalized_dog)
        sigma_list.append(sigma_init)
        gauss_init=gauss_next
        sigma_init=sigma_next

    after_conv_time=time.time()
    size=len(laplacian)
    laplacian_np=np.array(laplacian)
    coordinates=list(set(keypointdetect(img, laplacian_np,size,sigma_list)))
    coordinates=np.array(coordinates)

    return laplacian

def laplacianoctave(img, k, initial_scale, num_layers):

    img_h, img_w = img.shape[:2]
    max_layers=1+math.ceil(math.log(math.sqrt(img_h*img_w)/4, 2));

    if num_layers> max_layers:
        print("Invalid number of pyramid layers")
        exit()

    else:

        laplacian_octave=[]
        laplacian_temp_out=[]

        for i in range(0,num_layers):

            if i > 0:
                img = downsample(img)
                initial_scale = 2*initial_scale

            laplacian_temp_out = laplacianscale(img, k, initial_scale)

            laplacian_octave.append(laplacian_temp_out)

    return laplacian_octave

def keypointdetect(img,laplacianscale,size,sigma_list):
    global end_time

    img_h,img_w= img.shape[:2]
    keypoints=[]


    fig, a=plot.subplots()
    nh, nw= img.shape
    a.imshow(img, interpolation="nearest", cmap="gray")
    c=None

    for i in range(0,size):

        window_size=int(4*sigma_list[i] + 0.5)
        pad=window_size//2

        for y in range(pad, img_h - pad):
            for x in range(pad, img_w - pad):
                max = laplacianscale[:, y-1:y+2, x-1:x+2]

                z, m, n = np.unravel_index(max.argmax(), max.shape)

                if z == i and m == 1 and n == 1:

                    result=np.amax(max)
                    laplacianscale[:,y-pad:y+pad+1,x-pad:x+pad+1]=1
                    if result>=0.035:
                        #keypoints.append((x,y,window_size))
                        pass
                        c = plot.Circle((x, y), window_size/ 2, color='red', linewidth=1.0, fill=False)
                        a.add_patch(c)
                else:
                    pass
    end_time=time.time()
    print(start_time)
    print(end_time)
    print(end_time-after_conv_time)
    print(end_time-start_time)

    a.plot()
    plot.show()
    exit()

    return keypoints



