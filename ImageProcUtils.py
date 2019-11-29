# Created by rajan at 11/10/19

import cv2
import numpy as np
from skimage.exposure import rescale_intensity
from enum import Enum
import math

import pyfftw


class Padtype(Enum):
    ZERO=0
    COPY=1
    REFLECT=2
    WRAPAROUND=3


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
    convoluted_image = (convoluted_image*250).astype("uint8")

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

    global fftw_column_object

    f_h, f_w = f.shape[:2]

    '''column_input = pyfftw.empty_aligned(f_h, dtype='complex128')
    row_input = pyfftw.empty_aligned(f_w, dtype='complex128')
    column_output = pyfftw.empty_aligned(f_h, dtype='complex128')
    row_output = pyfftw.empty_aligned(f_w, dtype='complex128')

    fft_row_object = pyfftw.FFTW(row_input,row_output)
    fftw_column_object = pyfftw.FFTW(column_input, column_output)'''

    f=f.astype('complex128')

    for x in range(0,f_h):

        fft_rowinput = np.copy(f[x, :])

        b=np.fft.fft(fft_rowinput)

        f[x, :]=b
        '''fft_row_object.update_arrays(row_input, row_output)'''

    first_transpose = np.transpose(f)

    for y in range(0,f_w):

        fft_input=np.copy(first_transpose[y,:])

        a=np.fft.fft(fft_input)

        first_transpose[y, :]=a

        '''column_input = np.copy(first_transpose[y, :])
        fftw_column_object.update_arrays(column_input, column_output)

        first_transpose[y, :] = np.copy(column_output)'''

    last_transpose = np.transpose(first_transpose)

    return last_transpose


def IDFT2(f):

    f=np.conj(f)
    fft_image= DFT2(f)
    img_h, img_w= f.shape[:2]

    ifft_image= np.conj(fft_image)/(img_h*img_w)

    return ifft_image

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

    #sigma = 1
    gauss_size = int(6*sigma-1)
    g_x=cv2.getGaussianKernel(gauss_size, sigma)
    g_y=cv2.getGaussianKernel(gauss_size, sigma)
    g_xy= g_x*g_y.transpose()

    return g_xy

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

def gausssmoothening(img,sigma):

    kern_size=int(6*sigma-1)
    g_x=cv2.getGaussianKernel(kern_size, sigma)
    g_y=cv2.getGaussianKernel(kern_size, sigma)
    g_xy=g_x*g_y.transpose()

    img=conv2(img,g_xy,0)

    return img

def laplacianscale(img, k, initial_scale):

    scale_size=5
    laplacian=[]

    for i in range(0, scale_size):

        g1 = getGaussKernel(pow(k, i)*initial_scale)
        g2 = getGaussKernel(pow(k, i+1)*initial_scale)

        img_out = conv2(img, g1-g2, 0)
        laplacian.append(img_out)

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