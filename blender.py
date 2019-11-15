# Created by rajan at 09/11/19

import cv2
import ImgStruct
import numpy as np
import ImageProcUtils as iputil

numlayers=4                                         #specify the number of pyramids layers here

lena_src_image = cv2.imread('/home/rajan/DIS/HW1/lena_blend.png', -1)   #use it for image pair 1
#lena_src_image = cv2.imread('/home/rajan/DIS/HW1/pennywise.png', -1)   #use it for image pair 2
#lena_src_image = cv2.imread('/home/rajan/DIS/HW1/trump.png', -1)       #use it for image pair 3
lena_src_image=lena_src_image[:,:,:3]
lena_target_image = cv2.imread('/home/rajan/DIS/HW1/lena.png', -1)      #use it for image pair 1
#lena_target_image = cv2.imread('/home/rajan/DIS/HW1/ballon_vendor.png', -1)    #use it for image pair 2
#lena_target_image = cv2.imread('/home/rajan/DIS/HW1/lincoln-portrait.jpg', -1)     use it for image pair 3
lena_target_image=lena_target_image[:,:,:3]
lena_src_gray=cv2.cvtColor(lena_src_image,cv2.COLOR_BGR2GRAY)
lena_target_gray=cv2.cvtColor(lena_target_image,cv2.COLOR_BGR2GRAY)
channels=lena_target_image.ndim
output=np.zeros((lena_src_image.shape[0], lena_src_image.shape[1]))


target_image=ImgStruct.Image(lena_target_image,"target image") #use it for image pair 1 and 2
#target_image=ImgStruct.Image(lena_target_gray,"target image") #use it for image pair 3
target_image.setPyramids(numlayers)
target_image.display_image_list(target_image.gaussian)

src_image=ImgStruct.Image(lena_src_image,"Source Image",target_image,(169,207)) #use it for image pair 2 which contains prealigned starting point, the starting point doesnt affect image pair 1 as they are of equal size.
#src_image=ImgStruct.Image(lena_src_gray,"Source Image",target_image,(169,0))   #use it for image pair 3 with praligned starting coordinates.
src_image.setPyramids(numlayers)
src_image.display_image_list(src_image.gaussian)

src_image.callGUI()

src_mask=ImgStruct.Image(src_image.mask,"source image mask")
src_mask.setPyramids(numlayers)
src_mask.normalize_mask()
src_mask.display_image_list(src_mask.gaussian)

src_mask_complement=ImgStruct.Image(src_image.mask_complement,"source mask complement")
src_mask_complement.setPyramids(numlayers)
src_mask_complement.normalize_mask()
src_mask_complement.display_image_list(src_mask_complement.gaussian)

result_laplacian=iputil.ImageBlend(src_image, src_mask, target_image, src_mask_complement)

output_image=ImgStruct.Image(output,"blended image")
output_image.setnumlayers(numlayers)
output_image.image_reconstruct(result_laplacian,channels)
output_image.display_image(output_image.img)
