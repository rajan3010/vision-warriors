# Created by rajan at 09/11/19

import cv2
import ImgStruct
import numpy as np
import ImageProcUtils as iputil

lena_src_image = cv2.imread('/home/rajan/DIS/HW1/lena_blend.png', -1)
lena_target_image = cv2.imread('/home/rajan/DIS/HW1/lena.png', -1)
channels=lena_target_image.ndim
output=np.zeros((lena_src_image.shape[0], lena_src_image.shape[1]))

src_image=ImgStruct.Image(lena_src_image[:,:,:3],"Source Image")
src_image.setPyramids(3)
src_image.display_image_list(src_image.gaussian)

src_image.callGUI()

src_mask=ImgStruct.Image(src_image.mask,"source image mask")
src_mask.setPyramids(3)
src_mask.display_image_list(src_mask.gaussian)

src_mask_complement=ImgStruct.Image(src_image.mask_complement,"source mask complement")
src_mask_complement.setPyramids(3)
src_mask_complement.display_image_list(src_mask_complement.gaussian)

target_image=ImgStruct.Image(lena_target_image,"target image")
target_image.setPyramids(3)
target_image.display_image_list(target_image.gaussian)

result_laplacian=iputil.ImageBlend(src_image, src_mask, target_image, src_mask_complement)

output_image=ImgStruct.Image(output,"blended image")
output_image.setnumlayers(3)
output_image.image_reconstruct(result_laplacian,channels)
output_image.display_image(output_image.img)
