# Created by rajan at 09/11/19

#A class for storing an image and doing all the operations and stroring their respective features

import cv2
import numpy as np
import ImageProcUtils
import guiInterface

class Image():

    gauss=ImageProcUtils.getGaussKernel(2)
    laplacian=[]
    gaussian=[]
    numlayers=0

    def __init__(self, img,name,target_img=None,startpoint=(0,0)):
        self.name=name
        self.img=img
        if target_img!=None:
            target_h, target_w = target_img.getshape()
            if img.shape[0]!=target_h or img.shape[1]!=target_w:
                self.refit(target_img,startpoint)
            else:
                pass
        else:
            pass

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')

    def setPyramids(self, numlayers):

        self.numlayers=numlayers
        self.laplacian, self.gaussian=ImageProcUtils.computePyr(self.img, self.numlayers)
    def setnumlayers(self,numlayer):

        self.numlayers=numlayer

    def getshape(self):

        return self.img.shape[0], self.img.shape[1]

    def callGUI(self):

        selection = input("Choose the type of selection 1.Free Form 2.Rectangle 3.Ellipse")

        self.mask = guiInterface.guiMaskSelect(self.img,int(selection)-1)
        self.mask_complement=255-self.mask

    def display_image(self,img):

        cv2.imshow(self.name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_image_list(self,list):

        for i in range(0,len(list)):

            cv2.imshow(self.name+" layer "+str(i), list[i])
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def image_reconstruct(self,laplacian_list,channels):

        self.laplacian=laplacian_list
        self.img=ImageProcUtils.lapCollapse(self.laplacian, self.gauss,self.numlayers, channels)

    def normalize_mask(self):

        for i in range(0,len(self.gaussian)):
            self.gaussian[i]=(np.around(self.gaussian[i]/255)*255).astype("uint8")

    def refit(self, tgt_img, startpoint):

        if tgt_img.img.ndim==3:
            padded_image=np.zeros((tgt_img.img.shape[0], tgt_img.img.shape[1], 3),dtype="uint8")
        else:
            padded_image=np.zeros((tgt_img.img.shape[0], tgt_img.img.shape[1]),dtype="uint8")

        padded_image[startpoint[1]:startpoint[1]+self.img.shape[0], startpoint[0]:startpoint[0]+self.img.shape[1]]=self.img

        self.img = padded_image
