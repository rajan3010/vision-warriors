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

    def __init__(self, img,name):
        self.name=name
        self.img=img
        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype='uint8')

    def setPyramids(self, numlayers):

        self.numlayers=numlayers
        self.laplacian, self.gaussian=ImageProcUtils.computePyr(self.img, self.numlayers)
    def setnumlayers(self,numlayer):

        self.numlayers=numlayer

    def getshape(self):

        return self.img.shape[0], self.img.shape[1]

    def callGUI(self):

        selection = input("Choose the type of selection 1.Free Form 2.Rectangle")

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
