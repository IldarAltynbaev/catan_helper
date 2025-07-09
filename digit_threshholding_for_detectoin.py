import cv2
import os
import numpy as np


hsv_green1 = np.asarray([52, 92, 154])   
hsv_green2 = np.asarray([179, 161, 255])   

hsv_red1 = np.asarray([0, 117, 175])   
hsv_red2 = np.asarray([3, 178, 255])   

hsv_red3 = np.asarray([175, 117,175])   
hsv_red4 = np.asarray([179, 178, 255]) 

#hsv_red1 = np.asarray([0, 91, 0])   
#hsv_red2 = np.asarray([4, 144, 255])   

image_path = "C:\catan_universe_project\catan_dataset\digit_detection_dataset\\negative/".replace('\\', "/")
output_path = 'C:/catan_universe_project/catan_dataset/digit_detection_dataset/grayscale/'.replace('\\', "/")

#box_img = cv2.imread('C:/catan_universe_project/catan_dataset/digit_detection_dataset/positive')


for dirpath, dirnames, filenames in os.walk(image_path):
    for filename in filenames:

        box_img = cv2.imread(image_path+filename)
        #gray_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2GRAY)    

        b, g, r = cv2.split(box_img)

   
        inverted_img = cv2.merge([b, r, g])
        #mask_green = cv2.inRange(box_img, hsv_green1, hsv_green2)
        #mask_green = np.repeat(np.expand_dims(mask_green,2), 3,axis=2)
        #cv2.imwrite(output_path+filename, mask_green)

        #mask_red1 = cv2.inRange(box_img, hsv_red1, hsv_red2)
        #mask_red2 = cv2.inRange(box_img, hsv_red3, hsv_red4)
        #mask_red = mask_red1 + mask_red2
        cv2.imwrite(output_path+filename, inverted_img)

