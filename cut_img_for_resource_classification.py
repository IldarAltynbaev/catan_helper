import cv2
import os

image_path = r"C:\catan_universe_project\catan_dataset\catan_resource_dataset\\temp/".replace("\\","/")
output_path = r'C:\catan_universe_project\catan_dataset\catan_resource_dataset\\temp/'.replace("\\","/")


for dirpath, dirnames, filenames in os.walk(image_path):
    for filename in filenames:
        box_img = cv2.imread(image_path+filename)
        _, width, _ = box_img.shape
        width_cutoff = int(width / 2.3)
        cropped_box_img = box_img[:, width_cutoff:]
        cv2.imwrite(output_path+filename, cropped_box_img)
