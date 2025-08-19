import cv2
import os

def crop_image(image_path, bounding_box, output_path):
    """
    Crops an image based on a bounding box.

    Args:
        image_path (str): Path to the input image.
        bounding_box (tuple): (x, y, w, h) coordinates of the bounding box.
        output_path (str): Path to save the cropped image.
    """
    img = cv2.imread(image_path)

    x, y, w, h = bounding_box

    cropped_img = img[y:y+h, x:x+w]

    cv2.imwrite(output_path, cropped_img)
    
# Example usage:
image_path = "C:/catan_universe_project/catan_dataset/catan_train_dataset/images/val/"
labels_path = "C:/catan_universe_project/catan_dataset/catan_train_dataset/labels/val/"
output_path = 'C:/catan_universe_project/catan_dataset/cropped_detections/'

def find_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None



for dirpath, dirnames, filenames in os.walk(image_path):
    for filename in filenames:
        #print(filename)
        replaced_filename = filename.replace('png','txt')
        replaced_filename = replaced_filename.replace('jpg','txt')
        label_file = find_file(labels_path,filename.replace('png','txt'))

        with open(label_file, 'r') as file:
            count = 0
            for line in file:
                
                replaced_line = line.replace('\n','')
                boundings = replaced_line.split(' ')

                center_width = int(float(boundings[1])*1920)
                center_height = int(float(boundings[2])*1200)

                width = int(float(boundings[3])*1920)
                height = int(float(boundings[4])*1200)

                left_width = center_width - int(float(boundings[3])*1920/2)
                left_hight = center_height - int(float(boundings[4])*1200/2)

                bounding_box = (left_width, left_hight, width, height)
                crop_image(image_path+filename, bounding_box, output_path+str(count)+ '_' + filename)
                count = count + 1
                
                
                #print(boundings) # strip() removes the newline character
        
    
        #crop_image(image_path, bounding_box, output_path)
        
        
'''
center_width = int(0.437646*1920)
center_height = int(0.129527*1200)

width = int(0.044969*1920)
height = int(0.042399*1200)

left_width = center_width - int(0.044969*1920/2)
left_hight = center_height - int(0.042399*1200/2)

bounding_box = (left_width, left_hight, width, height) # Example bounding box
output_path = "C:/catan_universe_project/catan_dataset/cropped_image.jpg"
crop_image(image_path+'101.png', bounding_box, output_path)
'''