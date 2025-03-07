import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget
from configparser import ConfigParser
from threading import Thread
import cv2 
import numpy as np 
import easyocr
#import matplotlib.pyplot as plt
#import pyautogui
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image
#from matplotlib import cm
#from shapely.geometry import Polygon
import datetime

#config_file = 'data/settings.ini'
#config = ConfigParser()
#config.read(config_file)
#backgroundImage = config.get("Theme", "background")
#from queenui import Ui_MainWindow

class Catan_player():
        def __init__(self):
            self.previous_detection_time = datetime.datetime.now()
            self.previous_resource_count = ''
            self.previous_recource_type = ''
            self.stone_count = 0
            self.wheat_count = 0
            self.sheep_count = 0
            self.brick_count = 0
            self.wood_count = 0
            self.current_detection_count = 0

class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('1UI')
        MainWindow.setMinimumSize(QtCore.QSize(424, 424))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 1000))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
             
        self.background = QtWidgets.QLabel(self.centralwidget)
        self.background.resize(800, 800)
     
        self.queenBrowser = QtWidgets.QTextBrowser(self.background)
        self.queenBrowser.setGeometry(QtCore.QRect(200, 190, 600, 600))  
        self.queenBrowser.setReadOnly(True)
        self.queenBrowser.setObjectName('queenBrowser')         
      
        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

def move_right_bottom_corner(win):
    screen_geometry = QApplication.desktop().availableGeometry()
    screen_size = (screen_geometry.width(), screen_geometry.height())
    win_size = (win.frameSize().width(), win.frameSize().height())
    x = screen_size[0] - win_size[0]
    y = screen_size[1] - win_size[1]
    win.move(x, 50)
    
class CatanHelperWorker(QObject):
    dataChanged = pyqtSignal(str)

    def start(self):
        Thread(target=self._execute, daemon=True).start()

    def create_blank_model(device, freeze_layers = True):
        model = models.resnet34(pretrained=True)
        if freeze_layers:
            for param in model.parameters():
                #if isinstance(param, torch.nn.Conv2d):
                    param.requires_grad = False

        model.fc = torch.nn.Linear(model.fc.in_features, 5)    
        #model = model.to(device)
        return model
    
    def create_transforms(self):

        pred_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])    
        return pred_transforms

    def initialize_models(self):

        model_yolo = YOLO("best_model_yolo11.pt")
        model_resnet = self.create_blank_model(freeze_layers = True)
        model_resnet.load_state_dict(torch.load('C:/catan_github/catan_helper/best_model_resnet.pt',
                                               map_location=torch.device('cpu')))
        #mode_resnet.to(device)
        model_resnet.eval()

        return model_yolo, model_resnet

    def initialize_reader(self):

        reader = easyocr.Reader(['en'], gpu=True)

        return reader

    def get_resource_name(self, resource_result):

        if resource_result == 0:
            resource_name = 'brick'
        elif resource_result == 1:
            resource_name = 'sheep'
        elif resource_result == 2:
            resource_name = 'stone'
        elif resource_result == 3:
            resource_name = 'wheat'
        elif resource_result == 4:
            resource_name = 'wood'
        
        return resource_name

    def get_current_player_by_region(self, each_box):

        width = each_box.orig_shape[1]
        margin1 = width*1/3
        margin2 = width*1/2
        current_position = each_box.xyxy[0][0]
        if current_position < margin1:
            current_player = 1
        elif (current_position >= margin1) & (current_position < margin2):
            current_player = 2
        else:
            current_player = 3

        return current_player
    
    def apply_ocr_reader(self,mask,allowlist,reader):
    
        text_ = reader.readtext(mask,low_text = 0.1)
        current_text = ''
        for t_, t in enumerate(text_):      
            recognized_text = t[1]
            if recognized_text== '':
                break

            recognized_text = self.replace_wrong_char(recognized_text)
            current_text = current_text + recognized_text

        return current_text

    def replace_wrong_char(self,OCR_string):

        OCR_string = OCR_string.replace('#','+')
        OCR_string = OCR_string.replace('O','0')
        OCR_string = OCR_string.replace('o','0')
        OCR_string = OCR_string.replace('l','1')

        return OCR_string
    
    def get_resource_count_from_img(self, results,
                                        each_box,
                                        reader,
                                        hsv_green1,
                                        hsv_green2,
                                        hsv_red1, 
                                        hsv_red2, 
                                        hsv_red3, 
                                        hsv_red4,
                                        box_img,
                                        allowlist):
      
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'cropped' + str(count) + '.png', box_img)
        box_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2HSV)
                                       
        mask_green = cv2.inRange(box_img, hsv_green1, hsv_green2)

        mask_red1 = cv2.inRange(box_img, hsv_red1, hsv_red2)
        mask_red2 = cv2.inRange(box_img, hsv_red3, hsv_red4)

        mask_red = mask_red1 + mask_red2

        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'cropped' + str(count) + '.png', croppeed_img_quantity)
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'red' + str(count) + '.png', mask_red)
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'green' + str(count) + '.png', mask_green)
        ''' 
        text_ = reader.readtext(mask_green,allowlist=allowlist)
        current_text = ''
        for t_, t in enumerate(text_):      
            if t[1] == '':
                break

            t[1] = self.replace_wrong_char(t[1])
            current_text = current_text + t[1]
        '''

        current_text = self.apply_ocr_reader(mask_green, allowlist,reader)
        
        if current_text == '':

            kernel = np.ones((2,2),np.uint8)
            mask = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
            scale_factor = 2
            mask = cv2.dilate(mask,kernel,iterations = 1)
            closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
            #upscaled = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            #blur = cv2.blur(upscaled, (5, 5))
            #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'red' + str(count) + '.png', blur)
            current_text = self.apply_ocr_reader(opening,allowlist,reader)
            

        
        return current_text, mask_green, mask_red

    def get_resource_type_from_img(self, pred_transforms, model_resnet, box_img,device):

        _, width, _ = box_img.shape
        width_cutoff = int(width / 2.3)
        cropped_box_img = box_img[:, width_cutoff:]
        cropped_box_img = cv2.cvtColor(cropped_box_img, cv2.COLOR_BGR2RGB)
        cropped_box_img = Image.fromarray(cropped_box_img)
        transformed_cropped_img = pred_transforms(cropped_box_img)
        transformed_cropped_img = transformed_cropped_img.to(device)
        pred = model_resnet(transformed_cropped_img.unsqueeze(0))
        resource_result = pred.argmax().item()
        resource_name = self.get_resource_name(resource_result)

        return resource_name

    def check_for_the_same_detection(self,
                                     prior_resource_type_pl,
                                     prior_resource_count_pl,
                                     prior_time_of_detection_pl,
                                     resource_type,
                                     resource_count,
                                     current_time,
                                     delay_delta):
        
        #if (prior_resource_type_pl == resource_type) & \
        if (current_time - prior_time_of_detection_pl) <= delay_delta:   
           #(prior_resource_count_pl == resource_count) & \
           return True   

        return False

    def create_hsv_arrays(self):

        hsv_green1 = np.asarray([50, 0, 0])   
        hsv_green2 = np.asarray([59, 255, 255])   

        hsv_red1 = np.asarray([0, 117, 175])   
        hsv_red2 = np.asarray([3, 178, 255])   

        hsv_red3 = np.asarray([175, 117,175])   
        hsv_red4 = np.asarray([179, 178, 255]) 

        return hsv_green1, hsv_green2, hsv_red1, hsv_red2, hsv_red3, hsv_red4

    def resource_count_is_correct(self,resource_count):
        if len(resource_count) > 1:
            if (resource_count[0] == '+') | (resource_count[0] == '-'):
                    if resource_count[1:].isdigit():

                        return True
            
        return False

    def _execute(self):
              
        reader = self.initialize_reader()
        allowlist = '0123456789+-Oo#l'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_yolo, model_resnet = self.initialize_models()
        model_yolo.to(device)
        model_resnet.to(device)

    
        hsv_green1, hsv_green2, hsv_red1, hsv_red2, hsv_red3, hsv_red4 = self.create_hsv_arrays()   
        pred_transforms = self.create_transforms()

        delay_delta = datetime.timedelta(seconds=0.5)
        
        
        display_text = ''

        player1 = Catan_player()
        player3 = Catan_player()
        
        for results in model_yolo.track(source='screen',conf=0.5, stream = True, show=False):
            for each_box in results.boxes:
                if (each_box.conf > 0.5) & (each_box.id is not None):

                    
                    box_img = results.orig_img[int(each_box.xyxy[0,1]):int(each_box.xyxy[0,3]),\
                                                    int(each_box.xyxy[0,0]):int(each_box.xyxy[0,2])]
                    
                    current_player = self.get_current_player_by_region(each_box)   
                    resource_count,green_pic, red_pic = self.get_resource_count_from_img(results,
                                                     each_box,
                                                     reader,
                                                     hsv_green1,
                                                     hsv_green2,
                                                     hsv_red1, 
                                                     hsv_red2, 
                                                     hsv_red3, 
                                                     hsv_red4,
                                                     box_img,
                                                     allowlist
                                                     )
                   
                    
                    resource_type = self.get_resource_type_from_img(pred_transforms,
                                                                    model_resnet,
                                                                    box_img,device) 
                    
                    green_pic = np.expand_dims(green_pic, axis = 2)
                    green_pic =np.repeat(green_pic, 3 , axis=2)
                    
                    red_pic = np.expand_dims(red_pic, axis = 2)
                    red_pic = np.repeat(red_pic, 3 , axis=2)

                    res_pic = cv2.vconcat([box_img, green_pic, red_pic])

                    if self.resource_count_is_correct(resource_count):
                        if current_player == 1:
                            if datetime.datetime.now() - player1.previous_detection_time > delay_delta:

                                player1.current_detection_count = player1.current_detection_count + 1                                                                                      
                                if display_text.count('\n') > 9:
                                    display_text = ''

                                display_text = display_text + str(current_player) + '_' + str(player1.current_detection_count) +'_' + str(resource_count) +'_' + str(resource_type) + '\n'
                                player1.previous_recource_type = resource_type
                                player1.previous_resource_count = resource_count

                            else:
                                
                                if (resource_type != player1.previous_recource_type) & (resource_count != player1.previous_resource_count):
                                    
                                    if display_text.count('\n') > 9:
                                        display_text = ''
                                    display_text = display_text + str(current_player) + '_' + str(player1.current_detection_count) +'_' + str(resource_count) +'_' + str(resource_type) + '\n'
                                player1.previous_recource_type = resource_type
                                player1.previous_resource_count = resource_count

                            #cv2.imwrite('C:/catan_github/catan_helper/testing/'+'_' + str(current_player) + '_' + str(count_1) +'_' + str(resource_count) +'_' + str(resource_type) + '.png', res_pic)
                            player1.previous_detection_time = datetime.datetime.now()

                        elif current_player == 3:
                            if datetime.datetime.now() - player3.previous_detection_time > delay_delta:

                                player3.current_detection_count = player3.current_detection_count + 1                                                                                      
                                if display_text.count('\n') > 9:
                                    display_text = ''

                                display_text = display_text + str(current_player) + '_' + str(player3.current_detection_count) +'_' + str(resource_count) +'_' + str(resource_type) + '\n'
                                player3.previous_recource_type = resource_type
                                player3.previous_resource_count = resource_count

                            else:
                                
                                if (resource_type != player3.previous_recource_type) & (resource_count != player3.previous_resource_count):
                                    
                                    if display_text.count('\n') > 9:
                                        display_text = ''
                                    display_text = display_text + str(current_player) + '_' + str(player3.current_detection_count) +'_' + str(resource_count) +'_' + str(resource_type) + '\n'
                                player3.previous_recource_type = resource_type
                                player3.previous_resource_count = resource_count

                            #cv2.imwrite('C:/catan_github/catan_helper/testing/'+'_' + str(current_player) + '_' + str(count_1) +'_' + str(resource_count) +'_' + str(resource_type) + '.png', res_pic)
                            player3.previous_detection_time = datetime.datetime.now()
                    
                    
                    cv2.imwrite('C:/catan_github/catan_helper/testing/'+'_' + str(current_player) + '_' + str(count_1) +'_' + str(resource_count) +'_' + str(resource_type) + '.png', res_pic)
                    '''  
                    try:
                        if resource_count[0] == '+':

                            if current_player == 1:
                                if resource_type == 'wood':
                                    player1_wood = player1_wood + int(resource_count[1:])
                                elif resource_type == 'sheep':
                                    player1_sheep = player1_sheep + int(resource_count[1:])
                                elif resource_type == 'wheat':
                                    player1_wheat = player1_wheat + int(resource_count[1:])
                                elif resource_type == 'stone':
                                    player1_stone = player1_stone + int(resource_count[1:])
                                else:
                                    player1_brick = player1_brick + int(resource_count[1:])
                            elif current_player == 2:
                                if resource_type == 'wood':
                                    player2_wood = player2_wood + int(resource_count[1:])
                                elif resource_type == 'sheep':
                                    player2_sheep = player2_sheep + int(resource_count[1:])
                                elif resource_type == 'wheat':
                                    player2_wheat = player2_wheat + int(resource_count[1:])
                                elif resource_type == 'stone':
                                    player2_stone = player2_stone + int(resource_count[1:])
                                else:
                                    player2_brick = player2_brick + int(resource_count[1:])
                            else:
                                if resource_type == 'wood':
                                    player3_wood = player3_wood + int(resource_count[1:])
                                elif resource_type == 'sheep':
                                    player3_sheep = player3_sheep + int(resource_count[1:])
                                elif resource_type == 'wheat':
                                    player3_wheat = player3_wheat + int(resource_count[1:])
                                elif resource_type == 'stone':
                                    player3_stone = player3_stone + int(resource_count[1:])
                                else:
                                    player3_brick = player3_brick + int(resource_count[1:])

                        else:
                            if current_player == 1:
                                if resource_type == 'wood':
                                    player1_wood = player1_wood - int(resource_count)
                                elif resource_type == 'sheep':
                                    player1_sheep = player1_sheep - int(resource_count)
                                elif resource_type == 'wheat':
                                    player1_wheat = player1_wheat - int(resource_count)
                                elif resource_type == 'stone':
                                    player1_stone = player1_stone - int(resource_count)
                                else:
                                    player1_brick = player1_brick - int(resource_count)
                            elif current_player == 2:
                                if resource_type == 'wood':
                                    player2_wood = player2_wood - int(resource_count)
                                elif resource_type == 'sheep':
                                    player2_sheep = player2_sheep - int(resource_count)
                                elif resource_type == 'wheat':
                                    player2_wheat = player2_wheat - int(resource_count)
                                elif resource_type == 'stone':
                                    player2_stone = player2_stone - int(resource_count)
                                else:
                                    player2_brick = player2_brick - int(resource_count)
                            else:
                                if resource_type == 'wood':
                                    player3_wood = player3_wood - int(resource_count)
                                elif resource_type == 'sheep':
                                    player3_sheep = player3_sheep - int(resource_count)
                                elif resource_type == 'wheat':
                                    player3_wheat = player3_wheat - int(resource_count)
                                elif resource_type == 'stone':
                                    player3_stone = player3_stone - int(resource_count)
                                else:
                                    player3_brick = player3_brick - int(resource_count)
                    except:
                        break

                    
                    display_text = 'player 1:'  + \
                                   '  player 2:' \
                                   '  player 3:' + \
                                   '\nWood:  ' + str(player1_wood) + \
                                    '  Wood:  '+ str(player2_wood) + \
                                    '  Wood:  '+ str(player3_wood) + \
                                    '\nSheep: ' + str(player1_sheep) + \
                                    '  Sheep: '+ str(player2_sheep) + \
                                    '  Sheep: '+ str(player3_sheep) + \
                                    '\nWheat:' + str(player1_wheat) + \
                                    '  Wheat:'+ str(player2_wheat) + \
                                    '  Wheat:'+ str(player3_wheat) + \
                                     '\nStone:  ' + str(player1_stone) + \
                                    '  Stone:  '+ str(player2_stone) + \
                                    '  Stone:  '+ str(player3_stone) + \
                                    '\nBrick:   ' + str(player1_brick) + \
                                    '  Brick:    '+ str(player2_brick) + \
                                    '  Brick:  '+ str(player3_brick)
                                    
                                  
                    '\n Sheep:' + str(player1_sheep) + \
                    '\n Wheat:' + str(player1_wheat) + \
                    '\n Stone:' + str(player1_stone) + \
                    '\n Brick:' + str(player1_brick) + \
                    
                    '\n Sheep:' + str(player2_sheep) + \
                    '\n Wheat:' + str(player2_wheat) + \
                    '\n Stone:' + str(player2_stone) + \
                    '\n Brick:' + str(player2_brick) + \
                    
                    '\n Sheep:' + str(player3_sheep) + \
                    '\n Wheat:' + str(player3_wheat) + \
                    '\n Stone:' + str(player3_stone) + \
                    '\n Brick:' + str(player3_brick)
                     
                    '''
                    self.dataChanged.emit(display_text)        
                    
                    
#test
class MainWindow(QtWidgets.QMainWindow):
             
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setAcceptDrops(True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.queenBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ui.queenBrowser.setText(str( 'Starting\nthe\ngame'))
        self.setStyleSheet("font: 16pt Comic Sans MS")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        #self.height = 300
        

    def handle_data_changed(self, text):
        self.ui.queenBrowser.setText(text)
        #self.label.adjustSize()

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    move_right_bottom_corner(window)
    window.resize(1000,500)
    window.show()

    worker = CatanHelperWorker()
    worker.dataChanged.connect(window.handle_data_changed)
    worker.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
   
