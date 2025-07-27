import sys
from mss import mss
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget
from configparser import ConfigParser
from threading import Thread
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms, models
from screeninfo import get_monitors
import pandas as pd
import datetime
import cv2
from PIL import Image
from itertools import zip_longest
from PyQt5.QtGui import QFont





class Catan_player():
        
        def __init__(self, player_number):
            self.previous_detection_time = datetime.datetime.now()
            self.previous_resource_count = ''
            self.previous_recource_type = ''
            self.previous_sign_result = ''
            self.stone_count = 0
            self.wheat_count = 0
            self.sheep_count = 0
            self.brick_count = 0
            self.wood_count = 0
            self.unknown_count_plus = 0
            self.unknown_count_minus = 0
            self.current_detection_count = 0
            self.player_number = player_number

        def get_list_of_resources(self):

            output_list = []
            output_list.append(self.stone_count)
            output_list.append(self.wheat_count)
            output_list.append(self.sheep_count)
            output_list.append(self.brick_count)
            output_list.append(self.wood_count)
            output_list.append(self.unknown_count_plus)
            output_list.append(self.unknown_count_minus)

            return output_list
            

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

        model.fc = torch.nn.Linear(model.fc.in_features, 6)    
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

        #model_yolo = YOLO("C:/catan_universe_project/catan_helper/best_model_yolo11.pt")
        model_yolo = YOLO("C:\catan_universe_project\catan_helper\yolo11m_detection.pt")
        model_yolo_segmentagion = YOLO("C:/catan_universe_project/catan_helper/yolo11_segmentation_all_v3.pt")
        model_resnet = self.create_blank_model(freeze_layers = True)
        model_resnet.load_state_dict(torch.load('C:/catan_universe_project/catan_helper/best_model_resnet_7.pt',
                                               map_location=torch.device('cpu')))
        #mode_resnet.to(device)
        #model_yolo_digit_detection = YOLO("C:\catan_universe_project\catan_helper\yolo11_digit_detection.pt")
        model_resnet.eval()

        #resnet_mnist = timm.create_model("resnet18", pretrained=False, num_classes=10)
        #resnet_mnist.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #resnet_mnist.load_state_dict(
        #torch.hub.load_state_dict_from_url(
        #    "https://huggingface.co/gpcarl123/resnet18_mnist/resolve/main/resnet18_mnist.pth",
        #    map_location="cpu",
        #    file_name="resnet18_mnist.pth",
        #)
        #)
        #resnet_mnist.eval()

        return model_yolo, model_resnet, model_yolo_segmentagion #, model_yolo_digit_detection, resnet_mnist

    

    def get_resource_name(self, resource_result):

        if resource_result == 0:
            resource_name = 'brick'
        elif resource_result == 1:
            resource_name = 'sheep'
        elif resource_result == 2:
            resource_name = 'stone'
        elif resource_result == 3:
            resource_name = 'unknown'
        elif resource_result == 4:
            resource_name = 'wheat'
        elif resource_result == 5:
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
                                        allowlist,
                                        max_right_point,
                                        sign_result,
                                        model_yolo_digit_detection,
                                        count,
                                        resnet_mnist,
                                        transforms_mnist,
                                        preprocessor_mnist):
        overall_text = ''
        current_text = ''
        box_img = cv2.cvtColor(box_img, cv2.COLOR_BGR2HSV)    
        mask_green = cv2.inRange(box_img, hsv_green1, hsv_green2)
        mask_green[:,0:max_right_point+2] = 0
        mask_green = np.repeat(np.expand_dims(mask_green,2), 3,axis=2)

        mask_red1 = cv2.inRange(box_img, hsv_red1, hsv_red2)
        mask_red2 = cv2.inRange(box_img, hsv_red3, hsv_red4)
        mask_red = mask_red1 + mask_red2
        mask_red[:,0:max_right_point+2] = 0
        mask_red = np.repeat(np.expand_dims(mask_red,2), 3,axis=2)

        detection_img = None

        if sign_result == '+':
             
            results = model_yolo_digit_detection(mask_green)
            #cv2.imwrite('C:/catan_testing/test_green_'+str(count)+'.png', mask_green)

        elif sign_result == '-':
           
            results = model_yolo_digit_detection(mask_red)
            #cv2.imwrite('C:/catan_testing/test_red'+str(count)+'.png', mask_red)
            #cv2.imwrite('C:/catan_testing/test'+str(count)+'.png', mask_red)                

        else:
            return current_text, mask_green, mask_red, detection_img
        
        for each_result in results:
            for each_box in each_result.boxes:
                if (each_box.conf > 0.9):
                    detection_img = each_result.orig_img[int(each_box.xyxy[0,1]):int(each_box.xyxy[0,3]),\
                                                        int(each_box.xyxy[0,0]):int(each_box.xyxy[0,2])]
                    
                    detection_img = np.expand_dims(detection_img[:,:,0], axis=2)
                    resized_image = cv2.resize(detection_img, (28, 28))
                 
                    
                    current_text = self.perform_mnist_prediction(resnet_mnist,preprocessor_mnist,transforms_mnist,resized_image) 

                    if current_text == '7':

                        kernel = np.ones((3,3),np.uint8)
                        kernel_image = cv2.erode(resized_image,kernel,iterations = 1)
                        current_text = self.perform_mnist_prediction(resnet_mnist,preprocessor_mnist,transforms_mnist,kernel_image) 
                        
                        if current_text == '7':

                            kernel = np.ones((3,3),np.uint8)
                            kernel_image = cv2.erode(resized_image,kernel,iterations = 2)
                            current_text = self.perform_mnist_prediction(resnet_mnist,preprocessor_mnist,transforms_mnist,kernel_image) 

                    
                    overall_text = overall_text + current_text
                    #cv2.imwrite('C:/catan_testing/test_detection_'+str(current_text)+'.png', box_img)
                    
        
        return overall_text, mask_green, mask_red, detection_img
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'cropped' + str(count) + '.png', croppeed_img_quantity)
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'red' + str(count) + '.png', mask_red)
        #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'green' + str(count) + '.png', mask_green)
        
        return current_text, mask_green, mask_red

    def perform_mnist_prediction(self, resnet_mnist ,preprocessor_mnist,transforms_mnist,kernel_image):

        trans_img = transforms_mnist(kernel_image).unsqueeze(dim=0)
        mnist_result = resnet_mnist(preprocessor_mnist(trans_img))

        return str(mnist_result.argmax(dim=1).item())         

    def get_resource_type_from_img(self, pred_transforms, model_resnet, box_img,device):

        _, width, _ = box_img.shape
        width_cutoff = int(width / 2)
        cropped_box_img = box_img[:, width_cutoff:]
        cropped_box_img = cv2.cvtColor(cropped_box_img, cv2.COLOR_BGR2RGB)
        cropped_box_img = Image.fromarray(cropped_box_img)
        transformed_cropped_img = pred_transforms(cropped_box_img)
        transformed_cropped_img = transformed_cropped_img.to(device)
        pred = model_resnet(transformed_cropped_img.unsqueeze(0))
        resource_result = pred.argmax().item()

        confidence = torch.nn.functional.softmax(pred[0], dim=0).max().item()
        resource_name = self.get_resource_name(resource_result)
        
        return resource_name, confidence

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

        
        hsv_green1 = np.asarray([52, 92, 154])   
        hsv_green2 = np.asarray([179, 161, 255])   

        #hsv_green1 = np.asarray([50, 0, 0])   
        #hsv_green2 = np.asarray([59, 255, 255])   

        hsv_red1 = np.asarray([0, 117, 175])   
        hsv_red2 = np.asarray([3, 178, 255])   

        hsv_red3 = np.asarray([175, 117,175])   
        hsv_red4 = np.asarray([179, 178, 255]) 

        return hsv_green1, hsv_green2, hsv_red1, hsv_red2, hsv_red3, hsv_red4

    def resource_count_is_correct(self, sign_count_result, resource_confidence):
    
        if len(sign_count_result) >= 2:
            if (sign_count_result[0] == '+') | (sign_count_result[0] == '-'):            
                if sign_count_result[1:].isdigit():
                    if resource_confidence >= 0.9:
                        return True
            
        return False
    
    def create_mnist_transforms(self):

        return transforms.Compose([transforms.ToTensor()])
    
    def create_mnist_preprocessor(self):

        return torchvision.transforms.Normalize((0.1307,), (0.3081,))
    
    def check_and_add_resource_to_player(self, delay_delta, player, sign_result, resource_count, resource_type):

        display_text = ''
        
        if datetime.datetime.now() - player.previous_detection_time > delay_delta:

            player.current_detection_count = player.current_detection_count + 1                                                                                      
            #if display_text.count('\n') > 9:
            #    display_text = ''

            #display_text = display_text + str(player.player_number) + '_' + sign_result +'_' + str(resource_count) +'_' + str(resource_type) + '\n'
            self.add_resource_to_player(player, resource_type, resource_count, sign_result)
            player.previous_recource_type = resource_type
            player.previous_resource_count = resource_count
            player.previous_sign_result = sign_result

        else:
        
            if (resource_type != player.previous_recource_type) | (resource_count != player.previous_resource_count) \
                | (sign_result != player.previous_sign_result):
                self.add_resource_to_player(player, resource_type, resource_count, sign_result)
                #display_text = display_text + str(player.player_number) + '_' + sign_result + '_' + str(resource_count) +'_' + str(resource_type) + '\n'
            player.previous_recource_type = resource_type
            player.previous_resource_count = resource_count
            player.previous_sign_result = sign_result

            #cv2.imwrite('C:/catan_github/catan_helper/testing/'+'_' + str(current_player) + '_' + str(count_1) +'_' + str(resource_count) +'_' + str(resource_type) + '.png', res_pic)
        player.previous_detection_time = datetime.datetime.now()

        #return display_text
    
    def get_result_string(self, each_box,result_tensor,output_string):

        xy_list = each_box.masks.xy
        first_elements = np.array([x.min(axis=0)[0] for x in xy_list])

        sort_key = torch.from_numpy(first_elements)
        sorted_indices = torch.argsort(sort_key)
        sorted_tensor = result_tensor[sorted_indices]

        for each_element in sorted_tensor:
            if each_element == 0:
                output_string = output_string + "-"
            elif each_element == 11:
                output_string = output_string + "+"
            elif each_element == 1:
                output_string = output_string + "1"
            elif each_element == 2:
                output_string = output_string + "2"
            elif each_element == 3:
                output_string = output_string + "3"
            elif each_element == 4:
                output_string = output_string + "4"
            elif each_element == 5:
                output_string = output_string + "5"
            elif each_element == 6:
                output_string = output_string + "6"
            elif each_element == 7:
                output_string = output_string + "7"
            elif each_element == 8:
                output_string = output_string + "8"
            elif each_element == 9:
                output_string = output_string + "9"
            elif each_element == 10:
                output_string = output_string + "0"

        return output_string

    def add_resource_to_player(self,player, resource_type, resource_count, sign_result):
        
        formula = "player_count" + sign_result + "current_count"
        current_count = int(resource_count)

        if resource_type == 'wood':
            player_count = player.wood_count
            if (eval(formula) < 0) & (player.unknown_count_plus>0):
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)
            else:
                player.wood_count = eval(formula)
        elif resource_type == 'sheep':
            player_count = player.sheep_count
            if (eval(formula) < 0) & (player.unknown_count_plus>0):
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)
            else:
                player.sheep_count = eval(formula)
        elif resource_type == 'wheat':
            player_count = player.wheat_count
            if (eval(formula) < 0) & (player.unknown_count_plus>0):
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)
            else:
                player.wheat_count = eval(formula)
        elif resource_type == 'stone':
            player_count = player.stone_count
            if (eval(formula) < 0) & (player.unknown_count_plus>0):
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)
            else:
                player.stone_count = eval(formula)
        elif resource_type == 'brick':
            player_count = player.brick_count
            if (eval(formula) < 0) & (player.unknown_count_plus>0):
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)
            else:
                player.brick_count = eval(formula)
        elif resource_type == 'unknown':     
            if sign_result == '-':
                player_count = player.unknown_count_minus  
                player.unknown_count_minus = eval(formula)          
            elif sign_result == '+':      
                player_count = player.unknown_count_plus     
                player.unknown_count_plus = eval(formula)

    


    def get_sign_and_resource_count(self,model_yolo_segmentation, box_img):

        
        segmentation_result = model_yolo_segmentation(box_img,conf = 0.7, iou=0.45)
        output_string = ''

        need_invert_color = False

        for each_box in segmentation_result:
            
            if each_box.masks is None:
                return ''
            
            

            result_tensor = each_box.boxes.cls
            #confidence_tensor = each_box.boxes.conf
             
            exists = (result_tensor == 11).any()
             
            if not exists:
                need_invert_color = True
                break
            
            output_string = self.get_result_string(each_box,result_tensor,output_string)      

        if need_invert_color:

            blue, green, red = cv2.split(box_img)
            inverted_img = cv2.merge([blue, red, green])
            segmentation_result = model_yolo_segmentation(inverted_img, conf = 0.7, iou=0.45)
            for each_box in segmentation_result:

                result_tensor = each_box.boxes.cls
                if each_box.masks is None:
                    return ''
             
                exists = (result_tensor == 0).any()
                if not exists:
                    break

                output_string = self.get_result_string(each_box,result_tensor,output_string)

        return output_string      
    
    def format_columns(self, data, col_widths=None):
       
        if col_widths is None:
            col_widths = [
                max(len(str(row[i])) if i < len(row) else 0 for row in data) + 2
                for i in range(max(len(row) for row in data))
            ]
        
        html = ["<pre>"]
        for row in data:
            line = []
            for i in range(len(col_widths)):               
                cell = str(row[i]) if i < len(row) else ""              
                padded = cell.ljust(col_widths[i])[:col_widths[i]]
                spaced = padded.replace(" ", "&nbsp;")           
                spaced = f"<b>{spaced}</b>"
                line.append(spaced)
            html.append("".join(line))
        html.append("</pre>")
        
        return "\n".join(html)
        
    def _execute(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        monitor = get_monitors()[0]  # Index 0 is all monitors, 1 is primary
        screen_width = monitor.width
        screen_height = monitor.height
        
        one_fifth_hight = screen_height // 5
        moniro_top = {"top": 0, "left": 0, "width": screen_width, "height": one_fifth_hight}
        selected_region = moniro_top

        model_yolo, model_resnet, model_yolo_segmentation = self.initialize_models()

        model_yolo_segmentation.to(device)
        model_yolo_segmentation.eval()
      
        model_yolo.to(device)
        model_yolo.eval()
        model_resnet.to(device)
        model_resnet.eval()

    
        pred_transforms = self.create_transforms()

        delay_delta = datetime.timedelta(seconds=0.5)     
        
        display_text = ''

        player1 = Catan_player(player_number=1)
        player2 = Catan_player(player_number=2)
        player3 = Catan_player(player_number=3)
        count = 0
        
        list_of_resourses = ['stone', 'wheat' ,'sheep', 'brick', 'wood', 'unknown+', 'unknown-']
        
        with mss() as sct:
            while True:
                screen = np.array(sct.grab(selected_region))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                results_track = model_yolo.track(frame, persist=True)
                results = results_track[0]
        
                for each_box in results.boxes:
                    if each_box.cls[0].item() == 1.0:
                        break
                    if (each_box.conf > 0.5) & (each_box.id is not None):

                        
                        box_img = results.orig_img[int(each_box.xyxy[0,1]):int(each_box.xyxy[0,3]),\
                                                        int(each_box.xyxy[0,0]):int(each_box.xyxy[0,2])]
                       
                        #cv2.imwrite('C:/catan_universe_project/testing/' + str(count) + '_' + '.png', box_img)
                        count = count + 1
                        current_player = self.get_current_player_by_region(each_box)   
                        sign_count_result = self.get_sign_and_resource_count(model_yolo_segmentation,box_img)                  
                        
                        resource_type, resource_confidence = self.get_resource_type_from_img(pred_transforms,
                                                                        model_resnet,
                                                                        box_img,device) 
                                      
                        
                        if self.resource_count_is_correct(sign_count_result, resource_confidence):
                            sign_result = sign_count_result[0]
                            resource_count = sign_count_result[1:]

                            if current_player == 1:

                                self.check_and_add_resource_to_player(delay_delta, player1, sign_result, resource_count, resource_type)
                                                               
                            elif current_player == 2:
                            
                                self.check_and_add_resource_to_player(delay_delta, player2, sign_result, resource_count, resource_type)

                            elif current_player == 3:
                                
                                self.check_and_add_resource_to_player(delay_delta, player3, sign_result, resource_count, resource_type)

                            #if len(sign_count_result) > 0:
                            #    cv2.imwrite('C:/catan_universe_project/testing/' + str(count) + '_' + str(sign_count_result[0]) + '_' + \
                            #        str(sign_count_result[1:]) + '_' + resource_type + '_' + 'conf_' + str(resource_confidence) +'.png', box_img)


                        
                        pl1_list = player1.get_list_of_resources()
                        pl2_list = player2.get_list_of_resources()
                        pl3_list = player3.get_list_of_resources()

                        #df = pd.DataFrame({'Res':list_of_resourses, 'Player1': player1.get_list_of_resources(), 'Player2': player2.get_list_of_resources(), \
                        #                    'Player3': player3.get_list_of_resources()})
                       
                        #aligned_str = self.format_dataframe(df)
                        #print(self.format_dataframe(df))
                        

                     
                        #df = pd.DataFrame({'':list_of_resourses, 'Player1': player1.get_list_of_resources(), 'Player2': player2.get_list_of_resources(), \
                        #                    'Player3': player3.get_list_of_resources()})
                        '''
                        aligned_str = df.to_string(index=False, formatters={
                                            'Res type': '{:<15}'.format,      # Left align
                                            'Player1': '{:^5}'.format,       # Center align
                                            'Player2': '{:^5}'.format,
                                            'Player3': '{:^5}'.format},)
                        '''

                        col_widths = [9, 8, 8, 8]  # Adjust as needed
                        aligned_str = self.format_columns([['Res', 'Player1', 'Player2', 'Player3' ], 
                                                           [list_of_resourses[0], pl1_list[0], pl2_list[0], pl3_list[0]],
                                                           [list_of_resourses[1], pl1_list[1], pl2_list[1], pl3_list[1]],
                                                           [list_of_resourses[2], pl1_list[2], pl2_list[2], pl3_list[2]],
                                                           [list_of_resourses[3], pl1_list[3], pl2_list[3], pl3_list[3]],
                                                           [list_of_resourses[4], pl1_list[4], pl2_list[4], pl3_list[4]],
                                                           [list_of_resourses[5], pl1_list[5], pl2_list[5], pl3_list[5]],
                                                           [list_of_resourses[6], pl1_list[6], pl2_list[6], pl3_list[6]]
                                                           ], col_widths)
                               
                        self.dataChanged.emit(aligned_str)    
                        
                    

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

        #mono_font = QFont("Courier New", 10)
        #self.ui.queenBrowser.setFont(mono_font)
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
   
