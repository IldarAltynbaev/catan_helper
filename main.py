import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget
from configparser import ConfigParser
from threading import Thread
import cv2 
import numpy as np 
import easyocr
import matplotlib.pyplot as plt
import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO

config_file = 'data/settings.ini'
config = ConfigParser()
config.read(config_file)
#backgroundImage = config.get("Theme", "background")


#from queenui import Ui_MainWindow

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


class MyWin(QtWidgets.QMainWindow):     
             
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setAcceptDrops(True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.queenBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ui.queenBrowser.setText(str('Starting\nthe\ngame'))
        self.setStyleSheet("font: 20pt Comic Sans MS")
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

    def set_text(self,text):
        self.ui.queenBrowser.setText(text)


def move_right_bottom_corner(win):
    screen_geometry = QApplication.desktop().availableGeometry()
    screen_size = (screen_geometry.width(), screen_geometry.height())
    win_size = (win.frameSize().width(), win.frameSize().height())
    x = screen_size[0] - win_size[0]
    y = screen_size[1] - win_size[1]
    win.move(x, y)

'''
def create_screen_prompt():
    
    app = QtWidgets.QApplication(sys.argv)
    w = MyWin()

    move_right_bottom_corner(w)
    w.set_text(str('current_id'))
    w.show()
    sys.exit(app.exec_())
'''
    
        
'''
def start_screen_tracking():
    
    resolution = (1920, 1080)
    codec = cv2.VideoWriter_fourcc(*"XVID")
    filename = "Recording.avi"
    fps = 60.0
    model = YOLO("best.pt")
       
    #img = pyautogui.screenshot()
    #frame = np.array(img)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for results in model.track(source='screen',conf=0.5, stream = True, show=False):
        for each_box in results.boxes:
            if (each_box.conf > 0.5) & (each_box.id is not None):
                current_id = each_box.id.max()
                


    cv2.destroyAllWindows() 
'''
class CatanHelperWorker(QObject):
    dataChanged = pyqtSignal(str)

    def start(self):
        Thread(target=self._execute, daemon=True).start()



    def _execute(self):
        
        resolution = (1920, 1080)
        codec = cv2.VideoWriter_fourcc(*"XVID")
        reader = easyocr.Reader(['en'], gpu=True)

        filename = "Recording.avi"
        fps = 60.0
        model = YOLO("best.pt")
        current_id = 0
        hsv_green1 = np.asarray([50, 0, 0])   
        hsv_green2 = np.asarray([59, 255, 255])   

        
    
        hsv_red1 = np.asarray([0, 117, 175])   
        hsv_red2 = np.asarray([3, 178, 255])   

        hsv_red3 = np.asarray([175, 117,175])   
        hsv_red4 = np.asarray([179, 178, 255])   

        count = 0

        #img = pyautogui.screenshot()
        #frame = np.array(img)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for results in model.track(source='screen',conf=0.5, stream = True, show=False):
            for each_box in results.boxes:
                if (each_box.conf > 0.5) & (each_box.id is not None):
                    #cv2.imshow('orig',results.orig_img)
                    

                    croppeed_img = results.orig_img[int(each_box.xyxy[0,1]):int(each_box.xyxy[0,3]),\
                                                    int(each_box.xyxy[0,0]):int(each_box.xyxy[0,2])]
                    
                    croppeed_img = cv2.cvtColor(croppeed_img, cv2.COLOR_BGR2HSV)
                    
                    mask_green = cv2.inRange(croppeed_img, hsv_green1, hsv_green2)

                    mask_red1 = cv2.inRange(croppeed_img, hsv_red1, hsv_red2)
                    mask_red2 = cv2.inRange(croppeed_img, hsv_red3, hsv_red4)

                    mask_red = mask_red1 + mask_red2

                    #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'cropped' + str(count) + '.png', croppeed_img)
                    #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'red' + str(count) + '.png', mask_red)
                    #cv2.imwrite('C:\catan_github\catan_helper\ ' + 'green' + str(count) + '.png', mask_green)
    
                    
                    text_ = reader.readtext(mask_green)


                    for t_, t in enumerate(text_):
                        count = count + 1
                        current_id = t[1]
                        self.dataChanged.emit(str(count) + ' ' + str(current_id))

                    kernel = kernel = np.ones((2,2),np.uint8)
                    mask = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
                    scale_factor = 2
                    mask = cv2.dilate(mask,kernel,iterations = 1)
                    upscaled = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                    blur = cv2.blur(upscaled, (5, 5))
                    text_ = reader.readtext(blur)

                    for t_, t in enumerate(text_):
                        count = count + 1
                        current_id = t[1]
                        self.dataChanged.emit(str(count) + ' ' + str(current_id))

                    #mask_red = cv2.inRange(croppeed_img, lower_red, upper_red)
                    #imask = mask_red>0
                    #red = np.zeros_like(croppeed_img, np.uint8)
                    #red[imask] = croppeed_img[imask]
                    #reader = easyocr.Reader(['en'], gpu=True)
                    #text_ = reader.readtext(red)
                    #for t_, t in enumerate(text_):
                    #    current_id = t[1]
                    #    self.dataChanged.emit(str(current_id))
                    
                    #current_id = each_box.id.max()
            
                    



class MainWindow(QtWidgets.QMainWindow):
             
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setAcceptDrops(True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.queenBrowser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.ui.queenBrowser.setText(str( 'Starting\nthe\ngame'))
        self.setStyleSheet("font: 20pt Comic Sans MS")
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)

    def handle_data_changed(self, text):
        self.ui.queenBrowser.setText(text)
        #self.label.adjustSize()


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    move_right_bottom_corner(window)
    window.show()

    worker = CatanHelperWorker()
    worker.dataChanged.connect(window.handle_data_changed)
    worker.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
    #img = cv2.imread('C:\catan_github\catan_helper\ 4.png')
    #grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('test',grey)
    #cv2.waitKey(0)


    #cv2.destroyAllWindows()

  
    
