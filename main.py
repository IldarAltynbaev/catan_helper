import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget
from configparser import ConfigParser

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
        MainWindow.setMaximumSize(QtCore.QSize(424, 424))

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')

        #self.textBrowser = QtWidgets.QLineEdit("QLineEdit", self.centralwidget)
        #self.textBrowser.setGeometry(QtCore.QRect(170, 383, 251, 41))
        #self.textBrowser.setReadOnly(False)
        #self.textBrowser.setObjectName('textBrowser')

        #self.queenBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        #self.queenBrowser.setGeometry(QtCore.QRect(200, 190, 222, 71))  
        #self.queenBrowser.setReadOnly(True)
        #self.queenBrowser.setObjectName('queenBrowser')    
             
        #self.background = QtWidgets.QLabel(self.centralwidget)
        #self.background.resize(424, 433)
     
        self.queenBrowser = QtWidgets.QTextBrowser(self.background)
        self.queenBrowser.setGeometry(QtCore.QRect(200, 190, 20, 500))  
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
        self.ui.queenBrowser.setText('Starting\nthe\ngame')
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


if __name__ == '__main__':

    resolution = (1920, 1080)
    codec = cv2.VideoWriter_fourcc(*"XVID")
    filename = "Recording.avi"
    fps = 60.0
    #out = cv2.VideoWriter(filename, codec, fps, resolution)
    #cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Live", 480, 270)

    model = YOLO("best.pt")

    app = QtWidgets.QApplication(sys.argv)
    w = MyWin()
    
    #move_right_bottom_corner(w)
    w.show()
    '''
    while True:
       
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #out.write(frame)
        
        # Optional: Display the recording screen
        #cv2.imshow('Live', frame)
        results = model.track(frame)
        for each_box in results[0].boxes:
            if each_box.conf > 0.5:
                ccurrent_max_id = each_box.id.max()
                w.set_text(str(ccurrent_max_id))

                

        
        # Stop recording when we press 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release the Video writer
    #out.release()
    

    # Destroy all windows
    cv2.destroyAllWindows()
   
    '''
    sys.exit(app.exec_())
