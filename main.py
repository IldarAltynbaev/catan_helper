import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget
from configparser import ConfigParser
from threading import Thread



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

def create_screen_prompt():
    
    app = QtWidgets.QApplication(sys.argv)
    w = MyWin()

    move_right_bottom_corner(w)
    w.set_text(str('current_id'))
    w.show()
    sys.exit(app.exec_())
    
    
        

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

class CatanHelperWorker(QObject):
    dataChanged = pyqtSignal(str)

    def start(self):
        Thread(target=self._execute, daemon=True).start()

    def _execute(self):
        
        resolution = (1920, 1080)
        codec = cv2.VideoWriter_fourcc(*"XVID")
        filename = "Recording.avi"
        fps = 60.0
        model = YOLO("best.pt")
        current_id = 0

        #img = pyautogui.screenshot()
        #frame = np.array(img)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for results in model.track(source='screen',conf=0.5, stream = True, show=False):
            for each_box in results.boxes:
                if (each_box.conf > 0.5) & (each_box.id is not None):
                    current_id = each_box.id.max()
            
                    #bid_usd = mt5.symbol_info_tick("EURUSD").bid
                    self.dataChanged.emit(str(current_id))
                


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
  
    
