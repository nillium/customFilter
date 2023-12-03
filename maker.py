from PyQt6.QtWidgets import QWidget, QPushButton, QFrame, QGridLayout, QLabel, QFileDialog, QDial, QGraphicsPixmapItem, QHBoxLayout, QGraphicsTextItem, QGraphicsView, QVBoxLayout, QGraphicsRectItem, QApplication, QSlider, QGraphicsScene, QGraphicsScene, QGraphicsView, QMainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

import numpy as np
import cv2

class maker_main(QLabel):
    def __init__(self):
        super().__init__()

    class make_label(QLabel):
        def __init__(self, x, y, w, h, text="None", alignment="center", parent=None):
            super().__init__()      
            self.setText(text)
            self.setStyleSheet("background-color: #eeeeee; border: none; padding: 0px;")
            self.setFixedHeight(h)
            self.setFixedWidth(w)
            width = self.width()
            height = self.height()
            print("x:" + str(x))
            print("y:" + str(y))
            print("width:" + str(w))
            print("height:" + str(h))
            print("width/2:" + str(width/2))
            print("height/2:" + str(height/2))
            #print(round((x-(width/2))))
            #print(round((height/2)))
            self.setGeometry(x, y, w, h)
            self.setParent(parent)
            if alignment == "center":
                self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if alignment == "left":
                self.setAlignment(Qt.AlignmentFlag.AlignLeft)
            if alignment == "right":
                self.setAlignment(Qt.AlignmentFlag.AlignRight)


    class make_image_container(Self,QLabel):
        def __init__(self, image=None, x=0, y=0, w=128, h=32, parent=None):
            super().__init__()
            self.setGeometry(x, y, w, h)
            self.setStyleSheet("background-color: dddddd; border: 1px solid #dddddd; padding: 0px;")
            image = cv2.resize(image, (w,h), interpolation = cv2.INTER_NEAREST)
            image = QImage(image, w, h, (1*w), QImage.Format.Format_Grayscale8)
            image = QPixmap.fromImage(image)
            self.setPixmap(image)
            self.setParent(parent)

    class makeDial(QDial):
        def __init__(self, call_on_change, x, y, w, h, min, max, val, parent=None):
            super().__init__()
            self.setNotchesVisible(True)
            self.setMaximum(max)
            self.setMinimum(min)
            self.setValue(val)
            self.setGeometry(x,y,w,h)
            self.setParent(parent)
            self.valueChanged.connect(call_on_change)


    class makeButton(QPushButton):
        def __init__(self, label, call_on_press, x=0,y=0,w=128,h=32, parent=None):
            super().__init__()
            self.label = label
            self.call_on_press = call_on_press
            self.setText(label)
            self.setGeometry(x,y,w,h)
            self.clicked.connect(self.call_on_press)
            self.setParent(parent)


    class MainWindow(QMainWindow):
        def __init__(self, center=True, w, h):
            super().__init__()
            self.main_widget = QWidget()
            self.setFixedSize(w,h)
            self.setCentralWidget(self.main_widget)
            if center:
                self.center()

        def center(self):

            root_window = self.frameGeometry()
            self.cp = self.screen().availableGeometry().center()
            root_window.moveCenter(self.cp)
            self.move(root_window.topLeft()) 