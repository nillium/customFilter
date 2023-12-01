import sys, os
from PyQt6.QtWidgets import QWidget, QPushButton, QFrame, QGridLayout, QLabel, QFileDialog, QDial, QHBoxLayout, QVBoxLayout, QApplication, QSlider, QGraphicsScene, QGraphicsScene, QGraphicsView, QMainWindow
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QColor, QPen, QImage
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import keras.backend as K
import numpy as np
from keras import Input, layers, initializers
from keras.models import Model
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PyQt6.QtCore import Qt
import sys
from PyQt6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt, QRectF
import sys

class makeLabel(QLabel):
    def __init__(self, label_text=None, alignment="center", image_source_type="file", image_source=None, x=0, y=0, w=128, h=32):
        
        super().__init__()

        self.setGeometry(x, y, w, h)
        self.setStyleSheet("background-color: lightgray; border: 1px solid #cccccc; padding: 0px;")
        
        if not label_text == None:
            self.setText(label_text)

        if not ((alignment == "center") or (alignment == "right") or (alignment == "left")):
            print("Wrong label alignment option: Choose either left or right or center.")
        
        if not ((image_source_type == "file") or (image_source_type == "array")):
            print("Wrong image source type option: Choose either file or array.")

        if alignment == "center":
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if alignment == "left":
            self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        if alignment == "right":
            self.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.setGeometry(x,y,w,h)

        if not image_source==None:
            
            #image_source: path if file, array name if array.
            self.pixmap = QPixmap(image_source)
            
            if image_source_type == "file":
                
                if not self.pixmap.isNull():
                    self.setPixmap(self.pixmap)
                    self.setParent(main_window.main_widget)
                else:
                    print(f"Error loading image from {image_source}")
            
            if image_source_type == "array":

                if not self.pixmap.isNull():
                    self.setPixmap(self.pixmap)
                    self.setParent(main_window.main_widget)
                else:
                    print(f"Error loading image from array. Is array defined?")
        
        self.setParent(main_window.main_widget)


""" class makeCanvas(QGraphicsView):
    def __init__(self, x=0,y=0,w=512,h=512,r=0,g=0,b=0,a=0):
        
        super().__init__()

        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setBackgroundBrush(QColor(r, g, b, a))
        self.setGeometry(x, y, w, h)
        self.scene.setBackgroundBrush(QColor(r, g, b, a))  # Set the background color to light gray
        self.setParent(main_window.main_widget) """


class makeSlider(QSlider):
    def __init__(self, label, call_on_change, x=0,y=0,w=512,h=32,min=0,max=100,val=50):
        super().__init__()
        self.sliderlabel = QLabel()

        self.setOrientation(Qt.Orientation.Horizontal)
        self.setGeometry(x,y,w,h)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.setTickInterval(1)
        

        self.sliderlabel.setText(label + " [px] : %d" % self.value())
        self.call_on_change = call_on_change
        
        self.valueChanged.connect(call_on_change)
        
        self.sliderlabel.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sliderlabel.setGeometry(x,y-h+5,w,h)

        self.setParent(main_window.main_widget)
        self.sliderlabel.setParent(main_window.main_widget)

class makeDial(QDial):
    def __init__(self, call_on_change, x, y, w, h, min, max, val):
        super().__init__()
        
        self.setNotchesVisible(True)
        self.setMaximum(max)
        self.setMinimum(min)
        self.setValue(val)
        self.setGeometry(x,y,w,h)
        self.setParent(main_window.main_widget)
        self.valueChanged.connect(call_on_change)




class makeButton(QPushButton):
    def __init__(self, label, call_on_press, x=0,y=0,w=128,h=32):
        super().__init__()
        self.label = label
        self.call_on_press = call_on_press
        self.setText(label)
        self.setGeometry(x,y,w,h)
        self.clicked.connect(self.call_on_press)
        self.setParent(main_window.main_widget)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Create the main widget
        self.main_widget = QWidget()
        self.setFixedSize(1560,800)
        self.setCentralWidget(self.main_widget)

class loadImage(QGraphicsPixmapItem):
    def __init__(self, path, scene):
        super().__init__()
        self.image_path = path
        self.parent_scene = scene
        self.pixmap = QPixmap(self.image_path)
        

        if not self.pixmap.isNull():
            self.pixmap_item = self.pixmap
            self.parent_scene.addItem(self)
        else:
            print(f"Error loading image from {self.image_path}")


class Contents:
    def __init__(self):
        super().__init__()
    def make(self):
        

        #self.items[label_left] = makeLabel(image_source_type="file", image_source="checker.png", x=10, y=10, w=512, h=512)
        #self.leftCanvas = makeCanvas(x=10, y=10, r=200, a=0)
        #self.centerCanvas = makeCanvas(x=532, y=10,w=256,h=256)
        #self.rightCanvas = makeCanvas(x=798, y=10,w=512,h=512)^
        self.label_left = makeLabel(image_source_type="file", image_source="checker.png", x=10, y=10, w=512, h=512)
        self.label_center_top = makeLabel(alignment="left", image_source_type="file", image_source="checker.png", x=532, y=10, w=256, h=256)
        self.label_center_bot = makeLabel(alignment="left", image_source_type="file", image_source="checker.png", x=532, y=316+64+10, w=256, h=256)
        self.label_right = makeLabel(image_source_type="file", image_source="checker.png", x=798, y=10, w=512, h=512)
        
        self.change_input_BTN = makeButton("Change Input", self.change_input_press, x=10, y=532)

        
        #self.filter_size_SLDR = makeSlider("Filter Size", self.filter_size_SLDR_value_change, x=532, y=296, w=256, min=2, max=64, val=8)
        self.custom_filter_BTN = makeButton("Load Custom Filter", self.custom_filter_press, x=532, y=276, w=256, h=28)
        
        self.random_filter_BTN = makeButton("Random", self.random_filter_press, x=728, y=316, w=60, h=28)
        self.invert_filter_BTN = makeButton("Invert", self.invert_filter_press, x=728, y=316+64-28, w=60, h=28)
        #self.convolute_BTN = makeButton("--> Convolute -->", self.convolute_press, x=532, y=276+64+42+42, w=256)

        self.filter_size_DIAL = makeDial(self.filter_size_DIAL_value_change, 532, 316, 64, 64, 2, 64, 32)
        self.white_treshold_DIAL = makeDial(self.white_treshold_DIAL_value_change, 532+64, 316, 64, 64, 0, 126, 64)
        self.black_treshold_DIAL = makeDial(self.black_treshold_DIAL_value_change, 532+128, 316, 64, 64, 0, 126, 255)
        
        #self.slider.sliderlabel.setParent(main_window.main_widget)
        #self.slider.valueChanged.connect()
        #self.slider.setParent(main_window.main_widget)
        #self.slider.sliderlabel.setParent(main_window.main_widget)
        
        #self.input_image = loadImage("checker.png", self.leftCanvas.scene)
       
       
        self.items={
            "label_left" : self.label_left,
            "label_center_top" : self.label_center_top,
            "label_center_bot" : self.label_center_bot
            #"filter_size_SLDR" : self.filter_size_SLDR
        }
    

    
    
    def change_image(self, label_name, random=True, invert=False, black_treshold_change = False, filter_update=False, filter_size=None):
        
        self.display_width = self.items[label_name].width()
        self.display_heigth = self.items[label_name].height()

        if not random and not invert and not black_treshold_change:
            file_type_filter = 'Image File (*.png *.jpg)'
            response = QFileDialog.getOpenFileName(
                parent=main_window.main_widget,
                caption='Select a file',
                directory=os.getcwd(),
                filter=file_type_filter,
                initialFilter='Image File (*.png *.jpg)')
            file_dialog = QFileDialog()
            file_dialog.setWindowTitle('Open File')
            #HANDLE EMPTY selection!
            print(f'Selected File: {response[0]}')
            image_path = response[0]
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(self.image, (self.display_width,self.display_heigth))
        if random and not invert and not black_treshold_change:
            self.image = np.random.randint(0, 256, size=(filter_size, filter_size), dtype=np.uint8)
            self.image = cv2.resize(self.image, (self.display_width,self.display_heigth), interpolation = cv2.INTER_NEAREST)
            ret1,self.image = cv2.threshold(self.image,0,self.black_treshold_DIAL.value(),cv2.THRESH_BINARY)
        
        if not random and invert and not black_treshold_change:
            self.image = cv2.bitwise_not(self.image)
            ret1,self.image = cv2.threshold(self.image,0,self.black_treshold_DIAL.value(),cv2.THRESH_BINARY)

    
        if black_treshold_change and not random and not invert:
            pass


        
        self.display_image = QImage(self.image, self.display_width, self.display_heigth, (1*self.display_width), QImage.Format.Format_Grayscale8)
            
        self.display_pixmap = QPixmap.fromImage(self.display_image)
        self.items[label_name].setPixmap(self.display_pixmap)
        
    def change_input_press(self):
        print("change_input_press")
        self.change_image("label_left")
    
    def custom_filter_press(self):
        print("custom_filter_press")
        self.change_image("label_center_top", False)

    def random_filter_press(self):
        print("random_filter_press")
        
        self.change_image("label_center_bot", random=True, filter_update=True, filter_size=self.filter_size_DIAL.value())
    
    def invert_filter_press(self):
        print("invert_filter_press")
        self.change_image("label_center_bot", invert=True, random=False, filter_update=True, filter_size=self.filter_size_DIAL.value())

    def convolute_press(self):
        print("convolute_press")

    def filter_size_SLDR_value_change(self):
        self.items["filter_size_SLDR"].sliderlabel.setText("Filter Size [px] : %d" % self.filter_size_SLDR.value())

    def filter_size_DIAL_value_change(self):
        print("filter_size_DIAL_value_change")
    
    def white_treshold_DIAL_value_change(self):
        print("white_treshold_DIAL_value_change")
    
    def black_treshold_DIAL_value_change(self):
        print("black_treshold_DIAL_value_change")
        
        self.change_image("label_center_bot", invert=False, random=False, black_treshold_change=True, filter_update=True, filter_size=self.filter_size_DIAL.value())
        #ret2,minus = cv2.threshold(self.customfilter,self.lowerTreshold,127,cv2.THRESH_BINARY)




if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create the main window
    main_window = MainWindow()
    
    contents = Contents()
    contents.make()
    main_window.show()
    sys.exit(app.exec())








"""
class tools:
    def __init__(self):
        pass
             
    def createScene(self, bckcolor=(200, 200, 200),x=0,y=0,w=50,h=50):
        scene = QGraphicsScene()
        scene.setSceneRect(QRectF(x, y, w, h))
        scene.setBackgroundBrush(QColor(bckcolor))
        return scene


class Canvas(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.leftScene = tools.createScene((100, 200, 200), 10, 10, 592, 592)
        self.centerScene = tools.createScene((100, 200, 200), 612, 10, 336, 336)
        self.rightScene = tools.createScene((100, 200, 200), 958, 10, 592, 592)
        self.leftScene.setParent(self)
        self.centerScene.setParent(self)
        self.rightScene.setParent(self)

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainCanvas = Canvas()
    mainCanvas.show()
    sys.exit(app.exec())



class mainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        mainHBox = QHBoxLayout(self)
        self.leftVBOX = QVBoxLayout()
        leftHBOX = QHBoxLayout()
        self.centerVBOX = QVBoxLayout()
        self.rightVBOX = QVBoxLayout()

        self.leftScene = scene.createScene((100, 200, 200), 10, 10, 592, 592)
        

        

        self.centerScene = QGraphicsScene()
        self.centerScene.setSceneRect(QRectF(612, 0, 336, 336))
        self.centerScene.setBackgroundBrush(QColor(200, 200, 200))

        self.rightScene = QGraphicsScene()
        self.rightScene.setSceneRect(QRectF(958, 0, 592, 592))
        self.rightScene.setBackgroundBrush(QColor(200, 200, 200))
        

        #LOWER TRESHOLD SLIDER

        self.lowerTresholdSlider = QSlider(Qt.Orientation.Horizontal)
        self.lowerTresholdSlider.setMinimum(1)
        self.lowerTresholdSlider.setMaximum(127)
        self.lowerTresholdSlider.setValue(64)
        self.lowerTresholdSlider.setFixedWidth(512)
        self.lowerTresholdSlider.valueChanged.connect(self.lowerTresholdSliderChange)
        self.lowerTresholdSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.lowerTresholdSlider.setTickInterval(1)

        self.lowerTresholdSlider_LBL = QLabel('Lower Treshold = %d px' % self.lowerTresholdSlider.value(), self)
        self.lowerTresholdSlider_LBL.setAlignment(Qt.AlignmentFlag.AlignLeft)


        #UPPER TRESHOLD SLIDER
        
        self.upperTresholdSlider = QSlider(Qt.Orientation.Horizontal)
        self.upperTresholdSlider.setMinimum(128)
        self.upperTresholdSlider.setMaximum(254)
        self.upperTresholdSlider.setValue(192)
        self.upperTresholdSlider.setFixedWidth(512)
        self.upperTresholdSlider.valueChanged.connect(self.upperTresholdSliderChange)
        self.upperTresholdSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.upperTresholdSlider.setTickInterval(1)


        self.upperTresholdSlider_LBL = QLabel('Upper Treshold = %d px' % self.upperTresholdSlider.value(), self)
        self.upperTresholdSlider_LBL.setAlignment(Qt.AlignmentFlag.AlignLeft)


        #FILTER SIZE SLIDER

        self.filterSizeSlider = QSlider(Qt.Orientation.Horizontal)
        self.filterSizeSlider.setMinimum(10)
        self.filterSizeSlider.setMaximum(30)
        self.filterSizeSlider.setValue(20)
        self.filterSizeSlider.setFixedWidth(512)
        self.filterSizeSlider.valueChanged.connect(self.onSliderChange)
        self.filterSizeSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.filterSizeSlider.setTickInterval(1)

        self.filtersize_LBL = QLabel('Filter Size = %d px' % self.filterSizeSlider.value(), self)
        self.filtersize_LBL.setAlignment(Qt.AlignmentFlag.AlignLeft)

        #***

        infoText = QLabel('No filter selected. Please select a filter.', self)
        infoText.setAlignment(Qt.AlignmentFlag.AlignCenter)

        inputDefinition_LBL = QLabel('Input Definition', self)
        inputDefinition_LBL.setAlignment(Qt.AlignmentFlag.AlignCenter)

        changeInput_BTN = QPushButton('Change Input', self)
        changeInput_BTN.clicked.connect(self.changeInput)

        nextChannel_BTN = QPushButton('Next Channel', self)
        nextChannel_BTN.clicked.connect(self.changeInput)

        prevChannel_BTN = QPushButton('Previous Channel', self)
        prevChannel_BTN.clicked.connect(self.changeInput)

        selectFilter_BTN = QPushButton('Select Custom Filter', self)
        selectFilter_BTN.clicked.connect(self.selectFilter)

        randomFilter_BTN = QPushButton('Generate Random Filter', self)
        randomFilter_BTN.clicked.connect(self.randomFilter)

        convolute_BTN = QPushButton('--> Convolute -->', self)
        convolute_BTN.clicked.connect(self.convolute)

        quit_BTN = QPushButton('QUIT', self)
        quit_BTN.clicked.connect(self.close)

        saveOutput_BTN = QPushButton('Save Output', self)
        saveOutput_BTN.clicked.connect(self.selectFilter)


        #sc = MplCanvas(self, width=5, height=4, dpi=100)
        #sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        
    
       


        #self.resize(1000, 600)
        self.center()
        self.setWindowTitle('PyQt6 Example')
        self.setFixedSize(1560, 800)

        frameL = QFrame()
        frameL.setFixedSize(512, 512)
        frameL.setStyleSheet("background-color: Gray; border: 2px solid darkGray;")

        frameC = QFrame()
        frameC.setFixedSize(256, 256)
        frameC.setStyleSheet("background-color: lightGray; border: 2px solid darkGray;")

        frameR = QFrame()
        frameR.setFixedSize(512, 512)
        frameR.setStyleSheet("background-color: lightGray; border: 2px solid darkGray;")

        mainHBox.addLayout(self.leftVBOX)
        mainHBox.addLayout(self.centerVBOX)
        mainHBox.addLayout(self.rightVBOX)

        #mainHBox.addWidget(self.leftScene)
        #mainHBox.addWidget(self.rightScene)

        self.leftVBOX.addWidget(inputDefinition_LBL)
        #self.leftVBOX.addWidget(frameL)
        self.leftVBOX.addLayout(leftHBOX)
        leftHBOX.addWidget(changeInput_BTN)
        leftHBOX.addWidget(prevChannel_BTN)
        leftHBOX.addWidget(nextChannel_BTN)
        
        self.centerVBOX.addWidget(self.filtersize_LBL)
        self.centerVBOX.addWidget(self.filterSizeSlider)
        self.centerVBOX.addWidget(self.upperTresholdSlider_LBL)
        self.centerVBOX.addWidget(self.upperTresholdSlider)
        self.centerVBOX.addWidget(self.lowerTresholdSlider_LBL)
        self.centerVBOX.addWidget(self.lowerTresholdSlider)
        self.centerVBOX.addWidget(infoText)
        self.centerVBOX.addWidget(randomFilter_BTN)
        self.centerVBOX.addWidget(selectFilter_BTN)
        self.centerVBOX.addWidget(convolute_BTN)
        self.centerVBOX.addWidget(quit_BTN)
    
        #self.rightVBOX.addWidget(frameR)
        self.rightVBOX.addWidget(saveOutput_BTN)
        
    
    inputImageLoaded = False
    filterImageLoaded = False
    def changeInput(self):
        
        if self.inputImageLoaded:
            self.leftVBOX.removeWidget(self.canvasL)
        
        file_filter = 'Image File (*.png *.jpg)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Image File (*.png *.jpg)')
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle('Open File')
        image_path = response[0]
        self.input = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.input = cv2.resize(self.input, (512,512))
        print(f'Selected File: {response[0]}')


        self.figure, self.ax = plt.subplots(figsize=(6,6), dpi=72)
        self.canvasL = FigureCanvas(self.figure)
        
        
        self.ax.imshow(self.input, cmap='gray')  # You can specify a colormap, e.g., 'viridis'
        #self.ax.set_title('My Image')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_facecolor('none')
        self.ax.patch.set_alpha(0)
        #self.ax.colorbar()  # Add a colorbar if using a colormap

        self.leftVBOX.addWidget(self.canvasL)
        self.inputImageLoaded = True
        

    filterSize = 20
    upperTreshold = 192
    lowerTreshold = 64
        
    def onSliderChange(self):
        self.filtersize_LBL.setText('Filter Size = %d px' % self.filterSizeSlider.value())
        self.filterSize = self.filterSizeSlider.value()

    def lowerTresholdSliderChange(self):
        self.lowerTresholdSlider_LBL.setText('Lower Treshold = %d' % self.lowerTresholdSlider.value())
        self.lowerTreshold = self.lowerTresholdSlider.value()

    def upperTresholdSliderChange(self):
        self.upperTresholdSlider_LBL.setText('Upper Treshold = %d' % self.upperTresholdSlider.value())
        self.upperTreshold = self.upperTresholdSlider.value()
    

    def randomFilter(self):
        pass

        
    def selectFilter(self):
        if self.filterImageLoaded:
            self.centerVBOX.removeWidget(self.canvasC)
        
        file_filter = 'Image File (*.png *.jpg)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Image File (*.png *.jpg)')
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle('Open File')
        image_path = response[0]
        self.customfilter = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.customfilter = cv2.resize(self.customfilter, (self.filterSize,self.filterSize))
        ret1,plus = cv2.threshold(self.customfilter,127,self.upperTreshold,cv2.THRESH_BINARY)
        ret2,minus = cv2.threshold(self.customfilter,self.lowerTreshold,127,cv2.THRESH_BINARY)
        self.minuses = minus / -255
        self.pluses = plus / 255
        self.customfilter = self.minuses + self.pluses
        print("Pluses")
        print(self.pluses)
        print("Minuses")
        print(self.minuses)
        print("Filter")
        print(self.customfilter)
        print(f'Selected File: {response[0]}')


        self.figure, self.ax = plt.subplots(figsize=(3,3), dpi=72)
        self.canvasC = FigureCanvas(self.figure)
        
        self.ax.imshow(self.filter, cmap='gray')  # You can specify a colormap, e.g., 'viridis'
        #self.ax.set_title('My Image')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_facecolor('none')
        self.ax.patch.set_alpha(0)
        #self.ax.colorbar()  # Add a colorbar if using a colormap
        self.centerVBOX.addWidget(self.canvasC)
        self.filterImageLoaded = True



    def convolute(self):
        self.inputAsArray = np.asarray(self.input)
        self.inputAsArray = self.inputAsArray.reshape((1, 512, 512, 1))
        self.output = self.model.predict(self.inputAsArray)
        self.output = self.output[0,:,:,0]
        updateImage(self.output, self.rightImageBox)


    #custom filter
    def my_filter(self, shape, dtype=None):
        customFilter = customFilter.reshape((self.filterSize, self.filterSize, 1, 1))
        assert customFilter.shape == shape
        return K.variable(customFilter, dtype='float32')
        self.updateFilter(output)
            

    def build_model(self):
        inputs = tf.keras.Input(shape=(512, 512, 1))
        convolute = layers.Conv2D(filters=1, kernel_size = self.filterSize, kernel_initializer=self.my_filter, strides=1, padding='valid') (inputs)
        #flat = layers.Flatten()(x)
        #relu = layers.Dense(128, activation=tf.nn.relu)(flat)
        #dense = layers.Dense(2, activation=tf.nn.softmax)(relu)
        self.model = Model(inputs=inputs, outputs=convolute)
        
        #print("Model Output: "+ str(output))
        #return model
        

    def updateImage(self, image, updateLocation):
        self.figure, self.ax = plt.subplots(figsize=(3,3), dpi=72)
        self.canvasR = FigureCanvas(self.figure)        
        self.ax.imshow(image, cmap='gray')  # You can specify a colormap, e.g., 'viridis'
        #self.ax.set_title('My Image')
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        self.ax.set_facecolor('none')
        self.ax.patch.set_alpha(0)
        #self.ax.colorbar()  # Add a colorbar if using a colormap
        self.rightVBOX.addWidget(self.canvasR)
        
    
    def center(self):

        qr = self.frameGeometry()
        self.cp = self.screen().availableGeometry().center()

        qr.moveCenter(self.cp)
        self.move(qr.topLeft()) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = mainWindow()
    window.show()
    sys.exit(app.exec())


    """