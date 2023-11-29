import sys, os
from PyQt6.QtWidgets import QWidget, QPushButton, QFrame, QGridLayout, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QApplication, QSlider
from PyQt6.QtCore import Qt
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

        selectFilter_BTN = QPushButton('Select Filter', self)
        selectFilter_BTN.clicked.connect(self.selectFilter)

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
        self.setFixedSize(1400, 600)

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
        self.filter = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.filter = cv2.resize(self.filter, (self.filterSize,self.filterSize))
        ret1,plus = cv2.threshold(self.filter,127,self.upperTreshold,cv2.THRESH_BINARY)
        ret2,minus = cv2.threshold(self.filter,self.lowerTreshold,127,cv2.THRESH_BINARY)
        self.minuses = minus / -255
        self.pluses = plus / 255
        self.filter = self.minuses + self.pluses
        print("Pluses")
        print(self.pluses)
        print("Minuses")
        print(self.minuses)
        print("Filter")
        print(self.filter)
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
        inputAsArray = np.asarray(self.input)
        inputAsArray = inputAsArray.reshape((1, 512, 512, 1))

        # custom filter
        def my_filter(shape, dtype=None):
            a = self.filter
            

            f = np.array([
                [[[-1]], [[0]], [[1]]],

                [[[-1]], [[0]], [[1]]],

                [[[-1]], [[0]], [[1]]]

            ])
            #print(f.shape)
            a = a.reshape((self.filterSize, self.filterSize, 1, 1))
            assert a.shape == shape
            return K.variable(a, dtype='float32')
            

        def build_model():
            inputs = tf.keras.Input(shape=(512, 512, 1))

            convolute = layers.Conv2D(filters=1, 
                            kernel_size = self.filterSize,
                            kernel_initializer=my_filter,
                            strides=1, 
                            padding='valid') (inputs)

            #flat = layers.Flatten()(x)

            #relu = layers.Dense(128, activation=tf.nn.relu)(flat)

            #dense = layers.Dense(2, activation=tf.nn.softmax)(relu)

            model = Model(inputs=inputs, outputs=convolute)
            return model
        model = build_model()
        output = model.predict(inputAsArray)
        print("Bad: "+ str(output))

        self.figure, self.ax = plt.subplots(figsize=(3,3), dpi=72)
        self.canvasR = FigureCanvas(self.figure)

        output = output[0,:,:,0]
        
        self.ax.imshow(output, cmap='gray')  # You can specify a colormap, e.g., 'viridis'
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