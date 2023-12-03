import sys, os
#PYQT6 Imports
from PyQt6.QtWidgets import QWidget, QPushButton, QFrame, QGridLayout, QLabel, QFileDialog, QDial, QGraphicsPixmapItem, QHBoxLayout, QGraphicsTextItem, QGraphicsView, QVBoxLayout, QGraphicsRectItem, QApplication, QSlider, QGraphicsScene, QGraphicsScene, QGraphicsView, QMainWindow
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
#Tensorflow Imports
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras import layers, initializers
from keras.layers import Dense, Conv2D, InputLayer, MaxPooling2D, Flatten
from keras.models import Model, Sequential
#Others
import numpy as np
import cv2
#Custom Scripts
import Maker

class Contents:
    def __init__(self):
        super().__init__()
        self.maker = Maker()
    
       
        self.image_container_left = self.maker

        
        
        
        (image_source_type="file", image_source="checker.png", x=10, y=10, w=512, h=512)
        self.image_container_center_top = make_image_container(alignment="left", image_source_type="file", image_source="checker.png", x=532, y=10, w=256, h=256)
        self.image_container_center_bot = make_image_container(alignment="left", image_source_type="file", image_source="checker.png", x=532, y=428, w=256, h=256)
        self.image_container_right = make_image_container(image_source_type="file", image_source="checker.png", x=798, y=10, w=512, h=512)
        
        self.change_input_BTN = makeButton("Change Input", self.change_input_press, x=10, y=532)
        self.custom_filter_BTN = makeButton("Load Custom Filter", self.custom_filter_press, x=668, y=276, w=120, h=28)
        
        self.random_filter_BTN = makeButton("Randomize", self.random_filter_press, x=668, y=276+38, w=120, h=28)
        self.invert_filter_BTN = makeButton("Invert", self.invert_filter_press, x=668, y=276+38+38, w=120, h=28)
        self.convolute_BTN = makeButton("Convolute", self.convolute_press, x=668, y=276+38+38+38, w=120, h=28)
        
        self.filter_size_indicator_label = make_label(532, 395, 128, 24, "32 px","center")
        self.filter_size_DIAL = makeDial(self.filter_size_DIAL_value_change, 532+64-48, 305, 96, 96, 2, 64, 32)
       
        self.filter_size_label = make_label(532, 278, 128, 24, "Filter Size Control","center")
   
       
       
        self.items={
            "image_container_left" : self.image_container_left,
            "image_container_center_top" : self.image_container_center_top,
            "image_container_center_bot" : self.image_container_center_bot,
            "image_container_right" : self.image_container_right,
            #"filter_size_SLDR" : self.filter_size_SLDR
        }

        self.random_filter_press()

    def change_input_press(self):
        print("change_input_press")
        self.load_custom_image(filter=False, input=True)
        self.display("image_container_left", self.input_original, 1)
    
    def custom_filter_press(self):
        print("custom_filter_press")
        self.load_custom_image(filter=True, input=False)
        self.display("image_container_center_top", self.filter_original, 2)

    def random_filter_press(self):
        print("random_filter_press")
        self.update_values()   
        self.filter_original = np.random.randint(0, 256, size=(self.filter_size, self.filter_size), dtype=np.uint8)
        #self.randomize(self.filter_size)
        self.display("image_container_center_top", self.filter_original, 2)
        self.display("image_container_center_bot", self.filter_original, 2)
    
    def invert_filter_press(self):
        print("invert_filter_press")
        self.filter_to_be_used = cv2.bitwise_not(self.filter_to_be_used)
        self.display("image_container_center_bot", self.filter_to_be_used, 2)
        #self.change_image("label_center_bot", invert=True, random=False, filter_update=True, filter_size=self.filter_size_DIAL.value())

    
   

    def arbitrary_3x3_initializer(self,shape, dtype=None):
        # Define your custom 3x3 weights here
        weights = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        weights = weights.reshape((3, 3, 1, 1))
        return K.variable(weights,  dtype='float32')
    
    def convolute_press(self):
        print("convolute_press")
        self.model = Sequential()
        self.model.add(Conv2D(filters=1,kernel_size=(3, 3), activation='relu', kernel_initializer=self.arbitrary_3x3_initializer, input_shape=(512, 512, 1)))
        input = cv2.resize(self.input_original, (512,512), interpolation = cv2.INTER_NEAREST)
        input = input.reshape((1, 512, 512, 1))
        input = input/255      
        self.result = self.model.predict(input)
        self.result = self.result[0,:,:,0]
        self.display("image_container_right", self.result, 3)
        cv2.imwrite("result.png", self.result*255) 
        print("Result Shape :" + str(self.result.shape))

        """self.update_values()
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(512, 512, 1)))
        self.model.add(Conv2D(filters=1,kernel_size=(self.filter_size, self.filter_size), strides=(1,1), kernel_initializer=self.initialize_kernel))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        input = cv2.resize(self.input_original, (512,512), interpolation = cv2.INTER_NEAREST)
        input = input.reshape((1, 512, 512, 1))
        print(input.shape)
        self.result = self.model.predict(input)
        self.result = self.result[0,:,:,0]
        self.display("image_container_right", self.result, 3)
        print(self.result.shape) """

    def initialize_kernel(self, shape, dtype=None):
        self.customFilter = self.customFilter.reshape((self.filter_size, self.filter_size, 1, 1))
        assert self.customFilter.shape == shape
        return K.variable(self.customFilter, dtype='float32')
    
    def filter_size_SLDR_value_change(self):
        self.items["filter_size_SLDR"].sliderlabel.setText("Filter Size [px] : %d" % self.filter_size_SLDR.value())

    def filter_size_DIAL_value_change(self):
        self.filter_size = contents.filter_size_DIAL.value()
        self.filter_size_indicator_label.setText(str(self.filter_size)+" px")
        self.filter_to_be_used = cv2.resize(self.filter_original, (self.filter_size,self.filter_size), interpolation = cv2.INTER_NEAREST)
        self.display("image_container_center_bot", self.filter_to_be_used, 2)
        print("filter_size_DIAL_value_change")
    
    #def white_treshold_DIAL_value_change(self):
        #self.update_values()
        #print(self.white_treshold)
        #self.trim_from_top(self.white_treshold)
        #self.display("image_container_center_bot", self.filter_trimmed_from_top, 2)
        #print("white_treshold_DIAL_value_change")
    
    #def black_treshold_DIAL_value_change(self):
        #print("black_treshold_DIAL_value_change")
    
    #def process_filter_press(self):
        #print("process_filter_press")
        #self.update_values()
        #print(self.black_treshold)
        #self.filter_to_be_used = cv2.resize(self.filter_original, (self.filter_size,self.filter_size), interpolation = cv2.INTER_NEAREST)
        #self.display("image_container_center_bot", self.filter_to_be_used, 2)

    def update_values(self):
        self.filter_size = contents.filter_size_DIAL.value()
        #self.white_treshold = contents.white_treshold_DIAL.value()
        #self.black_treshold = contents.black_treshold_DIAL.value()
        return self.filter_size
        
    def load_custom_image(self, filter=True, input=False):
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
        if filter:
            self.filter_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(self.filter_original.shape)
            
        if input:
            self.input_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print(self.input_original.shape)



        
        
    #def trim_from_top(self, white_treshold):
        #ret,self.filter_trimmed_from_top = cv2.threshold(self.filter_original,white_treshold,255,cv2.THRESH_TRUNC)
        #ret1,self.filter_original = cv2.threshold(self.filter_original,0,self.white_treshold,cv2.THRESH_BINARY)

    #def trim_from_bottom(self, black_treshold):
        #ret,self.filter_trimmed_from_bottom = cv2.threshold(self.filter_original,0,black_treshold,cv2.THRESH_TRUNC)
        #ret1,self.filter_original = cv2.threshold(self.filter_original,0,black_treshold,cv2.THRESH_BINARY)

    def display(self,  image_container_name, image, image_category=1):
        display_width = contents.items[image_container_name].width()
        display_height = contents.items[image_container_name].height()
        print("Display Height:" + str(display_height))
        print("Display Width:" + str(display_width))
        if image_category == 2:
            self.customFilter = image
            filter_to_display = cv2.resize(image, (display_width,display_height), interpolation = cv2.INTER_NEAREST)
            filter_to_display_pix = QImage(filter_to_display, display_width, display_height, (1*display_width), QImage.Format.Format_Grayscale8)
            self.filter_display_pixmap = QPixmap.fromImage(filter_to_display_pix)
            contents.items[image_container_name].setPixmap(self.filter_display_pixmap)
        if image_category == 1:
            input_to_display = cv2.resize(image, (display_width,display_height), interpolation = cv2.INTER_NEAREST)
            input_to_display_pix = QImage(input_to_display, display_width, display_height, (1*display_width), QImage.Format.Format_Grayscale8)
            self.input_display_pixmap = QPixmap.fromImage(input_to_display_pix)
            contents.items[image_container_name].setPixmap(self.input_display_pixmap)
        if image_category == 3:
            output_to_display = cv2.resize(image, (display_width,display_height), interpolation = cv2.INTER_NEAREST)
            output_to_display_pix = QImage(output_to_display, display_width, display_height, (1*display_width), QImage.Format.Format_Grayscale8)
            self.output_display_pixmap = QPixmap.fromImage(output_to_display_pix)
            contents.items[image_container_name].setPixmap(self.output_display_pixmap)
        if image_category == None:
            print("No image name defined. Define image name as, input_original or filter_original or output_original.")
        






        #self.change_image("label_center_bot", invert=False, random=False, black_treshold_change=True, filter_update=True, filter_size=self.filter_size_DIAL.value())
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
        #flat = layers.Flatten()(convolute)
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