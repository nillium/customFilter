import sys, os
from PyQt6.QtWidgets import QWidget, QPushButton, QFrame, QGridLayout, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt

class mainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        mainHBox = QHBoxLayout(self)
        leftVBOX = QVBoxLayout()
        centerVBOX = QVBoxLayout()
        rightVBOX = QVBoxLayout()

        infoText = QLabel('Hello, PyQt6!')
        infoText.setAlignment(Qt.AlignmentFlag.AlignCenter)

        changeInput_BTN = QPushButton('Change Input', self)
        changeInput_BTN.clicked.connect(self.showFileDialog)

        selectFilter_BTN = QPushButton('Select Filter', self)
        selectFilter_BTN.clicked.connect(self.showFileDialog)

        convolute_BTN = QPushButton('--> Convolute -->', self)
        convolute_BTN.clicked.connect(self.showFileDialog)

        quit_BTN = QPushButton('QUIT', self)
        quit_BTN.clicked.connect(self.showFileDialog)

        saveOutput_BTN = QPushButton('Save Output', self)
        saveOutput_BTN.clicked.connect(self.showFileDialog)

        self.resize(900, 400)
        self.center()
        self.setWindowTitle('PyQt6 Example')

        frameL = QFrame(self)
        frameL.setFixedSize(512, 512)
        frameL.setStyleSheet("background-color: lightGray; border: 2px solid darkGray;")

        frameC = QFrame(self)
        frameC.setFixedSize(256, 256)
        frameC.setStyleSheet("background-color: lightGray; border: 2px solid darkGray;")

        frameR = QFrame(self)
        frameR.setFixedSize(512, 512)
        frameR.setStyleSheet("background-color: lightGray; border: 2px solid darkGray;")

        mainHBox.addLayout(leftVBOX)
        mainHBox.addLayout(centerVBOX)
        mainHBox.addLayout(rightVBOX)

        leftVBOX.addWidget(frameL)
        leftVBOX.addWidget(changeInput_BTN)

        centerVBOX.addWidget(frameC)
        centerVBOX.addWidget(infoText)
        centerVBOX.addWidget(selectFilter_BTN)
        centerVBOX.addWidget(convolute_BTN)
        centerVBOX.addWidget(quit_BTN)
    
        rightVBOX.addWidget(frameR)
        rightVBOX.addWidget(saveOutput_BTN)
        
        
        
        
        

    def showFileDialog(self):
        file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls);; Image File (*.png *.jpg)'
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select a file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Excel File (*.xlsx *.xls)')
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle('Open File')
        print(f'Selected File: {response}')
    
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