import sys, os
from PyQt6.QtWidgets import QWidget, QPushButton,QFrame,QGridLayout, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QApplication

class checkerPattern:
    def __init__(self, rows=6, columns=6, width=512, height=512, spacing=0):
        self.numberOfRows = rows
        self.numberOfColumns = columns    
        self.width = width
        self.height = height
        self.spacing = spacing
        self.grid = QGridLayout()
        self.grid.setSpacing(spacing)
        self.label = {}
        self.shuffle()
    
    def shuffle(self):
        print(self.numberOfRows)
        k = 0
        for i in range(self.numberOfRows):
            if (self.numberOfColumns % 2) == 0:
                k += 1
            else:
                pass
            for j in range(self.numberOfColumns):
                k += 1
                if (k % 2) == 0:
                    self.label[i+j] = QLabel()
                    self.label[i+j].setStyleSheet("background-color: gray;")
                    self.label[i+j].setFixedHeight(round(self.height/self.numberOfRows))
                    self.label[i+j].setFixedWidth(round(self.width/self.numberOfColumns))
                    self.grid.addWidget(self.label[i+j], i, j)
                else:
                    self.label[i+j] = QLabel()
                    self.label[i+j].setStyleSheet("background-color: white;")
                    self.label[i+j].setFixedHeight(round(self.height/self.numberOfRows))
                    self.label[i+j].setFixedWidth(round(self.width/self.numberOfColumns))
                    self.grid.addWidget(self.label[i+j], i, j)



class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.resize(900, 400)
        self.center()

        mainHBox = QHBoxLayout()
        centerVBOX = QVBoxLayout()

        checker1 = checkerPattern(10,10,512,512,0)
        checker2 = checkerPattern(3,3,256,256,0)
        checker3 = checkerPattern(20,20,512,512,0)
        selectFilter_btn = QPushButton(self)

        def getFileName(self):
            file_filter = 'Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls);; Image File (*.png *.jpg)'
            response = QFileDialog.getOpenFileName(
                parent=Example,
                caption='Select a file',
                directory=os.getcwd(),
                filter=file_filter,
                initialFilter='Excel File (*.xlsx *.xls)'
            )

        

        selectFilter_btn.setText("Select Filter")
        selectFilter_btn.clicked.connect(getFileName)

     

        mainHBox.addLayout(checker1.grid)
        mainHBox.addLayout(centerVBOX)
        centerVBOX.addWidget(selectFilter_btn)
        mainHBox.addLayout(checker3.grid)
        self.setLayout(mainHBox)

        self.setWindowTitle('Center')
        self.show()

    def center(self):

        qr = self.frameGeometry()
        self.cp = self.screen().availableGeometry().center()

        qr.moveCenter(self.cp)
        self.move(qr.topLeft())


def main():

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

    
