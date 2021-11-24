import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, qApp,  QMenu, QApplication, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QImage, QTransform
import cv2
import numpy as np

class Win(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.resize(800, 600)
        # self.move(300, 50)
        self.setWindowTitle('2D Image Manipulator Editor')

        self.createMenuBar()
        self.createToolbar()
        self.newFile()
        self.show()

    def createMenuBar(self):
        newAct = QAction(QIcon('newFile.png'), 'New', self)
        newAct.setShortcut('Ctrl+N')
        newAct.setStatusTip('Create a new picture.')
        newAct.triggered.connect(self.newFile)

        openAct = QAction(QIcon('openFile.png'),'Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open an existing picture.')
        openAct.triggered.connect(self.openFile)

        saveAct = QAction(QIcon('saveIcon.jpg'),'Save',self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('Save the current picture.')
        saveAct.triggered.connect(self.saveImg)

        exitAct = QAction(QIcon('exit.jpg'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application. ')
        exitAct.triggered.connect(qApp.quit)

        cropAct = QAction('&Crop',self)
        cropAct.setStatusTip('Crop the picture so it only contains the current selection. ')
        cropAct.triggered.connect(self.cropDialog)

        resizeAct = QAction('&Resize',self)
        resizeAct.setStatusTip('Resize the picture. ')
        resizeAct.triggered.connect(self.resizeDialog)

        rotateAct = QAction('&Rotate',self)
        rotateAct.setStatusTip('Rotate the picture. ')
        rotateAct.triggered.connect(self.rotateDialog)

        translateAct = QAction('&Translate',self)
        translateAct.setStatusTip('Translate the picture. ')
        translateAct.triggered.connect(self.translateDialog)

        self.colMenu = QMenu('Color Converter', self)
        self.colorMenu()

        rgbAct = QAction('&RGB Pixel', self)
        rgbAct.setStatusTip('Extracting the RGB values of a pixel')
        rgbAct.triggered.connect(self.rgbDialog)

        self.statusbar=self.statusBar()

        viewStatAct = QAction('View statusbar', self, checkable=True)
        viewStatAct.setStatusTip('View statusbar')
        viewStatAct.setChecked(True)
        viewStatAct.triggered.connect(self.toggleMenu)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newAct)
        fileMenu.addAction(openAct)
        fileMenu.addAction(saveAct)
        fileMenu.addAction(exitAct)

        funcMenu = menubar.addMenu('&Function')
        funcMenu.addAction(cropAct)
        funcMenu.addAction(resizeAct)
        funcMenu.addAction(rotateAct)
        funcMenu.addAction(translateAct)
        funcMenu.addMenu(self.colMenu)

        viewMenu = menubar.addMenu('View')
        viewMenu.addAction(rgbAct)
        viewMenu.addAction(viewStatAct)

    def colorMenu(self):
        act1 = QAction('Gray', self)
        act1.setStatusTip('Convert image color to gray scale')
        act1.triggered.connect(self.convertGray)
        act2 = QAction('BGRA', self)
        act2.setStatusTip('Convert image color to BGRA')
        act2.triggered.connect(self.convertBGRA)
        act3 = QAction('HSV', self)
        act3.setStatusTip('Convert image color to HSV')
        act3.triggered.connect(self.convertHSV)
        self.colMenu.addAction(act1)
        self.colMenu.addAction(act2)
        self.colMenu.addAction(act3)

    def toggleMenu(self, state):

        if state:
            self.statusbar.show()
        else:
            self.statusbar.hide()

    def createToolbar(self):

        drawCircle = QAction(QIcon('circle.png'),'Circle',self)
        drawCircle.triggered.connect(self.cirDialog)
        drawRect = QAction(QIcon('rectangle.png'),'Rect', self)
        drawRect.triggered.connect(self.rectDialog)
        drawEllipse = QAction(QIcon('ellipse.png'),'Ellipse', self)
        drawEllipse.triggered.connect(self.ellDialog)
        drawLine = QAction(QIcon('line.png'),'Line', self)
        drawLine.triggered.connect(self.lineDialog)
        drawPolygon = QAction(QIcon('polygon.png'),'Polygon', self)
        drawPolygon.triggered.connect(self.polyDialog)
        writeText = QAction('Text', self)
        writeText.triggered.connect(self.txtDialog)

        self.toolbar = self.addToolBar('Draw')
        self.toolbar.addAction(drawCircle)
        self.toolbar.addAction(drawRect)
        self.toolbar.addAction(drawEllipse)
        self.toolbar.addAction(drawLine)
        self.toolbar.addAction(drawPolygon)
        self.toolbar.addAction(writeText)

    def contextMenuEvent(self, event):
        cmenu = QMenu(self)

        newAct = cmenu.addAction("New")
        openAct = cmenu.addAction("Open")
        saveAct = cmenu.addAction("Save")
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.pos()))

        if action == newAct:
            self.newFile()
        elif action == openAct:
            self.openFile()
        elif action == saveAct:
            self.saveImg()
        elif action == quitAct:
            qApp.quit()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Close',"Are you sure to quit?",
                                     QMessageBox.Yes |QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:

            event.accept()
        else:

            event.ignore()

    def newFile(self):
        self.image = cv2.imread('white.jpg')
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.vbox = QtWidgets.QVBoxLayout(self.central_widget)

        self.label = QtWidgets.QLabel(self)
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                               QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img)
        self.label.setPixmap(self.pixmap)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText('Image Width: ' + str(self.imgWidth) + ' pixels \t Image Height : ' +
                             str(self.imgHeight) + ' pixels')

        self.vbox.addWidget(self.label)
        self.vbox.addWidget(self.label_2)

    def openFile(self):
        self.name = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.image = cv2.imread(self.name)
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        self.pixmap = QPixmap(self.name)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

        self.label_2.setText('Image Width: ' + str(self.imgWidth) + ' pixels \t Image Height : ' +
                             str(self.imgHeight) + ' pixels')

    def keepRatio(self):
        if self.imgHeight>550 and self.imgWidth<=780 :
            self.pixmap = self.pixmap.scaled(self.imgWidth, 550, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight<=550 and self.imgWidth>780 :
            self.pixmap = self.pixmap.scaled(780, self.imgHeight, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight > 550 and self.imgWidth > 780:
            self.pixmap = self.pixmap.scaled(780, 550, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight <= 550 and self.imgWidth <= 780:
            self.pixmap = self.pixmap.scaled(self.imgWidth, self.imgHeight, QtCore.Qt.KeepAspectRatio)

    def saveImg(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\',
                                                              "Image Files (*.jpg)")
        if fname:
            cv2.imwrite(fname, self.image)
        else:
            print('Error')

    def cropDialog(self):
        self.Cdlg = QMainWindow(self)
        self.Cdlg.setWindowTitle('Crop')

        self.Cdlg.central_widget = QtWidgets.QWidget()
        self.Cdlg.setCentralWidget(self.Cdlg.central_widget)
        self.Cdlg.grid = QtWidgets.QGridLayout(self.Cdlg.central_widget)

        self.Cdlg.label = QtWidgets.QLabel(self)
        self.Cdlg.label.setStyleSheet("font : 20pt")
        self.Cdlg.label.setText('Crop in Rectangular Shape')
        self.Cdlg.grid.addWidget(self.Cdlg.label, 0, 0, 1, 5)

        self.Cdlg.label_2 = QtWidgets.QLabel(self)
        self.Cdlg.label_2.setText(' ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_2,1,0,1,1)

        self.Cdlg.label_3 = QtWidgets.QLabel(self)
        self.Cdlg.label_3.setText('Start Point : ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_3,2,0,1,3)

        self.Cdlg.label_4 = QtWidgets.QLabel(self)
        self.Cdlg.label_4.setText('x = ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_4,3,0,1,1)

        self.start_x=QtWidgets.QLineEdit(self)
        self.Cdlg.grid.addWidget(self.start_x,3,1,1,1)

        self.Cdlg.label_5 = QtWidgets.QLabel(self)
        self.Cdlg.label_5.setText('y = ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_5,3,3,1,1)

        self.start_y = QtWidgets.QLineEdit(self)
        self.Cdlg.grid.addWidget(self.start_y,3,4,1,1)

        self.Cdlg.label_6 = QtWidgets.QLabel(self)
        self.Cdlg.label_6.setText('')
        self.Cdlg.grid.addWidget(self.Cdlg.label_6,4,0,1,1)

        self.Cdlg.label_7 = QtWidgets.QLabel(self)
        self.Cdlg.label_7.setText('End Point : ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_7,5,0,1,3)

        self.Cdlg.label_8 = QtWidgets.QLabel(self)
        self.Cdlg.label_8.setText('x = ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_8,6,0,1,1)

        self.end_x = QtWidgets.QLineEdit(self)
        self.Cdlg.grid.addWidget(self.end_x,6,1,1,1)

        self.Cdlg.label_9 = QtWidgets.QLabel(self)
        self.Cdlg.label_9.setText('y = ')
        self.Cdlg.grid.addWidget(self.Cdlg.label_9,6,3,1,1)

        self.end_y=QtWidgets.QLineEdit(self)
        self.Cdlg.grid.addWidget(self.end_y,6,4,1,1)

        self.Cdlg.label_10 = QtWidgets.QLabel(self)
        self.Cdlg.label_10.setText('')
        self.Cdlg.grid.addWidget(self.Cdlg.label_10,7,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.cropImg)
        btn.resize(btn.sizeHint())
        self.Cdlg.grid.addWidget(btn,8,0,1,5)

        self.Cdlg.show()

    def cropImg(self):
        self.Cdlg.close()
        startCol=int(self.start_x.text())
        startRow=int(self.start_y.text())
        endCol=int(self.end_x.text())
        endRow=int(self.end_y.text())

        self.image = self.image[startRow:endRow, startCol:endCol]
        # cv2.imshow('Croped image', self.image)

        self.imgHeight=self.image.shape[0]
        self.imgWidth=self.image.shape[1]

        self.pixmap = self.pixmap.copy(QtCore.QRect(startCol,startRow,self.imgWidth,self.imgHeight))
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

        self.label_2.setText('Image Width: ' + str(self.imgWidth) + ' pixels \t Image Height : ' +
                             str(self.imgHeight) + ' pixels')

    def resizeDialog(self):
        self.RIdlg = QMainWindow(self)
        self.RIdlg.setWindowTitle('Resize')

        self.RIdlg.central_widget = QtWidgets.QWidget()
        self.RIdlg.setCentralWidget(self.RIdlg.central_widget)
        self.RIdlg.grid = QtWidgets.QGridLayout(self.RIdlg.central_widget)

        self.RIdlg.label = QtWidgets.QLabel(self)
        self.RIdlg.label.setStyleSheet("font : 20pt")
        self.RIdlg.label.setText('Resize by Pixels')
        self.RIdlg.grid.addWidget(self.RIdlg.label,0,0,1,4)

        self.RIdlg.label_2 = QtWidgets.QLabel(self)
        self.RIdlg.label_2.setText('')
        self.RIdlg.grid.addWidget(self.RIdlg.label_2, 1, 0, 1, 4)

        self.RIdlg.label_3 = QtWidgets.QLabel(self)
        self.RIdlg.label_3.setText('Image Width : ')
        self.RIdlg.grid.addWidget(self.RIdlg.label_3,2,0,1,2)

        self.widthInput=QtWidgets.QLineEdit(self)
        self.RIdlg.grid.addWidget(self.widthInput,2,2,1,1)

        self.RIdlg.label_4 = QtWidgets.QLabel(self)
        self.RIdlg.label_4.setText('pixels')
        self.RIdlg.grid.addWidget(self.RIdlg.label_4, 2, 3, 1, 1)

        self.RIdlg.label_5 = QtWidgets.QLabel(self)
        self.RIdlg.label_5.setText('Image Height : ')
        self.RIdlg.grid.addWidget(self.RIdlg.label_5,3,0,1,2)

        self.heightInput=QtWidgets.QLineEdit(self)
        self.RIdlg.grid.addWidget(self.heightInput,3,2,1,1)

        self.RIdlg.label_6 = QtWidgets.QLabel(self)
        self.RIdlg.label_6.setText('pixels')
        self.RIdlg.grid.addWidget(self.RIdlg.label_6, 3, 3, 1, 1)

        self.RIdlg.label_7 = QtWidgets.QLabel(self)
        self.RIdlg.label_7.setText('')
        self.RIdlg.grid.addWidget(self.RIdlg.label_7,4,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.resizeImg)
        btn.resize(btn.sizeHint())
        self.RIdlg.grid.addWidget(btn,5,0,1,4)

        self.RIdlg.show()

    def resizeImg(self):
        self.RIdlg.close()
        self.imgHeight = int(self.heightInput.text())
        self.imgWidth = int(self.widthInput.text())

        dimension = (self.imgWidth,self.imgHeight)
        self.image = cv2.resize(self.image, dimension)
        # cv2.imshow('resize',self.image)

        self.keepRatio()
        self.label.setPixmap(self.pixmap)

        self.label_2.setText('Image Width: ' + str(self.imgWidth) + ' pixels \t Image Height : ' +
                             str(self.imgHeight) + ' pixels')

    def rotateDialog(self):
        self.RotDlg = QMainWindow(self)
        self.RotDlg.setWindowTitle('Rotate')

        self.RotDlg.central_widget = QtWidgets.QWidget()
        self.RotDlg.setCentralWidget(self.RotDlg.central_widget)
        self.RotDlg.grid = QtWidgets.QGridLayout(self.RotDlg.central_widget)

        self.RotDlg.label = QtWidgets.QLabel(self)
        self.RotDlg.label.setStyleSheet("font : 20pt")
        self.RotDlg.label.setText('Rotate in Clockwise Direction')
        self.RotDlg.grid.addWidget(self.RotDlg.label,0,0,1,6)

        self.RotDlg.label_2 = QtWidgets.QLabel(self)
        self.RotDlg.label_2.setText(' ')
        self.RotDlg.grid.addWidget(self.RotDlg.label_2,1,0,1,1)

        self.RotDlg.label_3 = QtWidgets.QLabel(self)
        self.RotDlg.label_3.setText('Degree : ')
        self.RotDlg.grid.addWidget(self.RotDlg.label_3,2,0,1,1)

        self.degreeInput=QtWidgets.QLineEdit(self)
        self.RotDlg.grid.addWidget(self.degreeInput,2,1,1,2)

        self.RotDlg.label_4 = QtWidgets.QLabel(self)
        self.RotDlg.label_4.setText('(Insert negative degree value if rotate in anticlowise direction)')
        self.RotDlg.grid.addWidget(self.RotDlg.label_4,3,1,1,5)

        self.RotDlg.label_5 = QtWidgets.QLabel(self)
        self.RotDlg.label_5.setText(' ')
        self.RotDlg.grid.addWidget(self.RotDlg.label_5, 4, 0, 1, 1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.rotateImg)
        btn.resize(btn.sizeHint())
        self.RotDlg.grid.addWidget(btn,5,0,1,6)

        self.RotDlg.show()

    def rotateImg(self):
        self.RotDlg.close()
        angle=int(self.degreeInput.text())*-1
        scale=1.0
        self.center=(self.image.shape[1]/2,self.image.shape[0]/2)
        M = cv2.getRotationMatrix2D(self.center, angle, scale)
        self.image=cv2.warpAffine(self.image,M,(self.image.shape[0],self.image.shape[1]))
        # strcaption = "Image Rotated by " + str(angle) + " degree"
        # cv2.imshow(strcaption, self.image)

        transform = QTransform().rotate(angle)
        self.pixmap = self.pixmap.transformed(transform)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def translateDialog(self):
        self.TDlg = QMainWindow(self)
        self.TDlg.resize(200, 200)
        self.TDlg.setWindowTitle('Translate')

        self.TDlg.central_widget = QtWidgets.QWidget()
        self.TDlg.setCentralWidget(self.TDlg.central_widget)
        self.TDlg.grid = QtWidgets.QGridLayout(self.TDlg.central_widget)

        self.TDlg.label = QtWidgets.QLabel(self)
        self.TDlg.label.setStyleSheet("font : 20pt")
        self.TDlg.label.setText('Translate Image')
        self.TDlg.grid.addWidget(self.TDlg.label,0,0,1,3)

        self.TDlg.label_2 = QtWidgets.QLabel(self)
        self.TDlg.label_2.setText('')
        self.TDlg.grid.addWidget(self.TDlg.label_2,1,0,1,1)

        self.TDlg.label_3 = QtWidgets.QLabel(self)
        self.TDlg.label_3.setText('tx : ')
        self.TDlg.grid.addWidget(self.TDlg.label_3, 2, 0, 1, 1)

        self.txInput=QtWidgets.QLineEdit(self)
        self.TDlg.grid.addWidget(self.txInput,2,1,1,1)

        self.TDlg.label_4 = QtWidgets.QLabel(self)
        self.TDlg.label_4.setText('(positive value towards right direction)')
        self.TDlg.grid.addWidget(self.TDlg.label_4, 3, 1, 1, 2)

        self.TDlg.label_5 = QtWidgets.QLabel(self)
        self.TDlg.label_5.setText(' ')
        self.TDlg.grid.addWidget(self.TDlg.label_5, 4, 0, 1, 1)

        self.TDlg.label_6 = QtWidgets.QLabel(self)
        self.TDlg.label_6.setText('ty: ')
        self.TDlg.grid.addWidget(self.TDlg.label_6,5,0,1,1)

        self.tyInput = QtWidgets.QLineEdit(self)
        self.TDlg.grid.addWidget(self.tyInput,5,1,1,1)

        self.TDlg.label_7 = QtWidgets.QLabel(self)
        self.TDlg.label_7.setText('(positive value towards down direction)')
        self.TDlg.grid.addWidget(self.TDlg.label_7, 6, 1, 1, 2)

        self.TDlg.label_8 = QtWidgets.QLabel(self)
        self.TDlg.label_8.setText('')
        self.TDlg.grid.addWidget(self.TDlg.label_8,7,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.translateImg)
        btn.resize(btn.sizeHint())
        self.TDlg.grid.addWidget(btn,8,0,1,4)

        self.TDlg.show()

    def translateImg(self):
        self.TDlg.close()
        tx=int(self.txInput.text())
        ty=int(self.tyInput.text())

        translationMatrix = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
        self.image = cv2.warpAffine(self.image, translationMatrix, (int(self.imgWidth), int(self.imgHeight)))
        # cv2.imshow("after translation",self.image )

        img_translate = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*3,
                            QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_translate)
        self.pixmap = self.pixmap.scaled(int(self.imgWidth),int(self.imgHeight),
                                         QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def convertGray(self):
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGRA2GRAY)
        # cv2.imshow("Gray Scale color image",self.image)

        gray_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1], QImage.Format_Grayscale8)
        self.pixmap = QPixmap(gray_image)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def convertBGRA(self):
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2BGRA)
        # cv2.imshow("Color image",self.image)

        color_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*4,
                            QImage.Format_ARGB32)
        self.pixmap = QPixmap(color_image)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def convertHSV(self):
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV image",self.image)

        hsv_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*3,
                            QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(hsv_image)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def rgbDialog(self):
        self.rgbDlg = QMainWindow(self)
        self.rgbDlg.setWindowTitle('RGB Pixel')

        self.rgbDlg.central_widget = QtWidgets.QWidget()
        self.rgbDlg.setCentralWidget(self.rgbDlg.central_widget)
        self.rgbDlg.grid = QtWidgets.QGridLayout(self.rgbDlg.central_widget)

        self.rgbDlg.label = QtWidgets.QLabel(self)
        self.rgbDlg.label.setStyleSheet("font : 20pt")
        self.rgbDlg.label.setText('Extract RGB Value of a Pixel')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label, 0, 0, 1, 5)

        self.rgbDlg.label_2 = QtWidgets.QLabel(self)
        self.rgbDlg.label_2.setText(' ')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label_2, 1, 0, 1, 1)

        self.rgbDlg.label_3 = QtWidgets.QLabel(self)
        self.rgbDlg.label_3.setText('Coordinate Pixel : ')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label_3, 2, 0, 1, 3)

        self.rgbDlg.label_4 = QtWidgets.QLabel(self)
        self.rgbDlg.label_4.setText('x = ')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label_4, 3, 0, 1, 1)

        self.x_rgb = QtWidgets.QLineEdit(self)
        self.rgbDlg.grid.addWidget(self.x_rgb, 3, 1, 1, 1)

        self.rgbDlg.label_5 = QtWidgets.QLabel(self)
        self.rgbDlg.label_5.setText('y = ')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label_5, 3, 3, 1, 1)

        self.y_rgb = QtWidgets.QLineEdit(self)
        self.rgbDlg.grid.addWidget(self.y_rgb, 3, 4, 1, 1)

        self.rgbDlg.label_6 = QtWidgets.QLabel(self)
        self.rgbDlg.label_6.setText('')
        self.rgbDlg.grid.addWidget(self.rgbDlg.label_6, 4, 0, 1, 1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.showRGB)
        btn.resize(btn.sizeHint())
        self.rgbDlg.grid.addWidget(btn, 5, 0, 1, 5)

        self.rgbDlg.show()

    def showRGB(self):
        self.rgbDlg.close()

        self.showDlg = QMainWindow(self)
        title= "RGB Value of ["+str(self.x_rgb.text())+" , "+str(self.y_rgb.text())+"] "
        self.showDlg.setWindowTitle(title)

        self.showDlg.central_widget = QtWidgets.QWidget()
        self.showDlg.setCentralWidget(self.showDlg.central_widget)
        self.showDlg.vbox = QtWidgets.QVBoxLayout(self.showDlg.central_widget)

        self.showDlg.label = QtWidgets.QLabel(self)
        pixmap = self.pixmap.copy(QtCore.QRect(int(self.x_rgb.text()), int(self.y_rgb.text()), 1,1))
        pixmap = pixmap.scaled(100, 100)
        self.showDlg.label.setPixmap(pixmap)
        self.showDlg.vbox.addWidget(self.showDlg.label)

        (b, g, r) = self.image[int(self.x_rgb.text()), int(self.y_rgb.text())]
        string = "Pixel at [" + str(self.x_rgb.text()) + "," + str(self.y_rgb.text()) + "] - R:{}, G:{}, B:{}".format(
            r, g, b)
        self.showDlg.label_2 = QtWidgets.QLabel(self)
        self.showDlg.label_2.setStyleSheet("font : 15pt")
        self.showDlg.label_2.setText(string)
        self.showDlg.vbox.addWidget(self.showDlg.label_2)

        self.showDlg.label_3 = QtWidgets.QLabel(self)
        self.showDlg.label_3.setText('')
        self.showDlg.vbox.addWidget(self.showDlg.label_3)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.closeRGB)
        btn.resize(btn.sizeHint())
        self.showDlg.vbox.addWidget(btn)

        self.showDlg.show()

    def closeRGB(self):
        self.showDlg.close()

    def cirDialog(self):
        self.circleDlg= QMainWindow(self)
        self.circleDlg.setWindowTitle('Circle')

        self.circleDlg.central_widget = QtWidgets.QWidget()
        self.circleDlg.setCentralWidget(self.circleDlg.central_widget)
        self.circleDlg.grid = QtWidgets.QGridLayout(self.circleDlg.central_widget)

        self.circleDlg.label = QtWidgets.QLabel(self)
        self.circleDlg.label.setStyleSheet("font : 20pt")
        self.circleDlg.label.setText('Draw a Circle on Image ')
        self.circleDlg.grid.addWidget(self.circleDlg.label,0,0,1,8)

        self.circleDlg.label_2 = QtWidgets.QLabel(self)
        self.circleDlg.label_2.setText('Center of Circle : ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_2,2,0,1,3)

        self.circleDlg.label_3 = QtWidgets.QLabel(self)
        self.circleDlg.label_3.setText('x = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_3,3,0,1,1)

        self.x_circle=QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.x_circle,3,1,1,2)

        self.circleDlg.label_4 = QtWidgets.QLabel(self)
        self.circleDlg.label_4.setText('y = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_4,3,4,1,1)

        self.y_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.y_circle,3,5,1,2)

        self.circleDlg.label_5 = QtWidgets.QLabel(self)
        self.circleDlg.label_5.setText(' ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_5,4,0,1,1)

        self.circleDlg.label_6 = QtWidgets.QLabel(self)
        self.circleDlg.label_6.setText('Radius of Circle = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_6,5,0,1,2)

        self.r_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.r_circle,5,2,1,2)

        self.circleDlg.label_7 = QtWidgets.QLabel(self)
        self.circleDlg.label_7.setText(' ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_7,6,0,1,1)

        self.circleDlg.label_8 = QtWidgets.QLabel(self)
        self.circleDlg.label_8.setText('Color of Border Line (B,G,R) : ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_8, 7, 0, 1, 3)

        self.circleDlg.label_9 = QtWidgets.QLabel(self)
        self.circleDlg.label_9.setText('Intensity of  Blue  (max 255) = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_9, 8, 0, 1, 3)

        self.blue_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.blue_circle, 8, 3, 1, 2)

        self.circleDlg.label_10 = QtWidgets.QLabel(self)
        self.circleDlg.label_10.setText('Intensity of Green (max 255) = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_10, 9, 0, 1, 3)

        self.green_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.green_circle, 9, 3, 1, 2)

        self.circleDlg.label_11 = QtWidgets.QLabel(self)
        self.circleDlg.label_11.setText('Intensity of   Red   (max 255) = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_11, 10, 0, 1, 3)

        self.red_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.red_circle, 10, 3, 1, 2)

        self.circleDlg.label_12 = QtWidgets.QLabel(self)
        self.circleDlg.label_12.setText(' ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_12, 11, 0, 1, 1)

        self.circleDlg.label_13 = QtWidgets.QLabel(self)
        self.circleDlg.label_13.setText('Thickness of Border Line = ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_13,12,0,1,3)

        self.thick_circle = QtWidgets.QLineEdit(self)
        self.circleDlg.grid.addWidget(self.thick_circle,12,3,1,2)

        self.circleDlg.label_14 = QtWidgets.QLabel(self)
        self.circleDlg.label_14.setText('(-1 to fill circle with (B,G,R) color) ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_14,13,3,1,5)

        self.circleDlg.label_15 = QtWidgets.QLabel(self)
        self.circleDlg.label_15.setText(' ')
        self.circleDlg.grid.addWidget(self.circleDlg.label_15,14,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawCir)
        btn.resize(btn.sizeHint())
        self.circleDlg.grid.addWidget(btn,15,0,1,8)

        self.circleDlg.show()

    def drawCir(self):
        self.circleDlg.close()

        cv2.circle(self.image, (int(self.x_circle.text()), int(self.y_circle.text())), int(self.r_circle.text()),
                   (int(self.blue_circle.text()), int(self.green_circle.text()),int(self.red_circle.text())),
                   int(self.thick_circle.text()))
        # cv2.imshow('draw circle', self.image)

        img_circle = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*3,
                            QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_circle)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def rectDialog(self):
        self.rectDlg= QMainWindow(self)
        self.rectDlg.setWindowTitle('Rectangle')

        self.rectDlg.central_widget = QtWidgets.QWidget()
        self.rectDlg.setCentralWidget(self.rectDlg.central_widget)
        self.rectDlg.grid = QtWidgets.QGridLayout(self.rectDlg.central_widget)

        self.rectDlg.label = QtWidgets.QLabel(self)
        self.rectDlg.label.setStyleSheet("font : 20pt")
        self.rectDlg.label.setText('Draw a Rectangle on Image')
        self.rectDlg.grid.addWidget(self.rectDlg.label,0,0,1,8)

        self.rectDlg.label_2 = QtWidgets.QLabel(self)
        self.rectDlg.label_2.setText('Start Point : ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_2,1,0,1,3)

        self.rectDlg.label_3 = QtWidgets.QLabel(self)
        self.rectDlg.label_3.setText('x = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_3,2,0,1,1)

        self.x1_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.x1_rect,2,1,1,2)

        self.rectDlg.label_4 = QtWidgets.QLabel(self)
        self.rectDlg.label_4.setText('y = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_4,2,4,1,1)

        self.y1_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.y1_rect,2,5,1,2)

        self.rectDlg.label_5 = QtWidgets.QLabel(self)
        self.rectDlg.label_5.setText('')
        self.rectDlg.grid.addWidget(self.rectDlg.label_5,3,0,1,1)

        self.rectDlg.label_6 = QtWidgets.QLabel(self)
        self.rectDlg.label_6.setText('End Point : ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_6,4,0,1,3)

        self.rectDlg.label_7 = QtWidgets.QLabel(self)
        self.rectDlg.label_7.setText('x = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_7,5,0,1,1)

        self.x2_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.x2_rect,5,1,1,2)

        self.rectDlg.label_8 = QtWidgets.QLabel(self)
        self.rectDlg.label_8.setText('y = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_8,5,4,1,1)

        self.y2_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.y2_rect,5,5,1,2)

        self.rectDlg.label_9 = QtWidgets.QLabel(self)
        self.rectDlg.label_9.setText('')
        self.rectDlg.grid.addWidget(self.rectDlg.label_9,6,0,1,1)

        self.rectDlg.label_10 = QtWidgets.QLabel(self)
        self.rectDlg.label_10.setText('Color of Border Line (B,G,R) : ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_10, 7, 0, 1, 3)

        self.rectDlg.label_11 = QtWidgets.QLabel(self)
        self.rectDlg.label_11.setText('Intensity of  Blue  (max 255) = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_11, 8, 0, 1, 3)

        self.blue_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.blue_rect, 8, 3, 1, 2)

        self.rectDlg.label_12 = QtWidgets.QLabel(self)
        self.rectDlg.label_12.setText('Intensity of Green (max 255) = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_12, 9, 0, 1, 3)

        self.green_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.green_rect, 9, 3, 1, 2)

        self.rectDlg.label_13 = QtWidgets.QLabel(self)
        self.rectDlg.label_13.setText('Intensity of   Red   (max 255) = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_13, 10, 0, 1, 3)

        self.red_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.red_rect, 10, 3, 1, 2)

        self.rectDlg.label_14 = QtWidgets.QLabel(self)
        self.rectDlg.label_14.setText(' ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_14, 11, 0, 1, 1)

        self.rectDlg.label_15 = QtWidgets.QLabel(self)
        self.rectDlg.label_15.setText('Thickness of Border Line = ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_15, 12, 0, 1, 3)

        self.thick_rect = QtWidgets.QLineEdit(self)
        self.rectDlg.grid.addWidget(self.thick_rect,12,3,1,2)

        self.rectDlg.label_16 = QtWidgets.QLabel(self)
        self.rectDlg.label_16.setText('(-1 to fill rectangle with (B,G,R) color) ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_16, 13, 3, 1, 5)

        self.rectDlg.label_17 = QtWidgets.QLabel(self)
        self.rectDlg.label_17.setText(' ')
        self.rectDlg.grid.addWidget(self.rectDlg.label_17,14,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawRect)
        btn.resize(btn.sizeHint())
        self.rectDlg.grid.addWidget(btn,15,0,1,8)

        self.rectDlg.show()

    def drawRect(self):
        self.rectDlg.close()

        cv2.rectangle(self.image, (int(self.x1_rect.text()), int(self.y1_rect.text())),
                      (int(self.x2_rect.text()),int(self.y2_rect.text())),
                      (int(self.blue_rect.text()),int(self.green_rect.text()),int(self.red_rect.text())),
                      int(self.thick_rect.text()))
        # cv2.imshow('draw rectangle', self.image)

        img_rect = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*3,
                            QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_rect)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def ellDialog(self):
        self.ellDlg= QMainWindow(self)
        self.ellDlg.setWindowTitle('Ellipse')

        self.ellDlg.central_widget = QtWidgets.QWidget()
        self.ellDlg.setCentralWidget(self.ellDlg.central_widget)
        self.ellDlg.grid = QtWidgets.QGridLayout(self.ellDlg.central_widget)

        self.ellDlg.label = QtWidgets.QLabel(self)
        self.ellDlg.label.setStyleSheet("font : 20pt")
        self.ellDlg.label.setText('Draw an Ellipse on Image ')
        self.ellDlg.grid.addWidget(self.ellDlg.label,0,0,1,7)

        self.ellDlg.label_2 = QtWidgets.QLabel(self)
        self.ellDlg.label_2.setText('Center Point : ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_2,1,0,1,3)

        self.ellDlg.label_3 = QtWidgets.QLabel(self)
        self.ellDlg.label_3.setText('\tx = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_3,2,0,1,1)

        self.x_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.x_ell,2,1,1,2)

        self.ellDlg.label_4 = QtWidgets.QLabel(self)
        self.ellDlg.label_4.setText('\ty = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_4,2,4,1,1)

        self.y_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.y_ell,2,5,1,1)

        self.ellDlg.label_5 = QtWidgets.QLabel(self)
        self.ellDlg.label_5.setText('')
        self.ellDlg.grid.addWidget(self.ellDlg.label_5,3,0,1,1)

        self.ellDlg.label_6 = QtWidgets.QLabel(self)
        self.ellDlg.label_6.setText('Axes Length: ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_6,4,0,1,1)

        self.ellDlg.label_7 = QtWidgets.QLabel(self)
        self.ellDlg.label_7.setText('Major Axis = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_7,5,0,1,1)

        self.major_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.major_ell,5,1,1,2)

        self.ellDlg.label_8 = QtWidgets.QLabel(self)
        self.ellDlg.label_8.setText('Minor Axis = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_8,5,4,1,1)

        self.minor_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.minor_ell,5,5,1,2)

        self.ellDlg.label_9 = QtWidgets.QLabel(self)
        self.ellDlg.label_9.setText('')
        self.ellDlg.grid.addWidget(self.ellDlg.label_9,6,0,1,1)

        self.ellDlg.label_10 = QtWidgets.QLabel(self)
        self.ellDlg.label_10.setText('Rotation Angle (in degree) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_10,7,0,1,2)

        self.angle_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.angle_ell,7,2,1,2)

        self.ellDlg.label_11 = QtWidgets.QLabel(self)
        self.ellDlg.label_11.setText('     Start Angle (in degree) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_11,8,0,1,2)

        self.startAng_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.startAng_ell,8,2,1,2)

        self.ellDlg.label_12 = QtWidgets.QLabel(self)
        self.ellDlg.label_12.setText('(Input 0 to draw full ellipse)')
        self.ellDlg.grid.addWidget(self.ellDlg.label_12, 8, 4, 1, 2)

        self.ellDlg.label_13 = QtWidgets.QLabel(self)
        self.ellDlg.label_13.setText('       End angle (in degree) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_13,9,0,1,2)

        self.endAng_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.endAng_ell,9,2,1,2)

        self.ellDlg.label_14 = QtWidgets.QLabel(self)
        self.ellDlg.label_14.setText('(Input 360 to draw full ellipse)')
        self.ellDlg.grid.addWidget(self.ellDlg.label_14, 9, 4, 1, 2)

        self.ellDlg.label_15 = QtWidgets.QLabel(self)
        self.ellDlg.label_15.setText('')
        self.ellDlg.grid.addWidget(self.ellDlg.label_15,10,0,1,1)

        self.ellDlg.label_16 = QtWidgets.QLabel(self)
        self.ellDlg.label_16.setText('Color of Border Line (B,G,R) : ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_16, 11, 0, 1, 3)

        self.ellDlg.label_17 = QtWidgets.QLabel(self)
        self.ellDlg.label_17.setText('Intensity of  Blue  (max 255) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_17, 12, 0, 1, 2)

        self.blue_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.blue_ell, 12, 2, 1, 2)

        self.ellDlg.label_18 = QtWidgets.QLabel(self)
        self.ellDlg.label_18.setText('Intensity of Green (max 255) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_18, 13, 0, 1, 2)

        self.green_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.green_ell, 13, 2, 1, 2)

        self.ellDlg.label_19 = QtWidgets.QLabel(self)
        self.ellDlg.label_19.setText('Intensity of   Red   (max 255) = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_19, 14, 0, 1, 2)

        self.red_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.red_ell, 14, 2, 1, 2)

        self.ellDlg.label_20 = QtWidgets.QLabel(self)
        self.ellDlg.label_20.setText(' ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_20, 15, 0, 1, 1)

        self.ellDlg.label_21 = QtWidgets.QLabel(self)
        self.ellDlg.label_21.setText('Thickness of Border Line = ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_21, 16, 0, 1, 2)

        self.thick_ell = QtWidgets.QLineEdit(self)
        self.ellDlg.grid.addWidget(self.thick_ell, 16, 2, 1, 2)

        self.ellDlg.label_22 = QtWidgets.QLabel(self)
        self.ellDlg.label_22.setText('(-1 to fill ellipse with (B,G,R) color) ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_22, 17, 2, 1, 5)

        self.ellDlg.label_23 = QtWidgets.QLabel(self)
        self.ellDlg.label_23.setText(' ')
        self.ellDlg.grid.addWidget(self.ellDlg.label_23,18,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawEll)
        btn.resize(btn.sizeHint())
        self.ellDlg.grid.addWidget(btn,19,0,1,7)

        self.ellDlg.show()

    def drawEll(self):
        self.ellDlg.close()

        cv2.ellipse(self.image, (int(self.x_ell.text()), int(self.y_ell.text())),
                    (int(self.major_ell.text()),int(self.minor_ell.text())), float(self.angle_ell.text()),
                    float(self.startAng_ell.text()), float(self.endAng_ell.text()),
                    (int(self.blue_ell.text()),int(self.green_ell.text()),int(self.red_ell.text())),
                    int(self.thick_ell.text()))
        # cv2.imshow('draw ellipse', self.image)

        img_ell = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*3,
                            QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_ell)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def lineDialog(self):
        self.lDlg = QMainWindow(self)
        self.lDlg.setWindowTitle('Line')

        self.lDlg.central_widget = QtWidgets.QWidget()
        self.lDlg.setCentralWidget(self.lDlg.central_widget)
        self.lDlg.grid = QtWidgets.QGridLayout(self.lDlg.central_widget)

        self.lDlg.label = QtWidgets.QLabel(self)
        self.lDlg.label.setStyleSheet("font : 20pt")
        self.lDlg.label.setText('Draw a Line on Image ')
        self.lDlg.grid.addWidget(self.lDlg.label,0,0,1,7)

        self.lDlg.label_2 = QtWidgets.QLabel(self)
        self.lDlg.label_2.setText('Start Point : ')
        self.lDlg.grid.addWidget(self.lDlg.label_2,1,0,1,2)

        self.lDlg.label_3 = QtWidgets.QLabel(self)
        self.lDlg.label_3.setText('x = ')
        self.lDlg.grid.addWidget(self.lDlg.label_3,2,0,1,1)

        self.x1_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.x1_line,2,1,1,2)

        self.lDlg.label_4 = QtWidgets.QLabel(self)
        self.lDlg.label_4.setText('\ty = ')
        self.lDlg.grid.addWidget(self.lDlg.label_4,2,4,1,1)

        self.y1_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.y1_line,2,5,1,1)

        self.lDlg.label_5 = QtWidgets.QLabel(self)
        self.lDlg.label_5.setText('')
        self.lDlg.grid.addWidget(self.lDlg.label_5,3,0,1,1)

        self.lDlg.label_6 = QtWidgets.QLabel(self)
        self.lDlg.label_6.setText('End Point : ')
        self.lDlg.grid.addWidget(self.lDlg.label_6,4,0,1,2)

        self.lDlg.label_7 = QtWidgets.QLabel(self)
        self.lDlg.label_7.setText('x = ')
        self.lDlg.grid.addWidget(self.lDlg.label_7,5,0,1,1)

        self.x2_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.x2_line,5,1,1,2)

        self.lDlg.label_8 = QtWidgets.QLabel(self)
        self.lDlg.label_8.setText('\ty = ')
        self.lDlg.grid.addWidget(self.lDlg.label_8,5,4,1,1)

        self.y2_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.y2_line,5,5,1,1)

        self.lDlg.label_9 = QtWidgets.QLabel(self)
        self.lDlg.label_9.setText('')
        self.lDlg.grid.addWidget(self.lDlg.label_9,6,0,1,1)

        self.lDlg.label_10 = QtWidgets.QLabel(self)
        self.lDlg.label_10.setText('Color of Line (B,G,R) = ')
        self.lDlg.grid.addWidget(self.lDlg.label_10,7,0,1,3)

        self.lDlg.label_11 = QtWidgets.QLabel(self)
        self.lDlg.label_11.setText('Intensity of  Blue  (max 255) = ')
        self.lDlg.grid.addWidget(self.lDlg.label_11,8,0,1,3)

        self.blue_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.blue_line, 8, 3, 1, 2)

        self.lDlg.label_12 = QtWidgets.QLabel(self)
        self.lDlg.label_12.setText('Intensity of Green (max 255) = ')
        self.lDlg.grid.addWidget(self.lDlg.label_12, 9, 0, 1, 3)

        self.green_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.green_line, 9, 3, 1, 2)

        self.lDlg.label_13 = QtWidgets.QLabel(self)
        self.lDlg.label_13.setText('Intensity of   Red   (max 255) = ')
        self.lDlg.grid.addWidget(self.lDlg.label_13, 10, 0, 1, 3)

        self.red_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.red_line, 10, 3, 1, 2)

        self.lDlg.label_14 = QtWidgets.QLabel(self)
        self.lDlg.label_14.setText('')
        self.lDlg.grid.addWidget(self.lDlg.label_14, 11, 0, 1, 3)

        self.lDlg.label_15 = QtWidgets.QLabel(self)
        self.lDlg.label_15.setText('Thickness of Line = ')
        self.lDlg.grid.addWidget(self.lDlg.label_15, 12, 0, 1, 3)

        self.thick_line = QtWidgets.QLineEdit(self)
        self.lDlg.grid.addWidget(self.thick_line,12,2,1,2)

        self.lDlg.label_16 = QtWidgets.QLabel(self)
        self.lDlg.label_16.setText('')
        self.lDlg.grid.addWidget(self.lDlg.label_16, 13, 0, 1, 3)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawLine)
        btn.resize(btn.sizeHint())
        self.lDlg.grid.addWidget(btn,14,0,1,7)

        self.lDlg.show()

    def drawLine(self):
        self.lDlg.close()

        cv2.line(self.image, (int(self.x1_line.text()), int(self.y1_line.text())),
                 (int(self.x2_line.text()), int(self.y2_line.text())),
                 (int(self.blue_line.text()),int(self.green_line.text()),int(self.red_line.text())),
                 int(self.thick_line.text()))
        # cv2.imshow('draw line', self.image)

        img_line = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                          QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_line)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def polyDialog(self):
        self.pDlg = QMainWindow(self)
        self.pDlg.setWindowTitle('Polygon')

        self.pDlg.central_widget = QtWidgets.QWidget()
        self.pDlg.setCentralWidget(self.pDlg.central_widget)
        self.pDlg.vbox = QtWidgets.QVBoxLayout(self.pDlg.central_widget)

        self.pDlg.label = QtWidgets.QLabel(self)
        self.pDlg.label.setStyleSheet("font : 20pt")
        self.pDlg.label.setText('Number of Points : ')
        self.pDlg.vbox.addWidget(self.pDlg.label)

        self.pDlg.label_1 = QtWidgets.QLabel(self)
        self.pDlg.label_1.setText(' ')
        self.pDlg.vbox.addWidget(self.pDlg.label_1)

        btn_3 = QtWidgets.QPushButton('3 (Triangle) ', self)
        btn_3.clicked.connect(self.triDialog)
        btn_3.resize(btn_3.sizeHint())
        self.pDlg.vbox.addWidget(btn_3)

        self.pDlg.label_2 = QtWidgets.QLabel(self)
        self.pDlg.label_2.setText(' ')
        self.pDlg.vbox.addWidget(self.pDlg.label_2)

        btn_4 = QtWidgets.QPushButton('4 (Quadrilateral) ', self)
        btn_4.clicked.connect(self.quadDialog)
        btn_4.resize(btn_4.sizeHint())
        self.pDlg.vbox.addWidget(btn_4)

        self.pDlg.label_3 = QtWidgets.QLabel(self)
        self.pDlg.label_3.setText(' ')
        self.pDlg.vbox.addWidget(self.pDlg.label_3)

        # btn_5 = QtWidgets.QPushButton('5', self)
        # btn_5.clicked.connect(self.penDialog)
        # btn_5.resize(btn_5.sizeHint())
        # self.pDlg.vbox.addWidget(btn_5)
        #
        # self.pDlg.label_4 = QtWidgets.QLabel(self)
        # self.pDlg.label_4.setText(' ')
        # self.pDlg.vbox.addWidget(self.pDlg.label_4)
        #
        # btn_6 = QtWidgets.QPushButton('6', self)
        # btn_6.clicked.connect(self.hexDialog)
        # btn_6.resize(btn_6.sizeHint())
        # self.pDlg.vbox.addWidget(btn_6)
        #
        # self.pDlg.label_5 = QtWidgets.QLabel(self)
        # self.pDlg.label_5.setText(' ')
        # self.pDlg.vbox.addWidget(self.pDlg.label_5)

        self.pDlg.show()

    def triDialog(self):
        self.pDlg.close()

        self.triDlg = QMainWindow(self)
        self.triDlg.setWindowTitle('Triangle')

        self.triDlg.central_widget = QtWidgets.QWidget()
        self.triDlg.setCentralWidget(self.triDlg.central_widget)
        self.triDlg.grid = QtWidgets.QGridLayout(self.triDlg.central_widget)

        self.triDlg.label = QtWidgets.QLabel(self)
        self.triDlg.label.setStyleSheet("font : 20pt")
        self.triDlg.label.setText('Draw a Triangle on Image')
        self.triDlg.grid.addWidget(self.triDlg.label,0,0,1,8)

        self.triDlg.label_2 = QtWidgets.QLabel(self)
        self.triDlg.label_2.setText('Point_1 : ')
        self.triDlg.grid.addWidget(self.triDlg.label_2,1,0,1,2)

        self.triDlg.label_3 = QtWidgets.QLabel(self)
        self.triDlg.label_3.setText('\tx = ')
        self.triDlg.grid.addWidget(self.triDlg.label_3, 2, 0, 1, 1)

        self.x1_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.x1_tri,2,1,1,2)

        self.triDlg.label_4 = QtWidgets.QLabel(self)
        self.triDlg.label_4.setText('y = ')
        self.triDlg.grid.addWidget(self.triDlg.label_4,2,4,1,1)

        self.y1_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.y1_tri,2,5,1,2)

        self.triDlg.label_5 = QtWidgets.QLabel(self)
        self.triDlg.label_5.setText(' ')
        self.triDlg.grid.addWidget(self.triDlg.label_5,3,0,1,1)

        self.triDlg.label_6 = QtWidgets.QLabel(self)
        self.triDlg.label_6.setText('Point_2 : ')
        self.triDlg.grid.addWidget(self.triDlg.label_6,4,0,1,2)

        self.triDlg.label_7 = QtWidgets.QLabel(self)
        self.triDlg.label_7.setText('\tx = ')
        self.triDlg.grid.addWidget(self.triDlg.label_7, 5, 0, 1, 1)

        self.x2_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.x2_tri,5,1,1,2)

        self.triDlg.label_8 = QtWidgets.QLabel(self)
        self.triDlg.label_8.setText('y = ')
        self.triDlg.grid.addWidget(self.triDlg.label_8,5,4,1,1)

        self.y2_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.y2_tri,5,5,1,2)

        self.triDlg.label_9 = QtWidgets.QLabel(self)
        self.triDlg.label_9.setText(' ')
        self.triDlg.grid.addWidget(self.triDlg.label_9,6,0,1,1)

        self.triDlg.label_10 = QtWidgets.QLabel(self)
        self.triDlg.label_10.setText('Point_3 : ')
        self.triDlg.grid.addWidget(self.triDlg.label_10,7,0,1,2)

        self.triDlg.label_11 = QtWidgets.QLabel(self)
        self.triDlg.label_11.setText('\tx = ')
        self.triDlg.grid.addWidget(self.triDlg.label_11, 8, 0, 1, 1)

        self.x3_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.x3_tri,8,1,1,2)

        self.triDlg.label_12 = QtWidgets.QLabel(self)
        self.triDlg.label_12.setText('y = ')
        self.triDlg.grid.addWidget(self.triDlg.label_12,8,4,1,1)

        self.y3_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.y3_tri,8,5,1,2)

        self.triDlg.label_13 = QtWidgets.QLabel(self)
        self.triDlg.label_13.setText('')
        self.triDlg.grid.addWidget(self.triDlg.label_13,9,0,1,1)

        self.triDlg.label_14 = QtWidgets.QLabel(self)
        self.triDlg.label_14.setText('Color of Border Line (B,G,R) : ')
        self.triDlg.grid.addWidget(self.triDlg.label_14, 10, 0, 1, 3)

        self.triDlg.label_15 = QtWidgets.QLabel(self)
        self.triDlg.label_15.setText('Intensity of  Blue  (max 255) = ')
        self.triDlg.grid.addWidget(self.triDlg.label_15, 11, 0, 1, 2)

        self.blue_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.blue_tri, 11, 2, 1, 2)

        self.triDlg.label_16 = QtWidgets.QLabel(self)
        self.triDlg.label_16.setText('Intensity of Green (max 255) = ')
        self.triDlg.grid.addWidget(self.triDlg.label_16, 12, 0, 1, 2)

        self.green_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.green_tri, 12, 2, 1, 2)

        self.triDlg.label_17 = QtWidgets.QLabel(self)
        self.triDlg.label_17.setText('Intensity of   Red   (max 255) = ')
        self.triDlg.grid.addWidget(self.triDlg.label_17, 13, 0, 1, 2)

        self.red_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.red_tri, 13, 2, 1, 2)

        self.triDlg.label_18 = QtWidgets.QLabel(self)
        self.triDlg.label_18.setText(' ')
        self.triDlg.grid.addWidget(self.triDlg.label_18, 14, 0, 1, 1)

        self.triDlg.label_19 = QtWidgets.QLabel(self)
        self.triDlg.label_19.setText('Thickness of Border Line = ')
        self.triDlg.grid.addWidget(self.triDlg.label_19, 15, 0, 1, 2)

        self.thick_tri = QtWidgets.QLineEdit(self)
        self.triDlg.grid.addWidget(self.thick_tri,15,2,1,2)

        self.triDlg.label_20 = QtWidgets.QLabel(self)
        self.triDlg.label_20.setText('(-1 to fill triangle with (B,G,R) color) ')
        self.triDlg.grid.addWidget(self.triDlg.label_20, 16, 2, 1, 5)

        self.triDlg.label_21 = QtWidgets.QLabel(self)
        self.triDlg.label_21.setText(' ')
        self.triDlg.grid.addWidget(self.triDlg.label_21,17,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawTri)
        btn.resize(btn.sizeHint())
        self.triDlg.grid.addWidget(btn,18,0,1,8)

        self.triDlg.show()

    def drawTri(self):
        self.triDlg.close()

        self.pts = np.array([[int(self.x1_tri.text()),int(self.y1_tri.text())],
                             [int(self.x2_tri.text()),int(self.y2_tri.text())],
                             [int(self.x3_tri.text()),int(self.y3_tri.text())]],np.int32)
        cv2.polylines(self.image, [self.pts], True,
                      (int(self.blue_tri.text()),int(self.green_tri.text()),int(self.red_tri.text())),
                      int(self.thick_tri.text()))
        # cv2.imshow('draw triangle', self.image)

        img_tri = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                          QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_tri)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def quadDialog(self):
        self.pDlg.close()

        self.quadDlg = QMainWindow(self)
        self.quadDlg.setWindowTitle('4-Sided Polygon')

        self.quadDlg.central_widget = QtWidgets.QWidget()
        self.quadDlg.setCentralWidget(self.quadDlg.central_widget)
        self.quadDlg.grid = QtWidgets.QGridLayout(self.quadDlg.central_widget)

        self.quadDlg.label = QtWidgets.QLabel(self)
        self.quadDlg.label.setStyleSheet("font : 20pt")
        self.quadDlg.label.setText('Draw a Quadrilateral on Image')
        self.quadDlg.grid.addWidget(self.quadDlg.label,0,0,1,8)

        self.quadDlg.label_2 = QtWidgets.QLabel(self)
        self.quadDlg.label_2.setText('Point_1 : ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_2, 1, 0, 1, 2)

        self.quadDlg.label_3 = QtWidgets.QLabel(self)
        self.quadDlg.label_3.setText('\tx = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_3, 2, 0, 1, 1)

        self.x1_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.x1_quad, 2, 1, 1, 2)

        self.quadDlg.label_4 = QtWidgets.QLabel(self)
        self.quadDlg.label_4.setText('y = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_4, 2, 4, 1, 1)

        self.y1_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.y1_quad, 2, 5, 1, 2)

        self.quadDlg.label_5 = QtWidgets.QLabel(self)
        self.quadDlg.label_5.setText(' ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_5, 3, 0, 1, 1)

        self.quadDlg.label_6 = QtWidgets.QLabel(self)
        self.quadDlg.label_6.setText('Point_2 : ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_6, 4, 0, 1, 2)

        self.quadDlg.label_7 = QtWidgets.QLabel(self)
        self.quadDlg.label_7.setText('\tx = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_7, 5, 0, 1, 1)

        self.x2_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.x2_quad, 5, 1, 1, 2)

        self.quadDlg.label_8 = QtWidgets.QLabel(self)
        self.quadDlg.label_8.setText('y = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_8, 5, 4, 1, 1)

        self.y2_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.y2_quad, 5, 5, 1, 2)

        self.quadDlg.label_9 = QtWidgets.QLabel(self)
        self.quadDlg.label_9.setText(' ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_9, 6, 0, 1, 1)

        self.quadDlg.label_10 = QtWidgets.QLabel(self)
        self.quadDlg.label_10.setText('Point_3 : ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_10, 7, 0, 1, 2)

        self.quadDlg.label_11 = QtWidgets.QLabel(self)
        self.quadDlg.label_11.setText('\tx = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_11, 8, 0, 1, 1)

        self.x3_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.x3_quad, 8, 1, 1, 2)

        self.quadDlg.label_12 = QtWidgets.QLabel(self)
        self.quadDlg.label_12.setText('y = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_12, 8, 4, 1, 1)

        self.y3_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.y3_quad, 8, 5, 1, 2)

        self.quadDlg.label_13 = QtWidgets.QLabel(self)
        self.quadDlg.label_13.setText('')
        self.quadDlg.grid.addWidget(self.quadDlg.label_13, 9, 0, 1, 1)

        self.quadDlg.label_14 = QtWidgets.QLabel(self)
        self.quadDlg.label_14.setText('Point_4 : ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_14, 10, 0, 1, 2)

        self.quadDlg.label_15 = QtWidgets.QLabel(self)
        self.quadDlg.label_15.setText('\tx = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_15, 11, 0, 1, 1)

        self.x4_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.x4_quad, 11, 1, 1, 2)

        self.quadDlg.label_16 = QtWidgets.QLabel(self)
        self.quadDlg.label_16.setText('y = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_16, 11, 4, 1, 1)

        self.y4_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.y4_quad, 11, 5, 1, 2)

        self.quadDlg.label_17 = QtWidgets.QLabel(self)
        self.quadDlg.label_17.setText('')
        self.quadDlg.grid.addWidget(self.quadDlg.label_17, 12, 0, 1, 1)

        self.quadDlg.label_18 = QtWidgets.QLabel(self)
        self.quadDlg.label_18.setText('Color of Border Line (B,G,R) : ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_18, 13, 0, 1, 3)

        self.quadDlg.label_19 = QtWidgets.QLabel(self)
        self.quadDlg.label_19.setText('Intensity of  Blue  (max 255) = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_19, 14, 0, 1, 2)

        self.blue_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.blue_quad, 14, 2, 1, 2)

        self.quadDlg.label_20 = QtWidgets.QLabel(self)
        self.quadDlg.label_20.setText('Intensity of Green (max 255) = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_20, 15, 0, 1, 2)

        self.green_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.green_quad, 15, 2, 1, 2)

        self.quadDlg.label_21 = QtWidgets.QLabel(self)
        self.quadDlg.label_21.setText('Intensity of   Red   (max 255) = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_21, 16, 0, 1, 2)

        self.red_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.red_quad, 16, 2, 1, 2)

        self.quadDlg.label_22 = QtWidgets.QLabel(self)
        self.quadDlg.label_22.setText(' ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_22, 17, 0, 1, 1)

        self.quadDlg.label_23 = QtWidgets.QLabel(self)
        self.quadDlg.label_23.setText('Thickness of Border Line = ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_23, 18, 0, 1, 2)

        self.thick_quad = QtWidgets.QLineEdit(self)
        self.quadDlg.grid.addWidget(self.thick_quad, 18, 2, 1, 2)

        self.quadDlg.label_24 = QtWidgets.QLabel(self)
        self.quadDlg.label_24.setText('(-1 to fill quadrilateral with (B,G,R) color) ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_24, 19, 2, 1, 5)

        self.quadDlg.label_25 = QtWidgets.QLabel(self)
        self.quadDlg.label_25.setText(' ')
        self.quadDlg.grid.addWidget(self.quadDlg.label_25, 20, 0, 1, 1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawQuad)
        btn.resize(btn.sizeHint())
        self.quadDlg.grid.addWidget(btn, 21, 0, 1, 8)

        self.quadDlg.show()

    def drawQuad(self):
        self.quadDlg.close()

        self.pts = np.array([[int(self.x1_quad.text()),int(self.y1_quad.text())],
                             [int(self.x2_quad.text()),int(self.y2_quad.text())],
                             [int(self.x3_quad.text()),int(self.y3_quad.text())],
                             [int(self.x4_quad.text()),int(self.y4_quad.text())]],np.int32)
        cv2.polylines(self.image, [self.pts], True,
                      (int(self.blue_quad.text()),int(self.green_quad.text()),int(self.red_quad.text())),
                      int(self.thick_quad.text()))
        # cv2.imshow('draw quadrilateral', self.image)

        img_quad = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                          QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_quad)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def txtDialog(self):
        self.txtDlg = QMainWindow(self)
        self.txtDlg.setWindowTitle('Text')

        self.txtDlg.central_widget = QtWidgets.QWidget()
        self.txtDlg.setCentralWidget(self.txtDlg.central_widget)
        self.txtDlg.grid = QtWidgets.QGridLayout(self.txtDlg.central_widget)

        self.txtDlg.label = QtWidgets.QLabel(self)
        self.txtDlg.label.setStyleSheet("font : 20pt")
        self.txtDlg.label.setText('Print Text on Image')
        self.txtDlg.grid.addWidget(self.txtDlg.label,0,0,1,7)

        self.txtDlg.label_2 = QtWidgets.QLabel(self)
        self.txtDlg.label_2.setText('Text String = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_2,1,0,1,1)

        self.string = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.string,1,1,1,6)

        self.txtDlg.label_3 = QtWidgets.QLabel(self)
        self.txtDlg.label_3.setText('')
        self.txtDlg.grid.addWidget(self.txtDlg.label_3,2,0,1,1)

        self.txtDlg.label_4 = QtWidgets.QLabel(self)
        self.txtDlg.label_4.setText('Text String Coordinate : ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_4,3,0,1,3)

        self.txtDlg.label_5 = QtWidgets.QLabel(self)
        self.txtDlg.label_5.setText('\tx = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_5,4,0,1,1)

        self.x_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.x_txt,4,1,1,2)

        self.txtDlg.label_6 = QtWidgets.QLabel(self)
        self.txtDlg.label_6.setText('\ty = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_6,4,4,1,1)

        self.y_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.y_txt,4,5,1,2)

        self.txtDlg.label_7 = QtWidgets.QLabel(self)
        self.txtDlg.label_7.setText(' ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_7,5,0,1,1)

        self.txtDlg.label_8 = QtWidgets.QLabel(self)
        self.txtDlg.label_8.setText('Font Type : ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_8, 6, 0, 1, 1)

        self.txtDlg.comboBox = QtWidgets.QComboBox(self)
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_SIMPLEX")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_PLAIN")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_DUPLEX")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_COMPLEX")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_TRIPLEX")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_COMPLEX_SMALL")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_SCRIPT_SIMPLEX")
        self.txtDlg.comboBox.addItem("FONT_HERSHEY_SCRIPT_COMPLEX")
        self.txtDlg.comboBox.activated[str].connect(self.fontType)
        self.txtDlg.grid.addWidget(self.txtDlg.comboBox,6,1,1,4)

        self.txtDlg.label_9 = QtWidgets.QLabel(self)
        self.txtDlg.label_9.setText('Font Scale = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_9, 7, 0, 1, 1)

        self.fs_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.fs_txt,7,1,1,2)

        self.txtDlg.label_10 = QtWidgets.QLabel(self)
        self.txtDlg.label_10.setText('')
        self.txtDlg.grid.addWidget(self.txtDlg.label_10,8,0,1,1)

        self.txtDlg.label_11 = QtWidgets.QLabel(self)
        self.txtDlg.label_11.setText('Color of Text String (B,G,R) : ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_11, 9, 0, 1, 3)

        self.txtDlg.label_12 = QtWidgets.QLabel(self)
        self.txtDlg.label_12.setText('Intensity of  Blue  (max 255) = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_12, 10, 0, 1, 3)

        self.blue_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.blue_txt, 10, 3, 1, 2)

        self.txtDlg.label_13 = QtWidgets.QLabel(self)
        self.txtDlg.label_13.setText('Intensity of Green (max 255) = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_13, 11, 0, 1, 3)

        self.green_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.green_txt, 11, 3, 1, 2)

        self.txtDlg.label_14 = QtWidgets.QLabel(self)
        self.txtDlg.label_14.setText('Intensity of   Red   (max 255) = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_14, 12, 0, 1, 3)

        self.red_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.red_txt, 12, 3, 1, 2)

        self.txtDlg.label_15 = QtWidgets.QLabel(self)
        self.txtDlg.label_15.setText(' ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_15,13,0,1,1)

        self.txtDlg.label_16 = QtWidgets.QLabel(self)
        self.txtDlg.label_16.setText('Thickness of Line = ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_16,14,0,1,2)

        self.thick_txt = QtWidgets.QLineEdit(self)
        self.txtDlg.grid.addWidget(self.thick_txt,14,2,1,2)

        self.txtDlg.label_17 = QtWidgets.QLabel(self)
        self.txtDlg.label_17.setText(' ')
        self.txtDlg.grid.addWidget(self.txtDlg.label_17,15,0,1,1)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.printText)
        btn.resize(btn.sizeHint())
        self.txtDlg.grid.addWidget(btn,16,0,1,7)

        self.txtDlg.show()

    def fontType(self,text):
        if str(text)=="FONT_HERSHEY_SIMPLEX":
            self.font=cv2.FONT_HERSHEY_SIMPLEX
        elif str(text) == "FONT_HERSHEY_PLAIN":
            self.font = cv2.FONT_HERSHEY_PLAIN
        elif str(text) == "FONT_HERSHEY_DUPLEX":
            self.font=cv2.FONT_HERSHEY_DUPLEX
        elif str(text)=="FONT_HERSHEY_COMPLEX":
            self.font=cv2.FONT_HERSHEY_COMPLEX
        elif str(text)=="FONT_HERSHEY_TRIPLEX":
            self.font=cv2.FONT_HERSHEY_TRIPLEX
        elif str(text)=="FONT_HERSHEY_COMPLEX_SMALL":
            self.font=cv2.FONT_HERSHEY_COMPLEX_SMALL
        elif str(text)=="FONT_HERSHEY_SCRIPT_SIMPLEX":
            self.font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        elif str(text)=="FONT_HERSHEY_SCRIPT_COMPLEX":
            self.font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    def printText(self):
        self.txtDlg.close()

        cv2.putText(self.image, str(self.string.text()), (int(self.x_txt.text()),int(self.y_txt.text())),
                    self.font, int(self.fs_txt.text()),
                    (int(self.blue_txt.text()),int(self.green_txt.text()),int(self.red_txt.text())),
                    int(self.thick_txt.text()))
        # cv2.imshow('print string', self.image)

        img_txt = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                          QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img_txt)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

def main():
    app = QApplication(sys.argv)
    myWindow = Win()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()