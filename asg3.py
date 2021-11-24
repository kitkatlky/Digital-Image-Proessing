import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, qApp,  QMenu, QApplication, QMessageBox
from PyQt5.QtGui import QIcon, QPixmap, QImage, QTransform
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Win(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.resize(800, 600)
        self.setFixedSize(800,600)
        self.move(300, 50)
        self.setWindowTitle('Editor')

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
        rgbAct.setStatusTip('Extracting the RGB values of a pixel.')
        rgbAct.triggered.connect(self.rgbDialog)

        self.split = QMenu('Split',self)
        self.splitMenu()

        self.merge = QMenu('&Collage', self)
        self.mergeMenu()

        self.hist = QMenu('&Histogram', self)
        self.histMenu()

        self.blur = QMenu('&Blur',self)
        self.blurMenu()

        contrastAct = QAction('&Contrast',self)
        contrastAct.setStatusTip('Adjust contrast level of image.')
        contrastAct.triggered.connect(self.contrastDialog)

        sharpAct = QAction('&Sharpening',self)
        sharpAct.setStatusTip('Sharpening image.')
        sharpAct.triggered.connect(self.sharpDialog)

        sobelAct = QAction('&Sobel', self)
        sobelAct.setStatusTip('Detect the edges of an image using Sobel edge detector.')
        sobelAct.triggered.connect(self.sobelDialog)

        prewittAct = QAction('&Prewitt', self)
        prewittAct.setStatusTip('Detect the edges of an image using Prewitt edge detector.')
        prewittAct.triggered.connect(self.prewittDialog)

        robertsAct = QAction('&Roberts cross', self)
        robertsAct.setStatusTip('Detect the edges of an image using Roberts cross edge detector.')
        robertsAct.triggered.connect(self.robertsDialog)

        laplacianAct = QAction('&Laplacian', self)
        laplacianAct.setStatusTip('Detect the edges of an image using Laplacian edge detector.')
        laplacianAct.triggered.connect(self.laplacianDialog)

        cannyAct = QAction('&Canny', self)
        cannyAct.setStatusTip('Detect the edges of an image using Canny edge detector.')
        cannyAct.triggered.connect(self.cannyDialog)

        threshAct = QAction('&Thresholding', self)
        threshAct.setStatusTip('Image segmentation by using threshold techniques.')
        threshAct.triggered.connect(self.threshDialog)

        colSegAct = QAction('&Color Segmentation', self)
        colSegAct.setStatusTip('Image segmentation by detecting color space.')
        colSegAct.triggered.connect(self.colSegDialog)

        clusterAct = QAction('&K-Means Clustering', self)
        clusterAct.setStatusTip('Image segmentation by using K-Means clustering.')
        clusterAct.triggered.connect(self.kMeansDialog)

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

        func2Menu = menubar.addMenu('&Function_2')
        func2Menu.addMenu(self.split)
        func2Menu.addMenu(self.merge)
        func2Menu.addMenu(self.hist)

        filterMenu = menubar.addMenu('&Filter')
        filterMenu.addMenu(self.blur)
        filterMenu.addAction(contrastAct)
        filterMenu.addAction(sharpAct)

        edgeMenu = menubar.addMenu('&Edge Detection')
        edgeMenu.addAction(sobelAct)
        edgeMenu.addAction(prewittAct)
        edgeMenu.addAction(robertsAct)
        edgeMenu.addAction(laplacianAct)
        edgeMenu.addAction(cannyAct)

        segMenu = menubar.addMenu('Segmentation Technique')
        segMenu.addAction(threshAct)
        segMenu.addAction(colSegAct)
        segMenu.addAction(clusterAct)

    def colorMenu(self):
        act1 = QAction('Gray', self)
        act1.setStatusTip('Convert image color to gray scale')
        act1.triggered.connect(self.showGray)
        act2 = QAction('BGRA', self)
        act2.setStatusTip('Convert image color to BGRA')
        act2.triggered.connect(self.showBGRA)
        act3 = QAction('HSV', self)
        act3.setStatusTip('Convert image color to HSV')
        act3.triggered.connect(self.showHSV)
        self.colMenu.addAction(act1)
        self.colMenu.addAction(act2)
        self.colMenu.addAction(act3)

    def splitMenu(self):
        r2c1 = QAction(QIcon('1x2.png'),'1 x 2',self)
        r2c1.setStatusTip('Split image to two equal parts horizontally')
        r2c1.triggered.connect(self.split_1x2)
        self.split.addAction(r2c1)

        r3c1 = QAction(QIcon('1x3.png'), '1 x 3', self)
        r3c1.setStatusTip('Split image to three equal parts horizontally')
        r3c1.triggered.connect(self.split_1x3)
        self.split.addAction(r3c1)

        r1c2 = QAction(QIcon('2x1.png'), '2 x 1', self)
        r1c2.setStatusTip('Split image to two equal parts vertically')
        r1c2.triggered.connect(self.split_2x1)
        self.split.addAction(r1c2)

        r2c2 = QAction(QIcon('2x2.png'), '2 x 2', self)
        r2c2.setStatusTip('Split image to four equal parts')
        r2c2.triggered.connect(self.split_2x2)
        self.split.addAction(r2c2)

        r3c2 = QAction(QIcon('2x3.png'), '2 x 3', self)
        r3c2.setStatusTip('Split image to six equal parts in 2x3 format')
        r3c2.triggered.connect(self.split_2x3)
        self.split.addAction(r3c2)

        r1c3 = QAction(QIcon('3x1.png'), '3 x 1', self)
        r1c3.setStatusTip('Split image to three equal parts vertically')
        r1c3.triggered.connect(self.split_3x1)
        self.split.addAction(r1c3)

        r2c3 = QAction(QIcon('3x2.png'), '3 x 2', self)
        r2c3.setStatusTip('Split image to six equal parts in 3x2 format')
        r2c3.triggered.connect(self.split_3x2)
        self.split.addAction(r2c3)

        r3c3 = QAction(QIcon('3x3.png'), '3 x 3', self)
        r3c3.setStatusTip('Split image to nine equal parts')
        r3c3.triggered.connect(self.split_3x3)
        self.split.addAction(r3c3)

    def mergeMenu(self):
        r2c1 = QAction(QIcon('1x2.png'),'1 x 2',self)
        r2c1.setStatusTip('Merge two images horizontally')
        r2c1.triggered.connect(self.merge_1x2)
        self.merge.addAction(r2c1)

        r3c1 = QAction(QIcon('1x3.png'), '1 x 3', self)
        r3c1.setStatusTip('Merge three images horizontally')
        r3c1.triggered.connect(self.merge_1x3)
        self.merge.addAction(r3c1)

        r1c2 = QAction(QIcon('2x1.png'), '2 x 1', self)
        r1c2.setStatusTip('Merge two images vertically')
        r1c2.triggered.connect(self.merge_2x1)
        self.merge.addAction(r1c2)

        r2c2 = QAction(QIcon('2x2.png'), '2 x 2', self)
        r2c2.setStatusTip('Merge four images in 2x2 format')
        r2c2.triggered.connect(self.merge_2x2)
        self.merge.addAction(r2c2)

        r3c2 = QAction(QIcon('2x3.png'), '2 x 3', self)
        r3c2.setStatusTip('Merge six images in 2x3 format')
        r3c2.triggered.connect(self.merge_2x3)
        self.merge.addAction(r3c2)

        r1c3 = QAction(QIcon('3x1.png'), '3 x 1', self)
        r1c3.setStatusTip('Merge three images vertically')
        r1c3.triggered.connect(self.merge_3x1)
        self.merge.addAction(r1c3)

        r2c3 = QAction(QIcon('3x2.png'), '3 x 2', self)
        r2c3.setStatusTip('Merge nine images')
        r2c3.triggered.connect(self.merge_3x2)
        self.merge.addAction(r2c3)

        r3c3 = QAction(QIcon('3x3.png'), '3 x 3', self)
        r3c3.setStatusTip('Merge nine images')
        r3c3.triggered.connect(self.merge_3x3)
        self.merge.addAction(r3c3)

    def histMenu(self):
        grayH = QAction('Gray scale image',self)
        grayH.setStatusTip('Histogram equalization for gray scale image.')
        grayH.triggered.connect(self.gHistDialog)
        self.hist.addAction(grayH)

        colorH = QAction('Color image',self)
        colorH.setStatusTip('Histogram equalization for color image.')
        colorH.triggered.connect(self.colorHistDialog)
        self.hist.addAction(colorH)

    def blurMenu(self):
        blur_3 = QAction('3 x 3 kernel',self)
        blur_3.setStatusTip('Image apply blurring filter by using 3x3 kernel.')
        blur_3.triggered.connect(self.kernelSize_3)
        self.blur.addAction(blur_3)

        blur_5 = QAction('5 x 5 kernel', self)
        blur_5.setStatusTip('Image apply blurring filter by using 5x5 kernel.')
        blur_5.triggered.connect(self.kernelSize_5)
        self.blur.addAction(blur_5)

        blur_7 = QAction('7 x 7 kernel', self)
        blur_7.setStatusTip('Image apply blurring filter by using 7x7 kernel.')
        blur_7.triggered.connect(self.kernelSize_7)
        self.blur.addAction(blur_7)

        blur_c = QAction('Customize', self)
        blur_c.setStatusTip('Cuztomize size of kernel that used for image blurring filter .')
        blur_c.triggered.connect(self.kernelSize_c)
        self.blur.addAction(blur_c)

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
        self.name='white.jpg'
        self.image = cv2.imread(self.name)
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

        self.updateDetail()

    def keepRatio(self):
        if self.imgHeight>500 and self.imgWidth<=800 :
            self.pixmap = self.pixmap.scaled(self.imgWidth, 500, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight<=500 and self.imgWidth>800 :
            self.pixmap = self.pixmap.scaled(800, self.imgHeight, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight > 500 and self.imgWidth > 800:
            self.pixmap = self.pixmap.scaled(800, 500, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight <= 500 and self.imgWidth <= 800:
            self.pixmap = self.pixmap.scaled(self.imgWidth, self.imgHeight, QtCore.Qt.KeepAspectRatio)

    def updateDetail(self):
        self.label_2.setText('Image Width: ' + str(self.imgWidth) + ' pixels \t Image Height : ' +
                             str(self.imgHeight) + ' pixels')

    def saveImg(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\',
                                                              "Image Files (*.jpg);;Image Files (*.tiff);;Image Files (*.bmp)")
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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomCrop)
        btn.resize(btn.sizeHint())
        self.Cdlg.grid.addWidget(btn, 8, 0, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.cropImg)
        btn1.resize(btn1.sizeHint())
        self.Cdlg.grid.addWidget(btn1,8,3,1,2)

        self.Cdlg.show()

    def randomCrop(self):
        self.start_x.setText(str(random.randint(0,int(self.imgWidth/4))))
        self.start_y.setText(str(random.randint(0,int(self.imgHeight/4))))
        self.end_x.setText(str(random.randint(int(self.imgWidth/2),int(self.imgWidth))))
        self.end_y.setText(str(random.randint(int(self.imgHeight/2),int(self.imgHeight))))

    def cropImg(self):
        self.Cdlg.close()
        startCol=int(self.start_x.text())
        startRow=int(self.start_y.text())
        endCol=int(self.end_x.text())
        endRow=int(self.end_y.text())

        self.image = self.image[startRow:endRow, startCol:endCol]
        # cv2.imshow('Croped image', self.image)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        crop_img = QImage(self.image, self.imgWidth, self.imgHeight,QImage.Format_ARGB32 )
        self.pixmap=QPixmap.scaled(QPixmap.fromImage(crop_img), self.imgWidth,self.imgHeight,QtCore.Qt.KeepAspectRatio,
                                                     QtCore.Qt.SmoothTransformation)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

        self.updateDetail()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomResize)
        btn.resize(btn.sizeHint())
        self.RIdlg.grid.addWidget(btn, 5, 0, 1, 4)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.resizeImg)
        btn1.resize(btn1.sizeHint())
        self.RIdlg.grid.addWidget(btn1,6,0,1,4)

        self.RIdlg.show()

    def randomResize(self):
        self.widthInput.setText(str(random.randint(int(self.imgWidth/2),self.imgWidth*2)))
        self.heightInput.setText(str(random.randint(int(self.imgHeight/2), self.imgHeight*2)))

    def resizeImg(self):
        self.RIdlg.close()
        self.imgHeight = int(self.heightInput.text())
        self.imgWidth = int(self.widthInput.text())

        dimension = (self.imgWidth,self.imgHeight)
        self.image = cv2.resize(self.image, dimension)
        # cv2.imshow('resize',self.image)

        self.keepRatio()
        self.label.setPixmap(self.pixmap)
        self.updateDetail()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomRotate)
        btn.resize(btn.sizeHint())
        self.RotDlg.grid.addWidget(btn, 5, 0, 1, 3)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.rotateImg)
        btn1.resize(btn1.sizeHint())
        self.RotDlg.grid.addWidget(btn1,5,3,1,3)

        self.RotDlg.show()

    def randomRotate(self):
        self.degreeInput.setText(str(random.randint(0,360)))

    def rotateImg(self):
        self.RotDlg.close()
        angle=int(self.degreeInput.text())
        scale=1.0
        self.center=(self.image.shape[1]/2,self.image.shape[0]/2)
        M = cv2.getRotationMatrix2D(self.center, -angle, scale)
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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomTranslate)
        btn.resize(btn.sizeHint())
        self.TDlg.grid.addWidget(btn, 8, 0, 1, 4)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.translateImg)
        btn.resize(btn.sizeHint())
        self.TDlg.grid.addWidget(btn,9,0,1,4)

        self.TDlg.show()

    def randomTranslate(self):
        self.txInput.setText(str(random.randint(0,int(self.imgWidth/2))))
        self.tyInput.setText(str(random.randint(0,int(self.imgHeight/2))))

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

    def showGray(self):
        self.sgDlg = QMainWindow(self)
        self.sgDlg.resize(200, 200)
        self.sgDlg.setWindowTitle('Gray Scale')

        self.sgDlg.central_widget = QtWidgets.QWidget()
        self.sgDlg.setCentralWidget(self.sgDlg.central_widget)
        self.sgDlg.vbox = QtWidgets.QVBoxLayout(self.sgDlg.central_widget)

        self.sgDlg.label = QtWidgets.QLabel(self)
        self.sgDlg.label.setStyleSheet("font : 20pt")
        self.sgDlg.label.setText('Preview Gray Scale Color Image')
        self.sgDlg.vbox.addWidget(self.sgDlg.label)

        self.sgDlg.label_2=QtWidgets.QLabel(self)
        image=cv2.cvtColor(self.image,cv2.COLOR_BGRA2GRAY)
        gray_image = QImage(image, self.image.shape[1], self.image.shape[0], self.image.shape[1],
                            QImage.Format_Grayscale8)
        pixmap = QPixmap(gray_image)
        pixmap = pixmap.scaled(300,300,QtCore.Qt.KeepAspectRatio)
        self.sgDlg.label_2.setPixmap(pixmap)
        self.sgDlg.vbox.addWidget(self.sgDlg.label_2)

        self.sgDlg.label_3 = QtWidgets.QLabel(self)
        self.sgDlg.label_3.setText('')
        self.sgDlg.vbox.addWidget(self.sgDlg.label_3)

        btn = QtWidgets.QPushButton('Cancel', self)
        btn.clicked.connect(self.closeGray)
        btn.resize(btn.sizeHint())
        self.sgDlg.vbox.addWidget(btn)

        btn1 = QtWidgets.QPushButton('Apply', self)
        btn1.clicked.connect(self.convertGray)
        btn1.resize(btn1.sizeHint())
        self.sgDlg.vbox.addWidget(btn1)

        self.sgDlg.show()

    def closeGray(self):
        self.sgDlg.close()

    def convertGray(self):
        self.sgDlg.close()
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGRA2GRAY)
        # cv2.imshow("Gray Scale color image",self.image)
        self.updateGray()

    def updateGray(self):
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1],QImage.Format_Grayscale8)
        self.pixmap = QPixmap(img)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def showBGRA(self):
        self.bgraDlg = QMainWindow(self)
        self.bgraDlg.resize(200, 200)
        self.bgraDlg.setWindowTitle('BGRA')

        self.bgraDlg.central_widget = QtWidgets.QWidget()
        self.bgraDlg.setCentralWidget(self.bgraDlg.central_widget)
        self.bgraDlg.vbox = QtWidgets.QVBoxLayout(self.bgraDlg.central_widget)

        self.bgraDlg.label = QtWidgets.QLabel(self)
        self.bgraDlg.label.setStyleSheet("font : 20pt")
        self.bgraDlg.label.setText('Preview BGRA Color Image')
        self.bgraDlg.vbox.addWidget(self.bgraDlg.label)

        self.bgraDlg.label_2=QtWidgets.QLabel(self)
        image=cv2.cvtColor(self.image,cv2.COLOR_BGR2BGRA)
        color_image = QImage(image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*4,
                            QImage.Format_ARGB32)
        pixmap = QPixmap(color_image)
        pixmap = pixmap.scaled(300,300,QtCore.Qt.KeepAspectRatio)
        self.bgraDlg.label_2.setPixmap(pixmap)
        self.bgraDlg.vbox.addWidget(self.bgraDlg.label_2)

        self.bgraDlg.label_3 = QtWidgets.QLabel(self)
        self.bgraDlg.label_3.setText('')
        self.bgraDlg.vbox.addWidget(self.bgraDlg.label_3)

        btn = QtWidgets.QPushButton('Cancel', self)
        btn.clicked.connect(self.closeBGRA)
        btn.resize(btn.sizeHint())
        self.bgraDlg.vbox.addWidget(btn)

        btn1 = QtWidgets.QPushButton('Apply', self)
        btn1.clicked.connect(self.convertBGRA)
        btn1.resize(btn1.sizeHint())
        self.bgraDlg.vbox.addWidget(btn1)

        self.bgraDlg.show()

    def closeBGRA(self):
        self.bgraDlg.close()

    def convertBGRA(self):
        self.bgraDlg.close()
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2BGRA)
        # cv2.imshow("Color image",self.image)

        color_image = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1]*4,
                            QImage.Format_ARGB32)
        self.pixmap = QPixmap(color_image)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def showHSV(self):
        self.hsvDlg = QMainWindow(self)
        self.hsvDlg.resize(200, 200)
        self.hsvDlg.setWindowTitle('HSV')

        self.hsvDlg.central_widget = QtWidgets.QWidget()
        self.hsvDlg.setCentralWidget(self.hsvDlg.central_widget)
        self.hsvDlg.vbox = QtWidgets.QVBoxLayout(self.hsvDlg.central_widget)

        self.hsvDlg.label = QtWidgets.QLabel(self)
        self.hsvDlg.label.setStyleSheet("font : 20pt")
        self.hsvDlg.label.setText('Preview HSV Color Image')
        self.hsvDlg.vbox.addWidget(self.hsvDlg.label)

        self.hsvDlg.label_2 = QtWidgets.QLabel(self)
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv_image = QImage(image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                           QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(hsv_image)
        pixmap = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        self.hsvDlg.label_2.setPixmap(pixmap)
        self.hsvDlg.vbox.addWidget(self.hsvDlg.label_2)

        self.hsvDlg.label_3 = QtWidgets.QLabel(self)
        self.hsvDlg.label_3.setText('')
        self.hsvDlg.vbox.addWidget(self.hsvDlg.label_3)

        btn = QtWidgets.QPushButton('Cancel', self)
        btn.clicked.connect(self.closeHSV)
        btn.resize(btn.sizeHint())
        self.hsvDlg.vbox.addWidget(btn)

        btn1 = QtWidgets.QPushButton('Apply', self)
        btn1.clicked.connect(self.convertHSV)
        btn1.resize(btn1.sizeHint())
        self.hsvDlg.vbox.addWidget(btn1)

        self.hsvDlg.show()

    def closeHSV(self):
        self.hsvDlg.close()

    def convertHSV(self):
        self.hsvDlg.close()
        self.image=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV image",self.image)
        self.updateColor()

    def updateColor(self):
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                     QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img)
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

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomRGB)
        btn_1.resize(btn_1.sizeHint())
        self.rgbDlg.grid.addWidget(btn_1, 5, 0, 1, 5)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.showRGB)
        btn_2.resize(btn_2.sizeHint())
        self.rgbDlg.grid.addWidget(btn_2, 6, 0, 1, 5)

        self.rgbDlg.show()

    def randomRGB(self):
        self.x_rgb.setText(str(random.randint(0,self.imgWidth-1)))
        self.y_rgb.setText(str(random.randint(0, self.imgHeight-1)))

    def showRGB(self):
        self.rgbDlg.close()

        self.showDlg = QMainWindow(self)
        title= "RGB Value of ["+str(self.x_rgb.text())+" , "+str(self.y_rgb.text())+"] "
        self.showDlg.setWindowTitle(title)

        self.showDlg.central_widget = QtWidgets.QWidget()
        self.showDlg.setCentralWidget(self.showDlg.central_widget)
        self.showDlg.vbox = QtWidgets.QVBoxLayout(self.showDlg.central_widget)

        y=int(self.y_rgb.text())
        x=int(self.x_rgb.text())

        self.showDlg.label = QtWidgets.QLabel(self)
        pixel=self.image[y:y+1,x:x+1]
        pixel = cv2.resize(pixel, (100,100))
        pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2BGRA)
        pixel_img = QImage(pixel, 100, 100, QImage.Format_ARGB32)
        pixmap = QPixmap.scaled(QPixmap.fromImage(pixel_img), 100, 100,QtCore.Qt.KeepAspectRatio,
                                QtCore.Qt.SmoothTransformation)
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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomCir)
        btn.resize(btn.sizeHint())
        self.circleDlg.grid.addWidget(btn, 15, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.drawCir)
        btn1.resize(btn1.sizeHint())
        self.circleDlg.grid.addWidget(btn1,15,5,1,2)

        self.circleDlg.show()

    def randomCir(self):
        self.x_circle.setText(str(random.randint(int(self.imgWidth/4),int(self.imgWidth / 4*3))))
        self.y_circle.setText(str(random.randint(int(self.imgHeight/4),int(self.imgHeight / 4*3))))
        self.r_circle.setText(str(random.randint(10, 100)))
        self.blue_circle.setText(str(random.randint(0,255)))
        self.green_circle.setText(str(random.randint(0, 255)))
        self.red_circle.setText(str(random.randint(0, 255)))
        self.thick_circle.setText(str(random.randint(1, 10)))

    def drawCir(self):
        self.circleDlg.close()

        cv2.circle(self.image, (int(self.x_circle.text()), int(self.y_circle.text())), int(self.r_circle.text()),
                   (int(self.blue_circle.text()), int(self.green_circle.text()),int(self.red_circle.text())),
                   int(self.thick_circle.text()))
        # cv2.imshow('draw circle', self.image)
        self.updateColor()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomRect)
        btn.resize(btn.sizeHint())
        self.rectDlg.grid.addWidget(btn, 15, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.drawRect)
        btn1.resize(btn1.sizeHint())
        self.rectDlg.grid.addWidget(btn1,15,5,1,2)

        self.rectDlg.show()

    def randomRect(self):
        self.x1_rect.setText(str(random.randint(0, int(self.imgWidth / 2))))
        self.y1_rect.setText(str(random.randint(0, int(self.imgHeight / 2))))
        self.x2_rect.setText(str(random.randint(int(self.imgWidth / 2),self.imgWidth)))
        self.y2_rect.setText(str(random.randint(int(self.imgHeight / 2),self.imgHeight)))
        self.blue_rect.setText(str(random.randint(0, 255)))
        self.green_rect.setText(str(random.randint(0, 255)))
        self.red_rect.setText(str(random.randint(0, 255)))
        self.thick_rect.setText(str(random.randint(1, 10)))

    def drawRect(self):
        self.rectDlg.close()

        cv2.rectangle(self.image, (int(self.x1_rect.text()), int(self.y1_rect.text())),
                      (int(self.x2_rect.text()),int(self.y2_rect.text())),
                      (int(self.blue_rect.text()),int(self.green_rect.text()),int(self.red_rect.text())),
                      int(self.thick_rect.text()))
        # cv2.imshow('draw rectangle', self.image)
        self.updateColor()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomEll)
        btn.resize(btn.sizeHint())
        self.ellDlg.grid.addWidget(btn, 19, 1, 1, 2)

        btn = QtWidgets.QPushButton('OK', self)
        btn.clicked.connect(self.drawEll)
        btn.resize(btn.sizeHint())
        self.ellDlg.grid.addWidget(btn,19,4,1,2)

        self.ellDlg.show()

    def randomEll(self):
        self.x_ell.setText(str(random.randint(int(self.imgWidth / 4),int(self.imgWidth / 4*3))))
        self.y_ell.setText(str(random.randint(int(self.imgHeight / 4),int(self.imgHeight / 4*3))))
        self.major_ell.setText(str(random.randint(10, int(self.imgWidth / 4))))
        self.minor_ell.setText(str(random.randint(10, int(self.imgHeight / 4))))
        self.angle_ell.setText(str(random.randint(0, 360)))
        self.startAng_ell.setText("0")
        self.endAng_ell.setText("360")
        self.blue_ell.setText(str(random.randint(0, 255)))
        self.green_ell.setText(str(random.randint(0, 255)))
        self.red_ell.setText(str(random.randint(0, 255)))
        self.thick_ell.setText(str(random.randint(1, 10)))

    def drawEll(self):
        self.ellDlg.close()

        cv2.ellipse(self.image, (int(self.x_ell.text()), int(self.y_ell.text())),
                    (int(self.major_ell.text()),int(self.minor_ell.text())), float(self.angle_ell.text()),
                    float(self.startAng_ell.text()), float(self.endAng_ell.text()),
                    (int(self.blue_ell.text()),int(self.green_ell.text()),int(self.red_ell.text())),
                    int(self.thick_ell.text()))
        # cv2.imshow('draw ellipse', self.image)
        self.updateColor()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomLine)
        btn.resize(btn.sizeHint())
        self.lDlg.grid.addWidget(btn, 14, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.drawLine)
        btn1.resize(btn1.sizeHint())
        self.lDlg.grid.addWidget(btn1,14,5,1,2)

        self.lDlg.show()

    def randomLine(self):
        self.x1_line.setText(str(random.randint(0, int(self.imgWidth / 4*3))))
        self.y1_line.setText(str(random.randint(0, int(self.imgHeight / 4*3))))
        self.x2_line.setText(str(random.randint(int(self.imgWidth / 4), self.imgWidth)))
        self.y2_line.setText(str(random.randint(int(self.imgHeight / 4), self.imgHeight)))
        self.blue_line.setText(str(random.randint(0, 255)))
        self.green_line.setText(str(random.randint(0, 255)))
        self.red_line.setText(str(random.randint(0, 255)))
        self.thick_line.setText(str(random.randint(1, 10)))

    def drawLine(self):
        self.lDlg.close()

        cv2.line(self.image, (int(self.x1_line.text()), int(self.y1_line.text())),
                 (int(self.x2_line.text()), int(self.y2_line.text())),
                 (int(self.blue_line.text()),int(self.green_line.text()),int(self.red_line.text())),
                 int(self.thick_line.text()))
        # cv2.imshow('draw line', self.image)
        self.updateColor()

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

        # btn_5 = QtWidgets.QPushButton('5 (Pentagon)', self)
        # btn_5.clicked.connect(self.penDialog)
        # btn_5.resize(btn_5.sizeHint())
        # self.pDlg.vbox.addWidget(btn_5)
        #
        # self.pDlg.label_4 = QtWidgets.QLabel(self)
        # self.pDlg.label_4.setText(' ')
        # self.pDlg.vbox.addWidget(self.pDlg.label_4)

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomTri)
        btn.resize(btn.sizeHint())
        self.triDlg.grid.addWidget(btn, 18, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.drawTri)
        btn1.resize(btn1.sizeHint())
        self.triDlg.grid.addWidget(btn1,18,5,1,2)

        self.triDlg.show()

    def randomTri(self):
        self.x1_tri.setText(str(random.randint(int(self.imgWidth / 4),int(self.imgWidth / 4 * 3))))
        self.y1_tri.setText(str(random.randint(int(self.imgHeight / 4),int(self.imgHeight / 4 * 3))))
        self.x2_tri.setText(str(random.randint(int(self.imgWidth / 4), int(self.imgWidth / 4 * 3))))
        self.y2_tri.setText(str(random.randint(int(self.imgHeight / 4), int(self.imgHeight / 4 *3))))
        self.x3_tri.setText(str(random.randint(int(self.imgWidth / 4), int(self.imgWidth / 4 * 3))))
        self.y3_tri.setText(str(random.randint(int(self.imgHeight / 4), int(self.imgHeight / 4 * 3))))
        self.blue_tri.setText(str(random.randint(0, 255)))
        self.green_tri.setText(str(random.randint(0, 255)))
        self.red_tri.setText(str(random.randint(0, 255)))
        self.thick_tri.setText(str(random.randint(1, 10)))

    def drawTri(self):
        self.triDlg.close()

        self.pts = np.array([[int(self.x1_tri.text()),int(self.y1_tri.text())],
                             [int(self.x2_tri.text()),int(self.y2_tri.text())],
                             [int(self.x3_tri.text()),int(self.y3_tri.text())]],np.int32)
        cv2.polylines(self.image, [self.pts], True,
                      (int(self.blue_tri.text()),int(self.green_tri.text()),int(self.red_tri.text())),
                      int(self.thick_tri.text()))
        # cv2.imshow('draw triangle', self.image)
        self.updateColor()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomQuad)
        btn.resize(btn.sizeHint())
        self.quadDlg.grid.addWidget(btn, 21, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.drawQuad)
        btn1.resize(btn1.sizeHint())
        self.quadDlg.grid.addWidget(btn1, 21, 5, 1, 2)

        self.quadDlg.show()

    def randomQuad(self):
        self.x1_quad.setText(str(random.randint(0, int(self.imgWidth / 2))))
        self.y1_quad.setText(str(random.randint(0, int(self.imgHeight / 2))))
        self.x2_quad.setText(str(random.randint(0, int(self.imgWidth / 2))))
        self.y2_quad.setText(str(random.randint(int(self.imgHeight / 2), int(self.imgHeight / 4 * 3))))
        self.x3_quad.setText(str(random.randint(int(self.imgWidth / 4 * 3), self.imgWidth)))
        self.y3_quad.setText(str(random.randint(int(self.imgHeight / 2), int(self.imgHeight / 4 * 3))))
        self.x4_quad.setText(str(random.randint(int(self.imgWidth / 4 * 3), self.imgWidth)))
        self.y4_quad.setText(str(random.randint(0, int(self.imgHeight / 2))))
        self.blue_quad.setText(str(random.randint(0, 255)))
        self.green_quad.setText(str(random.randint(0, 255)))
        self.red_quad.setText(str(random.randint(0, 255)))
        self.thick_quad.setText(str(random.randint(1, 10)))

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
        self.updateColor()

    # def penDialog(self):
    #     self.pDlg.close()
    #
    #     self.penDlg = QMainWindow(self)
    #     self.penDlg.setWindowTitle('5-Sided Polygon')
    #
    #     self.penDlg.central_widget = QtWidgets.QWidget()
    #     self.penDlg.setCentralWidget(self.penDlg.central_widget)
    #     self.penDlg.grid = QtWidgets.QGridLayout(self.penDlg.central_widget)
    #
    #     self.penDlg.label = QtWidgets.QLabel(self)
    #     self.penDlg.label.setStyleSheet("font : 20pt")
    #     self.penDlg.label.setText('Draw a Pentagon on Image')
    #     self.penDlg.grid.addWidget(self.penDlg.label,0,0,1,8)
    #
    #     self.penDlg.label_2 = QtWidgets.QLabel(self)
    #     self.penDlg.label_2.setText('Point_1 : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_2, 1, 0, 1, 2)
    #
    #     self.penDlg.label_3 = QtWidgets.QLabel(self)
    #     self.penDlg.label_3.setText('\tx = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_3, 2, 0, 1, 1)
    #
    #     self.x1_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.x1_pen, 2, 1, 1, 2)
    #
    #     self.penDlg.label_4 = QtWidgets.QLabel(self)
    #     self.penDlg.label_4.setText('y = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_4, 2, 4, 1, 1)
    #
    #     self.y1_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.y1_pen, 2, 5, 1, 2)
    #
    #     self.penDlg.label_5 = QtWidgets.QLabel(self)
    #     self.penDlg.label_5.setText(' ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_5, 3, 0, 1, 1)
    #
    #     self.penDlg.label_6 = QtWidgets.QLabel(self)
    #     self.penDlg.label_6.setText('Point_2 : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_6, 4, 0, 1, 2)
    #
    #     self.penDlg.label_7 = QtWidgets.QLabel(self)
    #     self.penDlg.label_7.setText('\tx = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_7, 5, 0, 1, 1)
    #
    #     self.x2_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.x2_pen, 5, 1, 1, 2)
    #
    #     self.penDlg.label_8 = QtWidgets.QLabel(self)
    #     self.penDlg.label_8.setText('y = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_8, 5, 4, 1, 1)
    #
    #     self.y2_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.y2_pen, 5, 5, 1, 2)
    #
    #     self.penDlg.label_9 = QtWidgets.QLabel(self)
    #     self.penDlg.label_9.setText(' ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_9, 6, 0, 1, 1)
    #
    #     self.penDlg.label_10 = QtWidgets.QLabel(self)
    #     self.penDlg.label_10.setText('Point_3 : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_10, 7, 0, 1, 2)
    #
    #     self.penDlg.label_11 = QtWidgets.QLabel(self)
    #     self.penDlg.label_11.setText('\tx = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_11, 8, 0, 1, 1)
    #
    #     self.x3_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.x3_pen, 8, 1, 1, 2)
    #
    #     self.penDlg.label_12 = QtWidgets.QLabel(self)
    #     self.penDlg.label_12.setText('y = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_12, 8, 4, 1, 1)
    #
    #     self.y3_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.y3_pen, 8, 5, 1, 2)
    #
    #     self.penDlg.label_13 = QtWidgets.QLabel(self)
    #     self.penDlg.label_13.setText('')
    #     self.penDlg.grid.addWidget(self.penDlg.label_13, 9, 0, 1, 1)
    #
    #     self.penDlg.label_14 = QtWidgets.QLabel(self)
    #     self.penDlg.label_14.setText('Point_4 : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_14, 10, 0, 1, 2)
    #
    #     self.penDlg.label_15 = QtWidgets.QLabel(self)
    #     self.penDlg.label_15.setText('\tx = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_15, 11, 0, 1, 1)
    #
    #     self.x4_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.x4_pen, 11, 1, 1, 2)
    #
    #     self.penDlg.label_16 = QtWidgets.QLabel(self)
    #     self.penDlg.label_16.setText('y = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_16, 11, 4, 1, 1)
    #
    #     self.y4_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.y4_pen, 11, 5, 1, 2)
    #
    #     self.penDlg.label_17 = QtWidgets.QLabel(self)
    #     self.penDlg.label_17.setText('')
    #     self.penDlg.grid.addWidget(self.penDlg.label_17, 12, 0, 1, 1)
    #
    #     self.penDlg.label_18 = QtWidgets.QLabel(self)
    #     self.penDlg.label_18.setText('Point_5 : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_18, 13, 0, 1, 2)
    #
    #     self.penDlg.label_19 = QtWidgets.QLabel(self)
    #     self.penDlg.label_19.setText('\tx = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_19, 14, 0, 1, 1)
    #
    #     self.x5_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.x5_pen, 14, 1, 1, 2)
    #
    #     self.penDlg.label_20 = QtWidgets.QLabel(self)
    #     self.penDlg.label_20.setText('y = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_20, 14, 4, 1, 1)
    #
    #     self.y5_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.y5_pen, 14, 5, 1, 2)
    #
    #     self.penDlg.label_21 = QtWidgets.QLabel(self)
    #     self.penDlg.label_21.setText('')
    #     self.penDlg.grid.addWidget(self.penDlg.label_21, 15, 0, 1, 1)
    #
    #     self.penDlg.label_22 = QtWidgets.QLabel(self)
    #     self.penDlg.label_22.setText('Color of Border Line (B,G,R) : ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_22, 16, 0, 1, 3)
    #
    #     self.penDlg.label_23 = QtWidgets.QLabel(self)
    #     self.penDlg.label_23.setText('Intensity of  Blue  (max 255) = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_23, 17, 0, 1, 2)
    #
    #     self.blue_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.blue_pen, 17, 2, 1, 2)
    #
    #     self.penDlg.label_24 = QtWidgets.QLabel(self)
    #     self.penDlg.label_24.setText('Intensity of Green (max 255) = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_24, 18, 0, 1, 2)
    #
    #     self.green_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.green_pen, 18, 2, 1, 2)
    #
    #     self.penDlg.label_25 = QtWidgets.QLabel(self)
    #     self.penDlg.label_25.setText('Intensity of   Red   (max 255) = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_25, 19, 0, 1, 2)
    #
    #     self.red_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.red_pen, 19, 2, 1, 2)
    #
    #     self.penDlg.label_26 = QtWidgets.QLabel(self)
    #     self.penDlg.label_26.setText(' ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_26, 20, 0, 1, 1)
    #
    #     self.penDlg.label_27 = QtWidgets.QLabel(self)
    #     self.penDlg.label_27.setText('Thickness of Border Line = ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_27, 21, 0, 1, 2)
    #
    #     self.thick_pen = QtWidgets.QLineEdit(self)
    #     self.penDlg.grid.addWidget(self.thick_pen, 21, 2, 1, 2)
    #
    #     self.penDlg.label_28 = QtWidgets.QLabel(self)
    #     self.penDlg.label_28.setText('(-1 to fill pentagon with (B,G,R) color) ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_28, 22, 2, 1, 5)
    #
    #     self.penDlg.label_29 = QtWidgets.QLabel(self)
    #     self.penDlg.label_29.setText(' ')
    #     self.penDlg.grid.addWidget(self.penDlg.label_29, 23, 0, 1, 1)
    #
    #     btn = QtWidgets.QPushButton('OK', self)
    #     btn.clicked.connect(self.drawPen)
    #     btn.resize(btn.sizeHint())
    #     self.penDlg.grid.addWidget(btn, 24, 0, 1, 8)
    #
    #     self.penDlg.show()
    #
    # def drawPen(self):
    #     self.penDlg.close()
    #
    #     self.pts = np.array([[int(self.x1_pen.text()),int(self.y1_pen.text())],
    #                          [int(self.x2_pen.text()),int(self.y2_pen.text())],
    #                          [int(self.x3_pen.text()),int(self.y3_pen.text())],
    #                          [int(self.x4_pen.text()),int(self.y4_pen.text())],
    #                          [int(self.x5_pen.text()),int(self.y5_pen.text())]],np.int32)
    #     self.pts=self.pts.reshape((-1,1,2))
    #
    #     cv2.polylines(self.image, [self.pts], True,
    #                   (int(self.blue_pen.text()),int(self.green_pen.text()),int(self.red_pen.text())),
    #                   int(self.thick_pen.text()))
    #     cv2.imshow('draw pentagoon', self.image)
    #     self.updateColor()

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

        btn = QtWidgets.QPushButton('Random', self)
        btn.clicked.connect(self.randomText)
        btn.resize(btn.sizeHint())
        self.txtDlg.grid.addWidget(btn, 16, 1, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.printText)
        btn1.resize(btn1.sizeHint())
        self.txtDlg.grid.addWidget(btn1,16,4,1,2)

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

    def randomText(self):
        self.string.setText("typing")
        self.x_txt.setText(str(random.randint(0,int(self.imgWidth / 4))))
        self.y_txt.setText(str(random.randint(10, int(self.imgHeight / 4))))
        self.fontType("FONT_HERSHEY_SIMPLEX")
        self.fs_txt.setText(str(random.randint(1,3)))
        self.blue_txt.setText(str(random.randint(0, 255)))
        self.green_txt.setText(str(random.randint(0, 255)))
        self.red_txt.setText(str(random.randint(0, 255)))
        self.thick_txt.setText(str(random.randint(1, 10)))

    def printText(self):
        self.txtDlg.close()

        cv2.putText(self.image, str(self.string.text()), (int(self.x_txt.text()),int(self.y_txt.text())),
                    self.font, int(self.fs_txt.text()),
                    (int(self.blue_txt.text()),int(self.green_txt.text()),int(self.red_txt.text())),
                    int(self.thick_txt.text()))
        # cv2.imshow('print string', self.image)
        self.updateColor()

    def split_1x2(self):
        self.s1_2Dlg = QMainWindow(self)
        self.s1_2Dlg.setWindowTitle('Split 1x2')

        self.s1_2Dlg.central_widget = QtWidgets.QWidget()
        self.s1_2Dlg.setCentralWidget(self.s1_2Dlg.central_widget)
        self.s1_2Dlg.vbox = QtWidgets.QVBoxLayout(self.s1_2Dlg.central_widget)

        self.s1_2Dlg.label = QtWidgets.QLabel(self)
        self.s1_2Dlg.label.setStyleSheet("font : 20pt")
        self.s1_2Dlg.label.setText('Preview split 1x2')
        self.s1_2Dlg.vbox.addWidget(self.s1_2Dlg.label)

        averageHeight = math.ceil(self.imgHeight / 2)
        im_1, im_2 = np.vsplit(self.image, [averageHeight])
        self.listImg = [im_1, im_2]
        # cv2.imshow('slice_1', im_1)
        # cv2.imshow('slice_2', im_2)

        self.s1_2Dlg.label_2 = QtWidgets.QLabel(self)
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 150,QtCore.Qt.KeepAspectRatio)
        self.s1_2Dlg.label_2.setPixmap(pixmap)
        self.s1_2Dlg.vbox.addWidget(self.s1_2Dlg.label_2)

        self.s1_2Dlg.label_3 = QtWidgets.QLabel(self)
        self.s1_2Dlg.label_3.setText('')
        self.s1_2Dlg.vbox.addWidget(self.s1_2Dlg.label_3)

        self.s1_2Dlg.label_4 = QtWidgets.QLabel(self)
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s1_2Dlg.label_4.setPixmap(pixmap)
        self.s1_2Dlg.vbox.addWidget(self.s1_2Dlg.label_4)

        self.s1_2Dlg.label_5 = QtWidgets.QLabel(self)
        self.s1_2Dlg.label_5.setText('')
        self.s1_2Dlg.vbox.addWidget(self.s1_2Dlg.label_5)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS1_2)
        btn_1.resize(btn_1.sizeHint())
        self.s1_2Dlg.vbox.addWidget(btn_1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS1_2)
        btn_2.resize(btn_2.sizeHint())
        self.s1_2Dlg.vbox.addWidget(btn_2)

        self.s1_2Dlg.show()

    def closeS1_2(self):
        self.s1_2Dlg.close()

    def saveSplit(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\',
                                                              "Image Files (*.jpg);;Image Files (*.tiff);;Image Files (*.bmp)")

        for i in range(len(self.listImg)):
            if fname.endswith(".jpg"):
                name = fname[0:len(fname) - 4] + "_" + str(i + 1) + ".jpg"
            elif fname.endswith(".tiff"):
                name = fname[0:len(fname) - 5] + "_" + str(i + 1) + ".tiff"
            elif fname.endswith(".bmp"):
                name = fname[0:len(fname) - 4] + "_" + str(i + 1) + ".bmp"
            if name:
                cv2.imwrite(name, self.listImg[i])
            else:
                print('Error')

    def saveS1_2(self):
        self.closeS1_2()
        self.saveSplit()

    def split_1x3(self):
        self.s1_3Dlg = QMainWindow(self)
        self.s1_3Dlg.setWindowTitle('Split 1x3')

        self.s1_3Dlg.central_widget = QtWidgets.QWidget()
        self.s1_3Dlg.setCentralWidget(self.s1_3Dlg.central_widget)
        self.s1_3Dlg.vbox = QtWidgets.QVBoxLayout(self.s1_3Dlg.central_widget)

        self.s1_3Dlg.label = QtWidgets.QLabel(self)
        self.s1_3Dlg.label.setStyleSheet("font : 20pt")
        self.s1_3Dlg.label.setText('Preview split 1x3')
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label)

        averageHeight = math.ceil(self.imgHeight / 3)
        im_1, im_2, im_3 = np.vsplit(self.image, [averageHeight, averageHeight*2])
        self.listImg = [im_1, im_2, im_3]

        self.s1_3Dlg.label_2 = QtWidgets.QLabel(self)
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 100,QtCore.Qt.KeepAspectRatio)
        self.s1_3Dlg.label_2.setPixmap(pixmap)
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_2)

        self.s1_3Dlg.label_3 = QtWidgets.QLabel(self)
        self.s1_3Dlg.label_3.setText('')
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_3)

        self.s1_3Dlg.label_4 = QtWidgets.QLabel(self)
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s1_3Dlg.label_4.setPixmap(pixmap)
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_4)

        self.s1_3Dlg.label_5 = QtWidgets.QLabel(self)
        self.s1_3Dlg.label_5.setText('')
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_5)

        self.s1_3Dlg.label_6 = QtWidgets.QLabel(self)
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s1_3Dlg.label_6.setPixmap(pixmap)
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_6)

        self.s1_3Dlg.label_7 = QtWidgets.QLabel(self)
        self.s1_3Dlg.label_7.setText('')
        self.s1_3Dlg.vbox.addWidget(self.s1_3Dlg.label_7)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS1_3)
        btn_1.resize(btn_1.sizeHint())
        self.s1_3Dlg.vbox.addWidget(btn_1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS1_3)
        btn_2.resize(btn_2.sizeHint())
        self.s1_3Dlg.vbox.addWidget(btn_2)

        self.s1_3Dlg.show()

    def closeS1_3(self):
        self.s1_3Dlg.close()

    def saveS1_3(self):
        self.closeS1_3()
        self.saveSplit()

    def split_2x1(self):
        self.s2_1Dlg = QMainWindow(self)
        self.s2_1Dlg.setWindowTitle('Split 2x1')

        self.s2_1Dlg.central_widget = QtWidgets.QWidget()
        self.s2_1Dlg.setCentralWidget(self.s2_1Dlg.central_widget)
        self.s2_1Dlg.grid = QtWidgets.QGridLayout(self.s2_1Dlg.central_widget)

        self.s2_1Dlg.label = QtWidgets.QLabel(self)
        self.s2_1Dlg.label.setStyleSheet("font : 20pt")
        self.s2_1Dlg.label.setText('Preview split 2x1')
        self.s2_1Dlg.grid.addWidget(self.s2_1Dlg.label,0,0,1,7)

        averageWidth = math.ceil(self.imgWidth / 2)
        im_1, im_2 = np.hsplit(self.image,[averageWidth])
        self.listImg = [im_1, im_2]
        # cv2.imshow('a',im_1)
        # cv2.imshow('b',im_2)

        self.s2_1Dlg.label_2 = QtWidgets.QLabel(self)
        im_1=np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.s2_1Dlg.label_2.setPixmap(pixmap)
        self.s2_1Dlg.grid.addWidget(self.s2_1Dlg.label_2,1,0,1,1)

        self.s2_1Dlg.label_3 = QtWidgets.QLabel(self)
        im_2=np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.s2_1Dlg.label_3.setPixmap(pixmap)
        self.s2_1Dlg.grid.addWidget(self.s2_1Dlg.label_3, 1, 2, 1, 1)

        self.s2_1Dlg.label_4 = QtWidgets.QLabel(self)
        self.s2_1Dlg.label_4.setText('')
        self.s2_1Dlg.grid.addWidget(self.s2_1Dlg.label_4,2,1,1,1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS2_1)
        btn_1.resize(btn_1.sizeHint())
        self.s2_1Dlg.grid.addWidget(btn_1,3,0,1,1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS2_1)
        btn_2.resize(btn_2.sizeHint())
        self.s2_1Dlg.grid.addWidget(btn_2,3,2,1,1)

        self.s2_1Dlg.show()

    def closeS2_1(self):
        self.s2_1Dlg.close()

    def saveS2_1(self):
        self.s2_1Dlg.close()
        self.saveSplit()

    def split_2x2(self):
        self.s2_2Dlg = QMainWindow(self)
        self.s2_2Dlg.setWindowTitle('Split 2x2')

        self.s2_2Dlg.central_widget = QtWidgets.QWidget()
        self.s2_2Dlg.setCentralWidget(self.s2_2Dlg.central_widget)
        self.s2_2Dlg.grid = QtWidgets.QGridLayout(self.s2_2Dlg.central_widget)

        self.s2_2Dlg.label = QtWidgets.QLabel(self)
        self.s2_2Dlg.label.setStyleSheet("font : 20pt")
        self.s2_2Dlg.label.setText('Preview split 2x2')
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label,0,0,1,7)

        averageWidth = math.ceil(self.imgWidth / 2)
        averageHeight = math.ceil(self.imgHeight / 2)
        im_1, im_2 = np.hsplit(self.image,[averageWidth])
        im_1, im_3 = np.vsplit(im_1, [averageHeight])
        im_2, im_4 = np.vsplit(im_2, [averageHeight])
        self.listImg = [im_1, im_2, im_3, im_4]

        self.s2_2Dlg.label_2 = QtWidgets.QLabel(self)
        im_1=np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s2_2Dlg.label_2.setPixmap(pixmap)
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_2,1,0,1,1)

        self.s2_2Dlg.label_3 = QtWidgets.QLabel(self)
        im_2=np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s2_2Dlg.label_3.setPixmap(pixmap)
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_3, 1, 2, 1, 1)

        self.s2_2Dlg.label_4 = QtWidgets.QLabel(self)
        self.s2_2Dlg.label_4.setText('')
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_4,2,1,1,1)

        self.s2_2Dlg.label_5 = QtWidgets.QLabel(self)
        im_3 = np.require(im_3, np.uint8, 'C')
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s2_2Dlg.label_5.setPixmap(pixmap)
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_5, 3, 0, 1, 1)

        self.s2_2Dlg.label_6 = QtWidgets.QLabel(self)
        im_4 = np.require(im_4, np.uint8, 'C')
        img_4 = QImage(im_4, im_4.shape[1], im_4.shape[0], im_4.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_4)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s2_2Dlg.label_6.setPixmap(pixmap)
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_6, 3, 2, 1, 1)

        self.s2_2Dlg.label_7 = QtWidgets.QLabel(self)
        self.s2_2Dlg.label_7.setText('')
        self.s2_2Dlg.grid.addWidget(self.s2_2Dlg.label_7, 4, 1, 1, 1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS2_2)
        btn_1.resize(btn_1.sizeHint())
        self.s2_2Dlg.grid.addWidget(btn_1,5,0,1,1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS2_2)
        btn_2.resize(btn_2.sizeHint())
        self.s2_2Dlg.grid.addWidget(btn_2,5,2,1,1)

        self.s2_2Dlg.show()

    def closeS2_2(self):
        self.s2_2Dlg.close()

    def saveS2_2(self):
        self.s2_2Dlg.close()
        self.saveSplit()

    def split_2x3(self):
        self.s2_3Dlg = QMainWindow(self)
        self.s2_3Dlg.setWindowTitle('Split 2x3')

        self.s2_3Dlg.central_widget = QtWidgets.QWidget()
        self.s2_3Dlg.setCentralWidget(self.s2_3Dlg.central_widget)
        self.s2_3Dlg.grid = QtWidgets.QGridLayout(self.s2_3Dlg.central_widget)

        self.s2_3Dlg.label = QtWidgets.QLabel(self)
        self.s2_3Dlg.label.setStyleSheet("font : 20pt")
        self.s2_3Dlg.label.setText('Preview split 2x3')
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label, 0, 0, 1, 7)

        averageWidth = math.ceil(self.imgWidth / 2)
        averageHeight = math.ceil(self.imgHeight / 3)
        im_1, im_2 = np.hsplit(self.image, [averageWidth])
        im_1, im_3, im_5 = np.vsplit(im_1, [averageHeight, averageHeight*2])
        im_2, im_4, im_6 = np.vsplit(im_2, [averageHeight,averageHeight*2])
        self.listImg = [im_1, im_2, im_3, im_4, im_5, im_6]

        self.s2_3Dlg.label_2 = QtWidgets.QLabel(self)
        im_1 = np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_2.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_2, 1, 0, 1, 1)

        self.s2_3Dlg.label_3 = QtWidgets.QLabel(self)
        im_2 = np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_3.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_3, 1, 2, 1, 1)

        self.s2_3Dlg.label_4 = QtWidgets.QLabel(self)
        self.s2_3Dlg.label_4.setText('')
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_4, 2, 1, 1, 1)

        self.s2_3Dlg.label_5 = QtWidgets.QLabel(self)
        im_3 = np.require(im_3, np.uint8, 'C')
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_5.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_5, 3, 0, 1, 1)

        self.s2_3Dlg.label_6 = QtWidgets.QLabel(self)
        im_4 = np.require(im_4, np.uint8, 'C')
        img_4 = QImage(im_4, im_4.shape[1], im_4.shape[0], im_4.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_4)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_6.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_6, 3, 2, 1, 1)

        self.s2_3Dlg.label_7 = QtWidgets.QLabel(self)
        self.s2_3Dlg.label_7.setText('')
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_7, 4, 1, 1, 1)

        self.s2_3Dlg.label_8 = QtWidgets.QLabel(self)
        im_5 = np.require(im_5, np.uint8, 'C')
        img_5 = QImage(im_5, im_5.shape[1], im_5.shape[0], im_5.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_5)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_8.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_8, 5, 0, 1, 1)

        self.s2_3Dlg.label_9 = QtWidgets.QLabel(self)
        im_6 = np.require(im_6, np.uint8, 'C')
        img_6 = QImage(im_6, im_6.shape[1], im_6.shape[0], im_6.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_6)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s2_3Dlg.label_9.setPixmap(pixmap)
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_9, 5, 2, 1, 1)

        self.s2_3Dlg.label_10 = QtWidgets.QLabel(self)
        self.s2_3Dlg.label_10.setText('')
        self.s2_3Dlg.grid.addWidget(self.s2_3Dlg.label_10, 6, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS2_3)
        btn_1.resize(btn_1.sizeHint())
        self.s2_3Dlg.grid.addWidget(btn_1, 7, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS2_3)
        btn_2.resize(btn_2.sizeHint())
        self.s2_3Dlg.grid.addWidget(btn_2, 7, 2, 1, 1)

        self.s2_3Dlg.show()

    def closeS2_3(self):
        self.s2_3Dlg.close()

    def saveS2_3(self):
        self.s2_3Dlg.close()
        self.saveSplit()

    def split_3x1(self):
        self.s3_1Dlg = QMainWindow(self)
        self.s3_1Dlg.setWindowTitle('Split 3x1')

        self.s3_1Dlg.central_widget = QtWidgets.QWidget()
        self.s3_1Dlg.setCentralWidget(self.s3_1Dlg.central_widget)
        self.s3_1Dlg.grid = QtWidgets.QGridLayout(self.s3_1Dlg.central_widget)

        self.s3_1Dlg.label = QtWidgets.QLabel(self)
        self.s3_1Dlg.label.setStyleSheet("font : 20pt")
        self.s3_1Dlg.label.setText('Preview split 3x1')
        self.s3_1Dlg.grid.addWidget(self.s3_1Dlg.label,0,0,1,7)

        averageWidth = math.ceil(self.imgWidth / 3)
        im_1, im_2 , im_3= np.hsplit(self.image,[averageWidth, averageWidth*2])
        self.listImg = [im_1, im_2, im_3]

        self.s3_1Dlg.label_2 = QtWidgets.QLabel(self)
        im_1=np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.s3_1Dlg.label_2.setPixmap(pixmap)
        self.s3_1Dlg.grid.addWidget(self.s3_1Dlg.label_2,1,0,1,1)

        self.s3_1Dlg.label_3 = QtWidgets.QLabel(self)
        im_2=np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.s3_1Dlg.label_3.setPixmap(pixmap)
        self.s3_1Dlg.grid.addWidget(self.s3_1Dlg.label_3, 1, 2, 1, 1)

        self.s3_1Dlg.label_4 = QtWidgets.QLabel(self)
        im_3 = np.require(im_3, np.uint8, 'C')
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.s3_1Dlg.label_4.setPixmap(pixmap)
        self.s3_1Dlg.grid.addWidget(self.s3_1Dlg.label_4, 1, 4, 1, 1)

        self.s3_1Dlg.label_5 = QtWidgets.QLabel(self)
        self.s3_1Dlg.label_5.setText('')
        self.s3_1Dlg.grid.addWidget(self.s3_1Dlg.label_5,2,1,1,1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS3_1)
        btn_1.resize(btn_1.sizeHint())
        self.s3_1Dlg.grid.addWidget(btn_1,3,0,1,1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS3_1)
        btn_2.resize(btn_2.sizeHint())
        self.s3_1Dlg.grid.addWidget(btn_2,3,4,1,1)

        self.s3_1Dlg.show()

    def closeS3_1(self):
        self.s3_1Dlg.close()

    def saveS3_1(self):
        self.s3_1Dlg.close()
        self.saveSplit()

    def split_3x2(self):
        self.s3_2Dlg = QMainWindow(self)
        self.s3_2Dlg.setWindowTitle('Split 3x2')

        self.s3_2Dlg.central_widget = QtWidgets.QWidget()
        self.s3_2Dlg.setCentralWidget(self.s3_2Dlg.central_widget)
        self.s3_2Dlg.grid = QtWidgets.QGridLayout(self.s3_2Dlg.central_widget)

        self.s3_2Dlg.label = QtWidgets.QLabel(self)
        self.s3_2Dlg.label.setStyleSheet("font : 20pt")
        self.s3_2Dlg.label.setText('Preview split 3x2')
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label, 0, 0, 1, 7)

        averageWidth = math.ceil(self.imgWidth / 3)
        averageHeight = math.ceil(self.imgHeight / 2)
        im_1, im_2, im_3 = np.hsplit(self.image, [averageWidth, averageWidth*2])
        im_1, im_4 = np.vsplit(im_1, [averageHeight])
        im_2, im_5 = np.vsplit(im_2, [averageHeight])
        im_3, im_6 = np.vsplit(im_3, [averageHeight])
        self.listImg = [im_1, im_2, im_3, im_4, im_5, im_6]

        self.s3_2Dlg.label_2 = QtWidgets.QLabel(self)
        im_1 = np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_2.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_2, 1, 0, 1, 1)

        self.s3_2Dlg.label_3 = QtWidgets.QLabel(self)
        im_2 = np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_3.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_3, 1, 2, 1, 1)

        self.s3_2Dlg.label_4 = QtWidgets.QLabel(self)
        im_3 = np.require(im_3, np.uint8, 'C')
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_4.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_4, 1, 4, 1, 1)

        self.s3_2Dlg.label_5 = QtWidgets.QLabel(self)
        self.s3_2Dlg.label_5.setText('')
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_5, 2, 1, 1, 1)

        self.s3_2Dlg.label_6 = QtWidgets.QLabel(self)
        im_4 = np.require(im_4, np.uint8, 'C')
        img_4 = QImage(im_4, im_4.shape[1], im_4.shape[0], im_4.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_4)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_6.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_6, 3, 0, 1, 1)

        self.s3_2Dlg.label_7 = QtWidgets.QLabel(self)
        im_5 = np.require(im_5, np.uint8, 'C')
        img_5 = QImage(im_5, im_5.shape[1], im_5.shape[0], im_5.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_5)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_7.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_7, 3, 2, 1, 1)

        self.s3_2Dlg.label_8 = QtWidgets.QLabel(self)
        im_6 = np.require(im_6, np.uint8, 'C')
        img_6 = QImage(im_6, im_6.shape[1], im_6.shape[0], im_6.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_6)
        pixmap = pixmap.scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        self.s3_2Dlg.label_8.setPixmap(pixmap)
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_8, 3, 4, 1, 1)

        self.s3_2Dlg.label_9 = QtWidgets.QLabel(self)
        self.s3_2Dlg.label_9.setText('')
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_9, 4, 1, 1, 1)

        self.s3_2Dlg.label_10 = QtWidgets.QLabel(self)
        self.s3_2Dlg.label_10.setText('')
        self.s3_2Dlg.grid.addWidget(self.s3_2Dlg.label_10, 5, 1, 1, 1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS3_2)
        btn_1.resize(btn_1.sizeHint())
        self.s3_2Dlg.grid.addWidget(btn_1, 6, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS3_2)
        btn_2.resize(btn_2.sizeHint())
        self.s3_2Dlg.grid.addWidget(btn_2, 6, 4, 1, 1)

        self.s3_2Dlg.show()

    def closeS3_2(self):
        self.s3_2Dlg.close()

    def saveS3_2(self):
        self.s3_2Dlg.close()
        self.saveSplit()

    def split_3x3(self):
        self.s3_3Dlg = QMainWindow(self)
        self.s3_3Dlg.setWindowTitle('Split 3x3')

        self.s3_3Dlg.central_widget = QtWidgets.QWidget()
        self.s3_3Dlg.setCentralWidget(self.s3_3Dlg.central_widget)
        self.s3_3Dlg.grid = QtWidgets.QGridLayout(self.s3_3Dlg.central_widget)

        self.s3_3Dlg.label = QtWidgets.QLabel(self)
        self.s3_3Dlg.label.setStyleSheet("font : 20pt")
        self.s3_3Dlg.label.setText('Preview split 3x3')
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label, 0, 0, 1, 7)

        averageWidth = math.ceil(self.imgWidth / 3)
        averageHeight = math.ceil(self.imgHeight / 3)
        im_1, im_2, im_3 = np.hsplit(self.image, [averageWidth, averageWidth*2])
        im_1, im_4, im_7 = np.vsplit(im_1, [averageHeight,averageHeight*2])
        im_2, im_5, im_8 = np.vsplit(im_2, [averageHeight, averageHeight*2])
        im_3, im_6, im_9 = np.vsplit(im_3, [averageHeight, averageHeight * 2])
        self.listImg = [im_1, im_2, im_3, im_4, im_5, im_6, im_7, im_8, im_9]

        self.s3_3Dlg.label_2 = QtWidgets.QLabel(self)
        im_1 = np.require(im_1, np.uint8, 'C')
        img_1 = QImage(im_1, im_1.shape[1], im_1.shape[0], im_1.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_1)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_2.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_2, 1, 0, 1, 1)

        self.s3_3Dlg.label_3 = QtWidgets.QLabel(self)
        im_2 = np.require(im_2, np.uint8, 'C')
        img_2 = QImage(im_2, im_2.shape[1], im_2.shape[0], im_2.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_2)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_3.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_3, 1, 2, 1, 1)

        self.s3_3Dlg.label_4 = QtWidgets.QLabel(self)
        im_3 = np.require(im_3, np.uint8, 'C')
        img_3 = QImage(im_3, im_3.shape[1], im_3.shape[0], im_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_3)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_4.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_4, 1, 4, 1, 1)

        self.s3_3Dlg.label_5 = QtWidgets.QLabel(self)
        self.s3_3Dlg.label_5.setText('')
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_5, 2, 1, 1, 1)

        self.s3_3Dlg.label_6 = QtWidgets.QLabel(self)
        im_4 = np.require(im_4, np.uint8, 'C')
        img_4 = QImage(im_4, im_4.shape[1], im_4.shape[0], im_4.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_4)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_6.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_6, 3, 0, 1, 1)

        self.s3_3Dlg.label_7 = QtWidgets.QLabel(self)
        im_5 = np.require(im_5, np.uint8, 'C')
        img_5 = QImage(im_5, im_5.shape[1], im_5.shape[0], im_5.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_5)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_7.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_7, 3, 2, 1, 1)

        self.s3_3Dlg.label_8 = QtWidgets.QLabel(self)
        im_6 = np.require(im_6, np.uint8, 'C')
        img_6 = QImage(im_6, im_6.shape[1], im_6.shape[0], im_6.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_6)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_8.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_8, 3, 4, 1, 1)

        self.s3_3Dlg.label_9 = QtWidgets.QLabel(self)
        self.s3_3Dlg.label_9.setText('')
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_9, 4, 1, 1, 1)

        self.s3_3Dlg.label_10 = QtWidgets.QLabel(self)
        im_7 = np.require(im_7, np.uint8, 'C')
        img_7 = QImage(im_7, im_7.shape[1], im_7.shape[0], im_7.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_7)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_10.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_10, 5, 0, 1, 1)

        self.s3_3Dlg.label_11 = QtWidgets.QLabel(self)
        im_8 = np.require(im_8, np.uint8, 'C')
        img_8 = QImage(im_8, im_8.shape[1], im_8.shape[0], im_8.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_8)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_11.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_11, 5, 2, 1, 1)

        self.s3_3Dlg.label_12 = QtWidgets.QLabel(self)
        im_9 = np.require(im_9, np.uint8, 'C')
        img_9 = QImage(im_9, im_9.shape[1], im_9.shape[0], im_9.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img_9)
        pixmap = pixmap.scaled(200, 100, QtCore.Qt.KeepAspectRatio)
        self.s3_3Dlg.label_12.setPixmap(pixmap)
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_12, 5, 4, 1, 1)

        self.s3_3Dlg.label_13 = QtWidgets.QLabel(self)
        self.s3_3Dlg.label_13.setText('')
        self.s3_3Dlg.grid.addWidget(self.s3_3Dlg.label_13, 6, 1, 1, 1)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeS3_3)
        btn_1.resize(btn_1.sizeHint())
        self.s3_3Dlg.grid.addWidget(btn_1, 7, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Save All', self)
        btn_2.clicked.connect(self.saveS3_3)
        btn_2.resize(btn_2.sizeHint())
        self.s3_3Dlg.grid.addWidget(btn_2, 7, 4, 1, 1)

        self.s3_3Dlg.show()

    def closeS3_3(self):
        self.s3_3Dlg.close()

    def saveS3_3(self):
        self.s3_3Dlg.close()
        self.saveSplit()

    def merge_1x2(self):
        self.m1_2Dlg = QMainWindow(self)
        self.m1_2Dlg.setWindowTitle('Collage 1x2')

        self.m1_2Dlg.central_widget = QtWidgets.QWidget()
        self.m1_2Dlg.setCentralWidget(self.m1_2Dlg.central_widget)
        self.m1_2Dlg.grid = QtWidgets.QGridLayout(self.m1_2Dlg.central_widget)

        self.m1_2Dlg.label = QtWidgets.QLabel(self)
        self.m1_2Dlg.label.setStyleSheet("font : 20pt")
        self.m1_2Dlg.label.setText('Combine 2 image as 1x2')
        self.m1_2Dlg.grid.addWidget(self.m1_2Dlg.label, 0, 0, 1, 7)

        self.m1_2Dlg.label_2 = QtWidgets.QLabel(self)
        self.m1_2Dlg.label_2.setText('Image_1 = ')
        self.m1_2Dlg.grid.addWidget(self.m1_2Dlg.label_2, 1, 0, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m1_2Dlg.grid.addWidget(self.img_1, 1, 1, 1, 4)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m1_2Dlg.grid.addWidget(btn_1, 1, 5, 1, 1)

        self.m1_2Dlg.label_3 = QtWidgets.QLabel(self)
        self.m1_2Dlg.label_3.setText('Image_2 = ')
        self.m1_2Dlg.grid.addWidget(self.m1_2Dlg.label_3, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m1_2Dlg.grid.addWidget(self.img_2, 2, 1, 1, 4)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m1_2Dlg.grid.addWidget(btn_2, 2, 5, 1, 1)

        btn_3 = QtWidgets.QPushButton('OK', self)
        btn_3.clicked.connect(self.previewMerge_1x2)
        btn_3.resize(btn_3.sizeHint())
        self.m1_2Dlg.grid.addWidget(btn_3, 3, 1, 1, 3)

        self.m1_2Dlg.show()

    def addImage_1(self):
        self.name_1 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_1.setText(self.name_1)

    def addImage_2(self):
        self.name_2 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_2.setText(self.name_2)

    def previewMerge_1x2(self):
        self.m1_2Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.preview = self.vconcat_resize([self.image_1, self.image_2])

        self.previewMerge()

    def vconcat_resize(self,img_list,interpolation=cv2.INTER_CUBIC):
        w_min=min(img.shape[1] for img in img_list)
        im_list_resize=[cv2.resize(img,(w_min,int(img.shape[0]*w_min/img.shape[1])),interpolation=interpolation)
                        for img in img_list]
        return cv2.vconcat(im_list_resize)

    def previewMerge(self):
        self.pmDlg = QMainWindow(self)
        self.pmDlg.setWindowTitle('Collage')

        self.pmDlg.central_widget = QtWidgets.QWidget()
        self.pmDlg.setCentralWidget(self.pmDlg.central_widget)
        self.pmDlg.vbox = QtWidgets.QVBoxLayout(self.pmDlg.central_widget)

        self.pmDlg.label = QtWidgets.QLabel(self)
        self.pmDlg.label.setStyleSheet("font : 20pt")
        self.pmDlg.label.setText('Preview Merge')
        self.pmDlg.vbox.addWidget(self.pmDlg.label)

        self.pmDlg.label_2 = QtWidgets.QLabel(self)
        img = QImage(self.preview, self.preview.shape[1], self.preview.shape[0], self.preview.shape[1] * 3,
                     QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(img)
        pixmap = pixmap.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.pmDlg.label_2.setPixmap(pixmap)
        self.pmDlg.vbox.addWidget(self.pmDlg.label_2)

        self.pmDlg.label_3 = QtWidgets.QLabel(self)
        self.pmDlg.label_3.setText('')
        self.pmDlg.vbox.addWidget(self.pmDlg.label_3)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closePreview)
        btn_1.resize(btn_1.sizeHint())
        self.pmDlg.vbox.addWidget(btn_1)

        btn_2 = QtWidgets.QPushButton('Apply', self)
        btn_2.clicked.connect(self.applyMerge)
        btn_2.resize(btn_2.sizeHint())
        self.pmDlg.vbox.addWidget(btn_2)

        self.pmDlg.show()

    def closePreview(self):
        self.pmDlg.close()

    def applyMerge(self):
        self.pmDlg.close()
        self.showMerge()

    def showMerge(self):
        self.image = self.preview
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        self.updateColor()
        self.updateDetail()

    def merge_1x3(self):
        self.m1_3Dlg = QMainWindow(self)
        self.m1_3Dlg.setWindowTitle('Collage 1x3')

        self.m1_3Dlg.central_widget = QtWidgets.QWidget()
        self.m1_3Dlg.setCentralWidget(self.m1_3Dlg.central_widget)
        self.m1_3Dlg.grid = QtWidgets.QGridLayout(self.m1_3Dlg.central_widget)

        self.m1_3Dlg.label = QtWidgets.QLabel(self)
        self.m1_3Dlg.label.setStyleSheet("font : 20pt")
        self.m1_3Dlg.label.setText('Combine 3 image as 1x3')
        self.m1_3Dlg.grid.addWidget(self.m1_3Dlg.label, 0, 0, 1, 7)

        self.m1_3Dlg.label_2 = QtWidgets.QLabel(self)
        self.m1_3Dlg.label_2.setText('Image_1 = ')
        self.m1_3Dlg.grid.addWidget(self.m1_3Dlg.label_2, 1, 0, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m1_3Dlg.grid.addWidget(self.img_1, 1, 1, 1, 4)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m1_3Dlg.grid.addWidget(btn_1, 1, 5, 1, 1)

        self.m1_3Dlg.label_3 = QtWidgets.QLabel(self)
        self.m1_3Dlg.label_3.setText('Image_2 = ')
        self.m1_3Dlg.grid.addWidget(self.m1_3Dlg.label_3, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m1_3Dlg.grid.addWidget(self.img_2, 2, 1, 1, 4)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m1_3Dlg.grid.addWidget(btn_2, 2, 5, 1, 1)

        self.m1_3Dlg.label_4 = QtWidgets.QLabel(self)
        self.m1_3Dlg.label_4.setText('Image_3 = ')
        self.m1_3Dlg.grid.addWidget(self.m1_3Dlg.label_4, 3, 0, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m1_3Dlg.grid.addWidget(self.img_3, 3, 1, 1, 4)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m1_3Dlg.grid.addWidget(btn_3, 3, 5, 1, 1)

        btn_4 = QtWidgets.QPushButton('OK', self)
        btn_4.clicked.connect(self.previewMerge_1x3)
        btn_4.resize(btn_3.sizeHint())
        self.m1_3Dlg.grid.addWidget(btn_4, 4, 1, 1, 3)

        self.m1_3Dlg.show()

    def addImage_3(self):
        self.name_3 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_3.setText(self.name_3)

    def previewMerge_1x3(self):
        self.m1_3Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.preview = self.vconcat_resize([self.image_1, self.image_2, self.image_3])

        self.previewMerge()

    def merge_2x1(self):
        self.m2_1Dlg = QMainWindow(self)
        self.m2_1Dlg.setWindowTitle('Collage 2x1')

        self.m2_1Dlg.central_widget = QtWidgets.QWidget()
        self.m2_1Dlg.setCentralWidget(self.m2_1Dlg.central_widget)
        self.m2_1Dlg.grid = QtWidgets.QGridLayout(self.m2_1Dlg.central_widget)

        self.m2_1Dlg.label = QtWidgets.QLabel(self)
        self.m2_1Dlg.label.setStyleSheet("font : 20pt")
        self.m2_1Dlg.label.setText('Combine 2 image as 2x1')
        self.m2_1Dlg.grid.addWidget(self.m2_1Dlg.label, 0, 0, 1, 7)

        self.m2_1Dlg.label_2 = QtWidgets.QLabel(self)
        self.m2_1Dlg.label_2.setText('Image_1 : ')
        self.m2_1Dlg.grid.addWidget(self.m2_1Dlg.label_2, 1, 0, 1, 1)

        self.m2_1Dlg.label_3 = QtWidgets.QLabel(self)
        self.m2_1Dlg.label_3.setText('Image_2 : ')
        self.m2_1Dlg.grid.addWidget(self.m2_1Dlg.label_3, 1, 2, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m2_1Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m2_1Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m2_1Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m2_1Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        self.m2_1Dlg.label_4 = QtWidgets.QLabel(self)
        self.m2_1Dlg.label_4.setText('')
        self.m2_1Dlg.grid.addWidget(self.m2_1Dlg.label_4, 4, 0, 1, 1)

        btn_3 = QtWidgets.QPushButton('OK', self)
        btn_3.clicked.connect(self.previewMerge_2x1)
        btn_3.resize(btn_3.sizeHint())
        self.m2_1Dlg.grid.addWidget(btn_3, 5, 1, 1, 1)

        self.m2_1Dlg.show()

    def previewMerge_2x1(self):
        self.m2_1Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.preview= self.hconcat_resize([self.image_1, self.image_2])

        self.previewMerge()

    def hconcat_resize(self,img_list,interpolation=cv2.INTER_CUBIC):
        h_min=min(img.shape[0] for img in img_list)
        im_list_resize=[cv2.resize(img,(int(img.shape[1]*h_min/img.shape[0]),h_min), interpolation=interpolation)
                         for img in img_list]
        return cv2.hconcat(im_list_resize)

    def merge_2x2(self):
        self.m2_2Dlg = QMainWindow(self)
        self.m2_2Dlg.setWindowTitle('Collage 2x2')

        self.m2_2Dlg.central_widget = QtWidgets.QWidget()
        self.m2_2Dlg.setCentralWidget(self.m2_2Dlg.central_widget)
        self.m2_2Dlg.grid = QtWidgets.QGridLayout(self.m2_2Dlg.central_widget)

        self.m2_2Dlg.label = QtWidgets.QLabel(self)
        self.m2_2Dlg.label.setStyleSheet("font : 20pt")
        self.m2_2Dlg.label.setText('Combine 4 image as 2x2')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label, 0, 0, 1, 7)

        self.m2_2Dlg.label_2 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_2.setText('Image_1 : ')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_2, 1, 0, 1, 1)

        self.m2_2Dlg.label_3 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_3.setText('Image_2 : ')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_3, 1, 2, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m2_2Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m2_2Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m2_2Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m2_2Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        self.m2_2Dlg.label_4 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_4.setText('')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_4, 4, 0, 1, 1)

        self.m2_2Dlg.label_5 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_5.setText('Image_3 : ')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_5, 5, 0, 1, 1)

        self.m2_2Dlg.label_6 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_6.setText('Image_4 : ')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_6, 5, 2, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m2_2Dlg.grid.addWidget(self.img_3, 6, 0, 1, 1)

        self.img_4 = QtWidgets.QLineEdit(self)
        self.m2_2Dlg.grid.addWidget(self.img_4, 6, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m2_2Dlg.grid.addWidget(btn_3, 7, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('Add', self)
        btn_4.clicked.connect(self.addImage_4)
        btn_4.resize(btn_4.sizeHint())
        self.m2_2Dlg.grid.addWidget(btn_4, 7, 2, 1, 1)

        self.m2_2Dlg.label_7 = QtWidgets.QLabel(self)
        self.m2_2Dlg.label_7.setText('')
        self.m2_2Dlg.grid.addWidget(self.m2_2Dlg.label_7, 8, 0, 1, 1)

        btn_5 = QtWidgets.QPushButton('OK', self)
        btn_5.clicked.connect(self.previewMerge_2x2)
        btn_5.resize(btn_5.sizeHint())
        self.m2_2Dlg.grid.addWidget(btn_5, 9, 1, 1, 1)

        self.m2_2Dlg.show()

    def addImage_4(self):
        self.name_4 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_4.setText(self.name_4)

    def previewMerge_2x2(self):
        self.m2_2Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.image_4 = cv2.imread(self.name_4)
        h1 = self.hconcat_resize([self.image_1, self.image_2])
        h2 = self.hconcat_resize([self.image_3, self.image_4])
        v = self.vconcat_resize([h1, h2])
        self.preview = v

        self.previewMerge()

    def merge_2x3(self):
        self.m2_3Dlg = QMainWindow(self)
        self.m2_3Dlg.setWindowTitle('Collage 2x3')

        self.m2_3Dlg.central_widget = QtWidgets.QWidget()
        self.m2_3Dlg.setCentralWidget(self.m2_3Dlg.central_widget)
        self.m2_3Dlg.grid = QtWidgets.QGridLayout(self.m2_3Dlg.central_widget)

        self.m2_3Dlg.label = QtWidgets.QLabel(self)
        self.m2_3Dlg.label.setStyleSheet("font : 20pt")
        self.m2_3Dlg.label.setText('Combine 6 image as 2x3')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label, 0, 0, 1, 7)

        self.m2_3Dlg.label_2 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_2.setText('Image_1 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_2, 1, 0, 1, 1)

        self.m2_3Dlg.label_3 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_3.setText('Image_2 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_3, 1, 2, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        self.m2_3Dlg.label_4 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_4.setText('')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_4, 4, 0, 1, 1)

        self.m2_3Dlg.label_5 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_5.setText('Image_3 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_5, 5, 0, 1, 1)

        self.m2_3Dlg.label_6 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_6.setText('Image_4 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_6, 5, 2, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_3, 6, 0, 1, 1)

        self.img_4 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_4, 6, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_3, 7, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('Add', self)
        btn_4.clicked.connect(self.addImage_4)
        btn_4.resize(btn_4.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_4, 7, 2, 1, 1)

        self.m2_3Dlg.label_7 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_7.setText('Image_5 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_7, 8, 0, 1, 1)

        self.m2_3Dlg.label_8 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_8.setText('Image_6 : ')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_8, 8, 2, 1, 1)

        self.img_5 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_5, 9, 0, 1, 1)

        self.img_6 = QtWidgets.QLineEdit(self)
        self.m2_3Dlg.grid.addWidget(self.img_6, 9, 2, 1, 1)

        btn_5 = QtWidgets.QPushButton('Add', self)
        btn_5.clicked.connect(self.addImage_5)
        btn_5.resize(btn_5.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_5, 10, 0, 1, 1)

        btn_6 = QtWidgets.QPushButton('Add', self)
        btn_6.clicked.connect(self.addImage_6)
        btn_6.resize(btn_6.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_6, 10, 2, 1, 1)

        self.m2_3Dlg.label_9 = QtWidgets.QLabel(self)
        self.m2_3Dlg.label_9.setText('')
        self.m2_3Dlg.grid.addWidget(self.m2_3Dlg.label_9, 11, 0, 1, 1)

        btn_7 = QtWidgets.QPushButton('OK', self)
        btn_7.clicked.connect(self.previewMerge_2x3)
        btn_7.resize(btn_7.sizeHint())
        self.m2_3Dlg.grid.addWidget(btn_7, 12, 1, 1, 1)

        self.m2_3Dlg.show()

    def addImage_5(self):
        self.name_5 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_5.setText(self.name_5)

    def addImage_6(self):
        self.name_6 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_6.setText(self.name_6)

    def previewMerge_2x3(self):
        self.m2_3Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.image_4 = cv2.imread(self.name_4)
        self.image_5 = cv2.imread(self.name_5)
        self.image_6 = cv2.imread(self.name_6)
        h1 = self.hconcat_resize([self.image_1, self.image_2])
        h2 = self.hconcat_resize([self.image_3, self.image_4])
        h3 = self.hconcat_resize([self.image_5, self.image_6])
        v = self.vconcat_resize([h1, h2, h3])
        self.preview = v

        self.previewMerge()

    def merge_3x1(self):
        self.m3_1Dlg = QMainWindow(self)
        self.m3_1Dlg.setWindowTitle('Collage 3x1')

        self.m3_1Dlg.central_widget = QtWidgets.QWidget()
        self.m3_1Dlg.setCentralWidget(self.m3_1Dlg.central_widget)
        self.m3_1Dlg.grid = QtWidgets.QGridLayout(self.m3_1Dlg.central_widget)

        self.m3_1Dlg.label = QtWidgets.QLabel(self)
        self.m3_1Dlg.label.setStyleSheet("font : 20pt")
        self.m3_1Dlg.label.setText('Combine 3 image as 3x1')
        self.m3_1Dlg.grid.addWidget(self.m3_1Dlg.label, 0, 0, 1, 7)

        self.m3_1Dlg.label_2 = QtWidgets.QLabel(self)
        self.m3_1Dlg.label_2.setText('Image_1 : ')
        self.m3_1Dlg.grid.addWidget(self.m3_1Dlg.label_2, 1, 0, 1, 1)

        self.m3_1Dlg.label_3 = QtWidgets.QLabel(self)
        self.m3_1Dlg.label_3.setText('Image_2 : ')
        self.m3_1Dlg.grid.addWidget(self.m3_1Dlg.label_3, 1, 2, 1, 1)

        self.m3_1Dlg.label_4 = QtWidgets.QLabel(self)
        self.m3_1Dlg.label_4.setText('Image_3 : ')
        self.m3_1Dlg.grid.addWidget(self.m3_1Dlg.label_4, 1, 4, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m3_1Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m3_1Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m3_1Dlg.grid.addWidget(self.img_3, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m3_1Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m3_1Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m3_1Dlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.m3_1Dlg.label_5 = QtWidgets.QLabel(self)
        self.m3_1Dlg.label_5.setText('')
        self.m3_1Dlg.grid.addWidget(self.m3_1Dlg.label_5, 4, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('OK', self)
        btn_4.clicked.connect(self.previewMerge_3x1)
        btn_4.resize(btn_4.sizeHint())
        self.m3_1Dlg.grid.addWidget(btn_4, 5, 2, 1, 1)

        self.m3_1Dlg.show()

    def previewMerge_3x1(self):
        self.m3_1Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.preview = self.hconcat_resize([self.image_1, self.image_2, self.image_3])

        self.previewMerge()

    def merge_3x2(self):
        self.m3_2Dlg = QMainWindow(self)
        self.m3_2Dlg.setWindowTitle('Collage 3x2')

        self.m3_2Dlg.central_widget = QtWidgets.QWidget()
        self.m3_2Dlg.setCentralWidget(self.m3_2Dlg.central_widget)
        self.m3_2Dlg.grid = QtWidgets.QGridLayout(self.m3_2Dlg.central_widget)

        self.m3_2Dlg.label = QtWidgets.QLabel(self)
        self.m3_2Dlg.label.setStyleSheet("font : 20pt")
        self.m3_2Dlg.label.setText('Combine 6 image as 3x2')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label, 0, 0, 1, 7)

        self.m3_2Dlg.label_2 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_2.setText('Image_1 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_2, 1, 0, 1, 1)

        self.m3_2Dlg.label_3 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_3.setText('Image_2 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_3, 1, 2, 1, 1)

        self.m3_2Dlg.label_4 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_4.setText('Image_3 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_4, 1, 4, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_3, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.m3_2Dlg.label_5 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_5.setText('')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_5, 4, 0, 1, 1)

        self.m3_2Dlg.label_6 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_6.setText('Image_4 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_6, 5, 0, 1, 1)

        self.m3_2Dlg.label_7 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_7.setText('Image_5 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_7, 5, 2, 1, 1)

        self.m3_2Dlg.label_8 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_8.setText('Image_6 : ')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_8, 5, 4, 1, 1)

        self.img_4 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_4, 6, 0, 1, 1)

        self.img_5 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_5, 6, 2, 1, 1)

        self.img_6 = QtWidgets.QLineEdit(self)
        self.m3_2Dlg.grid.addWidget(self.img_6, 6, 4, 1, 1)

        btn_4 = QtWidgets.QPushButton('Add', self)
        btn_4.clicked.connect(self.addImage_4)
        btn_4.resize(btn_4.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_4, 7, 0, 1, 1)

        btn_5 = QtWidgets.QPushButton('Add', self)
        btn_5.clicked.connect(self.addImage_5)
        btn_5.resize(btn_5.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_5, 7, 2, 1, 1)

        btn_6 = QtWidgets.QPushButton('Add', self)
        btn_6.clicked.connect(self.addImage_6)
        btn_6.resize(btn_6.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_6, 7, 4, 1, 1)

        self.m3_2Dlg.label_9 = QtWidgets.QLabel(self)
        self.m3_2Dlg.label_9.setText('')
        self.m3_2Dlg.grid.addWidget(self.m3_2Dlg.label_9, 8, 0, 1, 1)

        btn_7 = QtWidgets.QPushButton('OK', self)
        btn_7.clicked.connect(self.previewMerge_3x2)
        btn_7.resize(btn_7.sizeHint())
        self.m3_2Dlg.grid.addWidget(btn_7, 13, 2, 1, 1)

        self.m3_2Dlg.show()

    def previewMerge_3x2(self):
        self.m3_2Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.image_4 = cv2.imread(self.name_4)
        self.image_5 = cv2.imread(self.name_5)
        self.image_6 = cv2.imread(self.name_6)
        h1 = self.hconcat_resize([self.image_1, self.image_2, self.image_3])
        h2 = self.hconcat_resize([self.image_4, self.image_5, self.image_6])
        v = self.vconcat_resize([h1, h2])
        self.preview = v

        self.previewMerge()

    def merge_3x3(self):
        self.m3_3Dlg = QMainWindow(self)
        self.m3_3Dlg.setWindowTitle('Collage 3x3')

        self.m3_3Dlg.central_widget = QtWidgets.QWidget()
        self.m3_3Dlg.setCentralWidget(self.m3_3Dlg.central_widget)
        self.m3_3Dlg.grid = QtWidgets.QGridLayout(self.m3_3Dlg.central_widget)

        self.m3_3Dlg.label = QtWidgets.QLabel(self)
        self.m3_3Dlg.label.setStyleSheet("font : 20pt")
        self.m3_3Dlg.label.setText('Combine 9 image as 3x3')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label, 0, 0, 1, 7)

        self.m3_3Dlg.label_2 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_2.setText('Image_1 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_2, 1, 0, 1, 1)

        self.m3_3Dlg.label_3 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_3.setText('Image_2 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_3, 1, 2, 1, 1)

        self.m3_3Dlg.label_4 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_4.setText('Image_3 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_4, 1, 4, 1, 1)

        self.img_1 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_1, 2, 0, 1, 1)

        self.img_2 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_2, 2, 2, 1, 1)

        self.img_3 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_3, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Add', self)
        btn_1.clicked.connect(self.addImage_1)
        btn_1.resize(btn_1.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Add', self)
        btn_2.clicked.connect(self.addImage_2)
        btn_2.resize(btn_2.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Add', self)
        btn_3.clicked.connect(self.addImage_3)
        btn_3.resize(btn_3.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.m3_3Dlg.label_5 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_5.setText('')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_5, 4, 0, 1, 1)

        self.m3_3Dlg.label_6 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_6.setText('Image_4 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_6, 5, 0, 1, 1)

        self.m3_3Dlg.label_7 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_7.setText('Image_5 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_7, 5, 2, 1, 1)

        self.m3_3Dlg.label_8 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_8.setText('Image_6 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_8, 5, 4, 1, 1)

        self.img_4 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_4, 6, 0, 1, 1)

        self.img_5 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_5, 6, 2, 1, 1)

        self.img_6 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_6, 6, 4, 1, 1)

        btn_4 = QtWidgets.QPushButton('Add', self)
        btn_4.clicked.connect(self.addImage_4)
        btn_4.resize(btn_4.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_4, 7, 0, 1, 1)

        btn_5 = QtWidgets.QPushButton('Add', self)
        btn_5.clicked.connect(self.addImage_5)
        btn_5.resize(btn_5.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_5, 7, 2, 1, 1)

        btn_6 = QtWidgets.QPushButton('Add', self)
        btn_6.clicked.connect(self.addImage_6)
        btn_6.resize(btn_6.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_6, 7, 4, 1, 1)

        self.m3_3Dlg.label_9 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_9.setText('')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_9, 8, 0, 1, 1)

        self.m3_3Dlg.label_10 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_10.setText('Image_7 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_10, 9, 0, 1, 1)

        self.m3_3Dlg.label_11 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_11.setText('Image_8 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_11, 9, 2, 1, 1)

        self.m3_3Dlg.label_12 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_12.setText('Image_9 : ')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_12, 9, 4, 1, 1)

        self.img_7 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_7, 10, 0, 1, 1)

        self.img_8 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_8, 10, 2, 1, 1)

        self.img_9 = QtWidgets.QLineEdit(self)
        self.m3_3Dlg.grid.addWidget(self.img_9, 10, 4, 1, 1)

        btn_7 = QtWidgets.QPushButton('Add', self)
        btn_7.clicked.connect(self.addImage_7)
        btn_7.resize(btn_7.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_7, 11, 0, 1, 1)

        btn_8 = QtWidgets.QPushButton('Add', self)
        btn_8.clicked.connect(self.addImage_8)
        btn_8.resize(btn_8.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_8, 11, 2, 1, 1)

        btn_9 = QtWidgets.QPushButton('Add', self)
        btn_9.clicked.connect(self.addImage_9)
        btn_9.resize(btn_6.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_9, 11, 4, 1, 1)

        self.m3_3Dlg.label_13 = QtWidgets.QLabel(self)
        self.m3_3Dlg.label_13.setText('')
        self.m3_3Dlg.grid.addWidget(self.m3_3Dlg.label_13, 12, 0, 1, 1)

        btn_10 = QtWidgets.QPushButton('OK', self)
        btn_10.clicked.connect(self.previewMerge_3x3)
        btn_10.resize(btn_10.sizeHint())
        self.m3_3Dlg.grid.addWidget(btn_10, 13, 2, 1, 1)

        self.m3_3Dlg.show()

    def addImage_7(self):
        self.name_7 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_7.setText(self.name_7)

    def addImage_8(self):
        self.name_8 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_8.setText(self.name_8)

    def addImage_9(self):
        self.name_9 = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.img_9.setText(self.name_9)

    def previewMerge_3x3(self):
        self.m3_3Dlg.close()

        self.image_1 = cv2.imread(self.name_1)
        self.image_2 = cv2.imread(self.name_2)
        self.image_3 = cv2.imread(self.name_3)
        self.image_4 = cv2.imread(self.name_4)
        self.image_5 = cv2.imread(self.name_5)
        self.image_6 = cv2.imread(self.name_6)
        self.image_7 = cv2.imread(self.name_7)
        self.image_8 = cv2.imread(self.name_8)
        self.image_9 = cv2.imread(self.name_9)
        h1 = self.hconcat_resize([self.image_1, self.image_2, self.image_3])
        h2 = self.hconcat_resize([self.image_4, self.image_5, self.image_6])
        h3 = self.hconcat_resize([self.image_7, self.image_8, self.image_9])
        v = self.vconcat_resize([h1, h2, h3])
        self.preview = v

        self.previewMerge()

    def gHistDialog(self):
        self.hDlg = QMainWindow(self)
        self.hDlg.setWindowTitle('Histogram')

        self.hDlg.central_widget = QtWidgets.QWidget()
        self.hDlg.setCentralWidget(self.hDlg.central_widget)
        self.hDlg.grid = QtWidgets.QGridLayout(self.hDlg.central_widget)

        self.hDlg.label = QtWidgets.QLabel(self)
        self.hDlg.label.setStyleSheet("font : 20pt")
        self.hDlg.label.setText('Histogram equalization for gray scale image')
        self.hDlg.grid.addWidget(self.hDlg.label,0,0,1,5)

        self.hDlg.label_2 = QtWidgets.QLabel(self)
        self.hDlg.label_2.setText('Before :')
        self.hDlg.grid.addWidget(self.hDlg.label_2,1,0,1,1)

        self.hDlg.label_3 = QtWidgets.QLabel(self)
        self.hDlg.label_3.setText('After :')
        self.hDlg.grid.addWidget(self.hDlg.label_3, 1, 2, 1, 1)

        self.img_b4=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        self.img_aft=cv2.equalizeHist(self.img_b4)

        self.hDlg.label_4 = QtWidgets.QLabel(self)
        b4_image = QImage(self.img_b4, self.img_b4.shape[1], self.img_b4.shape[0], self.img_b4.shape[1],
                          QImage.Format_Grayscale8)
        pixmap = QPixmap(b4_image)
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.hDlg.label_4.setPixmap(pixmap)
        self.hDlg.grid.addWidget(self.hDlg.label_4,2,0,1,1)

        self.hDlg.label_5 = QtWidgets.QLabel(self)
        aft_image = QImage(self.img_aft, self.img_aft.shape[1], self.img_aft.shape[0], self.img_aft.shape[1],
                           QImage.Format_Grayscale8)
        pixmap2 = QPixmap(aft_image)
        pixmap2 = pixmap2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.hDlg.label_5.setPixmap(pixmap2)
        self.hDlg.grid.addWidget(self.hDlg.label_5, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Show Graph', self)
        btn_1.clicked.connect(self.openGrayHist)
        btn_1.resize(btn_1.sizeHint())
        self.hDlg.grid.addWidget(btn_1, 3, 1, 1, 1)

        self.hDlg.label_6 = QtWidgets.QLabel(self)
        self.hDlg.label_6.setText('(red line = before, blue line = after)')
        self.hDlg.grid.addWidget(self.hDlg.label_6, 4, 1, 1, 1)

        self.hDlg.label_7 = QtWidgets.QLabel(self)
        self.hDlg.label_7.setText('')
        self.hDlg.grid.addWidget(self.hDlg.label_7, 5, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Cancel', self)
        btn_2.clicked.connect(self.closeGrayHist)
        btn_2.resize(btn_2.sizeHint())
        self.hDlg.grid.addWidget(btn_2,6,0,1,1)

        btn_3 = QtWidgets.QPushButton('Apply', self)
        btn_3.clicked.connect(self.grayHistEqua)
        btn_3.resize(btn_3.sizeHint())
        self.hDlg.grid.addWidget(btn_3,6,2,1,1)

        self.hDlg.show()

    def openGrayHist(self):
        hist = cv2.calcHist(self.img_b4, [0], None, [256], [0, 256])
        plt.plot(hist, color='r')
        hist2 = cv2.calcHist(self.img_aft, [0], None, [256], [0, 256])
        plt.plot(hist2,color='b')
        plt.show()

    def closeGrayHist(self):
        self.hDlg.close()

    def grayHistEqua(self):
        self.hDlg.close()

        self.image=self.img_aft
        self.updateGray()

    def colorHistDialog(self):
        self.chDlg = QMainWindow(self)
        self.chDlg.setWindowTitle('Histogram')

        self.chDlg.central_widget = QtWidgets.QWidget()
        self.chDlg.setCentralWidget(self.chDlg.central_widget)
        self.chDlg.grid = QtWidgets.QGridLayout(self.chDlg.central_widget)

        self.chDlg.label = QtWidgets.QLabel(self)
        self.chDlg.label.setStyleSheet("font : 20pt")
        self.chDlg.label.setText('Histogram equalization for color image')
        self.chDlg.grid.addWidget(self.chDlg.label,0,0,1,5)

        self.chDlg.label_2 = QtWidgets.QLabel(self)
        self.chDlg.label_2.setText('Before :')
        self.chDlg.grid.addWidget(self.chDlg.label_2,1,0,1,1)

        self.chDlg.label_3 = QtWidgets.QLabel(self)
        self.chDlg.label_3.setText('After :')
        self.chDlg.grid.addWidget(self.chDlg.label_3, 1, 2, 1, 1)

        channels=cv2.split(self.image)
        eq_channels=[]
        for ch,color in zip(channels,['B','G','R']):
            eq_channels.append(cv2.equalizeHist(ch))
        eq_image=cv2.merge(eq_channels)
        self.img_aft=cv2.cvtColor(eq_image,cv2.COLOR_BGR2RGB)

        self.chDlg.label_4 = QtWidgets.QLabel(self)
        pixmap = self.pixmap
        pixmap = pixmap.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.chDlg.label_4.setPixmap(pixmap)
        self.chDlg.grid.addWidget(self.chDlg.label_4,2,0,1,1)

        self.chDlg.label_5 = QtWidgets.QLabel(self)
        aft_image = QImage(self.img_aft, self.img_aft.shape[1], self.img_aft.shape[0], self.img_aft.shape[1]*3,
                           QImage.Format_RGB888).rgbSwapped()
        pixmap2 = QPixmap(aft_image)
        pixmap2 = pixmap2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.chDlg.label_5.setPixmap(pixmap2)
        self.chDlg.grid.addWidget(self.chDlg.label_5, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Show Graph Before', self)
        btn_1.clicked.connect(self.openHistB4)
        btn_1.resize(btn_1.sizeHint())
        self.chDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Show Graph After', self)
        btn_2.clicked.connect(self.openHistAft)
        btn_2.resize(btn_2.sizeHint())
        self.chDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        self.chDlg.label_6 = QtWidgets.QLabel(self)
        self.chDlg.label_6.setText('')
        self.chDlg.grid.addWidget(self.chDlg.label_6, 4, 0, 1, 1)

        btn_3 = QtWidgets.QPushButton('Cancel', self)
        btn_3.clicked.connect(self.closeColorHist)
        btn_3.resize(btn_3.sizeHint())
        self.chDlg.grid.addWidget(btn_3,5,0,1,1)

        btn_4 = QtWidgets.QPushButton('Apply', self)
        btn_4.clicked.connect(self.colorHistEqua)
        btn_4.resize(btn_4.sizeHint())
        self.chDlg.grid.addWidget(btn_4,5,2,1,1)

        self.chDlg.show()

    def openHistB4(self):
        channels=('b','g','r')
        for i,col in enumerate(channels):
            hist=cv2.calcHist([self.image],[i],None,[256],[0,256])
            plt.plot(hist,color=col)
        plt.show()

    def openHistAft(self):
        channels = ('b', 'g', 'r')
        for i, col in enumerate(channels):
            hist = cv2.calcHist([self.img_aft], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
        plt.show()

    def closeColorHist(self):
        self.chDlg.close()

    def colorHistEqua(self):
        self.chDlg.close()

        self.image = self.img_aft
        self.updateColor()

    def kernelSize_3(self):
        self.kernel_size=3
        self.blurDetail()

    def kernelSize_5(self):
        self.kernel_size=5
        self.blurDetail()

    def kernelSize_7(self):
        self.kernel_size=7
        self.blurDetail()

    def blurDetail(self):
        self.sigmaX = 0
        self.sigmaColor = 75
        self.sigmaSpace = 75
        self.blurDialog()

    def kernelSize_c(self):
        self.ksDlg = QMainWindow(self)
        self.ksDlg.setWindowTitle('Blur')

        self.ksDlg.central_widget = QtWidgets.QWidget()
        self.ksDlg.setCentralWidget(self.ksDlg.central_widget)
        self.ksDlg.grid = QtWidgets.QGridLayout(self.ksDlg.central_widget)

        self.ksDlg.label = QtWidgets.QLabel(self)
        self.ksDlg.label.setStyleSheet("font : 20pt")
        self.ksDlg.label.setText('Blurring filter')
        self.ksDlg.grid.addWidget(self.ksDlg.label, 0, 0, 1, 4)

        self.ksDlg.label_2 = QtWidgets.QLabel(self)
        self.ksDlg.label_2.setText('kernel size = ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_2, 1, 0, 1, 1)

        self.kernel_input = QtWidgets.QLineEdit(self)
        self.ksDlg.grid.addWidget(self.kernel_input, 1, 1, 1, 4)

        self.ksDlg.label_3 = QtWidgets.QLabel(self)
        self.ksDlg.label_3.setText('(enter a positive odd integer more than 2) \n(system automatic +1 if enter an even number)')
        self.ksDlg.grid.addWidget(self.ksDlg.label_3, 2, 1, 1, 4)

        self.ksDlg.label_4 = QtWidgets.QLabel(self)
        self.ksDlg.label_4.setText(' ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_4, 3, 0, 1, 1)

        self.ksDlg.label_5 = QtWidgets.QLabel(self)
        self.ksDlg.label_5.setText('Gaussian blur : ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_5, 4, 0, 1, 1)

        self.ksDlg.label_6 = QtWidgets.QLabel(self)
        self.ksDlg.label_6.setText('SigmaX = ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_6, 5, 0, 1, 1)

        self.std = QtWidgets.QLineEdit(self)
        self.ksDlg.grid.addWidget(self.std, 5, 1, 1, 4)

        self.ksDlg.label_7 = QtWidgets.QLabel(self)
        self.ksDlg.label_7.setText(' ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_7, 6, 0, 1, 4)

        self.ksDlg.label_8 = QtWidgets.QLabel(self)
        self.ksDlg.label_8.setText('Bilateral filter : ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_8, 7, 0, 1, 1)

        self.ksDlg.label_9 = QtWidgets.QLabel(self)
        self.ksDlg.label_9.setText('SigmaColor = ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_9, 8, 0, 1, 1)

        self.bilateralColor = QtWidgets.QLineEdit(self)
        self.ksDlg.grid.addWidget(self.bilateralColor, 8, 1, 1, 4)

        self.ksDlg.label_10 = QtWidgets.QLabel(self)
        self.ksDlg.label_10.setText('SigmaSpace = ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_10, 9, 0, 1, 1)

        self.bilateralSpace = QtWidgets.QLineEdit(self)
        self.ksDlg.grid.addWidget(self.bilateralSpace, 9, 1, 1, 4)

        self.ksDlg.label_11 = QtWidgets.QLabel(self)
        self.ksDlg.label_11.setText(' ')
        self.ksDlg.grid.addWidget(self.ksDlg.label_11, 10, 1, 1, 4)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomKernel)
        btn_1.resize(btn_1.sizeHint())
        self.ksDlg.grid.addWidget(btn_1, 11, 1, 1, 1)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.blur_c)
        btn_2.resize(btn_2.sizeHint())
        self.ksDlg.grid.addWidget(btn_2, 11, 3, 1, 1)

        self.ksDlg.show()

    def randomKernel(self):
        num=random.randint(3,15)
        if num%2==1:
            num=num
        else:
            num+=1
        self.kernel_input.setText(str(num))

        self.std.setText(str(random.randint(0,10)))
        self.bilateralColor.setText(str(random.randint(50,100)))
        self.bilateralSpace.setText(self.bilateralColor.text())

    def blur_c(self):
        self.ksDlg.close()

        ksize=int(self.kernel_input.text())
        if ksize<3:
            ksize=3

        if ksize%2==1:
            ksize=ksize
        else:
            ksize+=1
        self.kernel_size=ksize

        self.sigmaX=int(self.std.text())
        self.sigmaColor=int(self.bilateralColor.text())
        self.sigmaSpace=int(self.bilateralSpace.text())

        self.blurDialog()

    def blurDialog(self):
        self.bDlg = QMainWindow(self)
        self.bDlg.setWindowTitle('Blur')

        self.bDlg.central_widget = QtWidgets.QWidget()
        self.bDlg.setCentralWidget(self.bDlg.central_widget)
        self.bDlg.grid = QtWidgets.QGridLayout(self.bDlg.central_widget)

        self.bDlg.label = QtWidgets.QLabel(self)
        self.bDlg.label.setStyleSheet("font : 20pt")
        text='Preview blurring filter by '+str(self.kernel_size)+'x'+str(self.kernel_size)+' kernel'
        self.bDlg.label.setText(text)
        self.bDlg.grid.addWidget(self.bDlg.label, 0, 0, 1, 5)

        self.bDlg.label_2 = QtWidgets.QLabel(self)
        self.bDlg.label_2.setText('Gaussian Blur :')
        self.bDlg.grid.addWidget(self.bDlg.label_2, 1, 0, 1, 1)

        self.bDlg.label_3 = QtWidgets.QLabel(self)
        self.bDlg.label_3.setText('Median Blur :')
        self.bDlg.grid.addWidget(self.bDlg.label_3, 1, 2, 1, 1)

        self.bDlg.label_4 = QtWidgets.QLabel(self)
        self.bDlg.label_4.setText('Average Blur :')
        self.bDlg.grid.addWidget(self.bDlg.label_4, 1, 4, 1, 1)

        self.bDlg.label_5 = QtWidgets.QLabel(self)
        self.bDlg.label_5.setText('Bilateral Filter :')
        self.bDlg.grid.addWidget(self.bDlg.label_5, 1, 6, 1, 1)

        self.bDlg.label_6 = QtWidgets.QLabel(self)
        self.blur_image = cv2.GaussianBlur(self.image, (self.kernel_size,self.kernel_size), self.sigmaX)
        gaussian_img = QImage(self.blur_image, self.blur_image.shape[1], self.blur_image.shape[0], self.blur_image.shape[1] * 3,
                     QImage.Format_RGB888).rgbSwapped()
        pixmap_1 = QPixmap(gaussian_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.bDlg.label_6.setPixmap(pixmap_1)
        self.bDlg.grid.addWidget(self.bDlg.label_6, 2, 0, 1, 1)

        self.bDlg.label_7 = QtWidgets.QLabel(self)
        self.blur_median=cv2.medianBlur(self.image,self.kernel_size)
        median_img = QImage(self.blur_median, self.blur_median.shape[1], self.blur_median.shape[0], self.blur_median.shape[1] * 3,
                              QImage.Format_RGB888).rgbSwapped()
        pixmap_2 = QPixmap(median_img)
        pixmap_2 = pixmap_2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.bDlg.label_7.setPixmap(pixmap_2)
        self.bDlg.grid.addWidget(self.bDlg.label_7, 2, 2, 1, 1)

        self.bDlg.label_8 = QtWidgets.QLabel(self)
        self.blur_average=cv2.blur(self.image,(self.kernel_size,self.kernel_size))
        avg_img = QImage(self.blur_average, self.blur_average.shape[1], self.blur_average.shape[0], self.blur_average.shape[1] * 3,
                            QImage.Format_RGB888).rgbSwapped()
        pixmap_3 = QPixmap(avg_img)
        pixmap_3 = pixmap_3.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.bDlg.label_8.setPixmap(pixmap_3)
        self.bDlg.grid.addWidget(self.bDlg.label_8, 2, 4, 1, 1)

        self.bDlg.label_9 = QtWidgets.QLabel(self)
        self.blur_bilateral=cv2.bilateralFilter(self.image,self.kernel_size*self.kernel_size,self.sigmaColor,self.sigmaSpace)
        bilateral_img = QImage(self.blur_bilateral, self.blur_bilateral.shape[1], self.blur_bilateral.shape[0], self.blur_bilateral.shape[1] * 3,
                            QImage.Format_RGB888).rgbSwapped()
        pixmap_4 = QPixmap(bilateral_img)
        pixmap_4 = pixmap_4.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.bDlg.label_9.setPixmap(pixmap_4)
        self.bDlg.grid.addWidget(self.bDlg.label_9, 2, 6, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply Gaussian Blur', self)
        btn_1.clicked.connect(self.gaussianBlur)
        btn_1.resize(btn_1.sizeHint())
        self.bDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply Median Blur', self)
        btn_2.clicked.connect(self.medianBlur)
        btn_2.resize(btn_2.sizeHint())
        self.bDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Apply Average Blur', self)
        btn_3.clicked.connect(self.avgBlur)
        btn_3.resize(btn_3.sizeHint())
        self.bDlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        btn_4 = QtWidgets.QPushButton('Apply Bilateral', self)
        btn_4.clicked.connect(self.bilateral)
        btn_4.resize(btn_4.sizeHint())
        self.bDlg.grid.addWidget(btn_4, 3, 6, 1, 1)

        self.bDlg.label_10 = QtWidgets.QLabel(self)
        self.bDlg.label_10.setText('')
        self.bDlg.grid.addWidget(self.bDlg.label_10, 4, 0, 1, 1)

        btn_5 = QtWidgets.QPushButton('Cancel', self)
        btn_5.clicked.connect(self.closeBlur)
        btn_5.resize(btn_5.sizeHint())
        self.bDlg.grid.addWidget(btn_5, 5, 2, 1, 4)

        self.bDlg.show()

    def closeBlur(self):
        self.bDlg.close()
        self.updateColor()

    def gaussianBlur(self):
        self.image=self.blur_image
        self.closeBlur()

    def medianBlur(self):
        self.image=self.blur_median
        self.closeBlur()

    def avgBlur(self):
        self.image=self.blur_average
        self.closeBlur()

    def bilateral(self):
        self.image=self.blur_bilateral
        self.closeBlur()

    def contrastDialog(self):
        self.contrastDlg = QMainWindow(self)
        self.contrastDlg.setWindowTitle('Contrast')

        self.contrastDlg.central_widget = QtWidgets.QWidget()
        self.contrastDlg.setCentralWidget(self.contrastDlg.central_widget)
        self.contrastDlg.grid = QtWidgets.QGridLayout(self.contrastDlg.central_widget)

        self.contrastDlg.label = QtWidgets.QLabel(self)
        self.contrastDlg.label.setStyleSheet("font : 20pt")
        self.contrastDlg.label.setText('Contrast of image')
        self.contrastDlg.grid.addWidget(self.contrastDlg.label, 0, 0, 1, 5)

        self.contrastDlg.label_2 = QtWidgets.QLabel(self)
        self.contrastDlg.label_2.setText('alpha = ')
        self.contrastDlg.grid.addWidget(self.contrastDlg.label_2, 1, 0, 1, 1)

        self.alpha = QtWidgets.QLineEdit(self)
        self.contrastDlg.grid.addWidget(self.alpha, 1, 1, 1, 4)

        self.contrastDlg.label_3 = QtWidgets.QLabel(self)
        self.contrastDlg.label_3.setText('(aplha > 1 - high contrast, alpha < 1 - low contrast) ')
        self.contrastDlg.grid.addWidget(self.contrastDlg.label_3, 2, 1, 1, 4)

        self.contrastDlg.label_4 = QtWidgets.QLabel(self)
        self.contrastDlg.label_4.setText(' ')
        self.contrastDlg.grid.addWidget(self.contrastDlg.label_4, 3, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomContrast)
        btn_1.resize(btn_1.sizeHint())
        self.contrastDlg.grid.addWidget(btn_1, 4, 1, 1, 1)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.contrastPreview)
        btn_2.resize(btn_2.sizeHint())
        self.contrastDlg.grid.addWidget(btn_2, 4, 3, 1, 1)

        self.contrastDlg.show()

    def randomContrast(self):
        value="{:.1f}".format(random.uniform(0, 2))
        self.alpha.setText(str(value))

    def contrastPreview(self):
        self.contrastDlg.close()

        self.cpDlg = QMainWindow(self)
        self.cpDlg.setWindowTitle('Contrast')

        self.cpDlg.central_widget = QtWidgets.QWidget()
        self.cpDlg.setCentralWidget(self.cpDlg.central_widget)
        self.cpDlg.vbox = QtWidgets.QVBoxLayout(self.cpDlg.central_widget)

        self.cpDlg.label = QtWidgets.QLabel(self)
        self.cpDlg.label.setStyleSheet("font : 20pt")
        self.cpDlg.label.setText('Preview contrast image')
        self.cpDlg.vbox.addWidget(self.cpDlg.label)

        self.cpDlg.label_2 = QtWidgets.QLabel(self)
        self.constrastI = cv2.addWeighted(self.image, float(self.alpha.text()), np.zeros(self.image.shape, self.image.dtype), 0, 0)
        contrast_img = QImage(self.constrastI, self.constrastI.shape[1], self.constrastI.shape[0],self.constrastI.shape[1] * 3,
                                    QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(contrast_img)
        pixmap = pixmap.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.cpDlg.label_2.setPixmap(pixmap)
        self.cpDlg.vbox.addWidget(self.cpDlg.label_2)

        self.cpDlg.label_3 = QtWidgets.QLabel(self)
        self.cpDlg.label_3.setText('')
        self.cpDlg.vbox.addWidget(self.cpDlg.label_3)

        btn_1 = QtWidgets.QPushButton('Cancel', self)
        btn_1.clicked.connect(self.closeContrast)
        btn_1.resize(btn_1.sizeHint())
        self.cpDlg.vbox.addWidget(btn_1)

        btn_2 = QtWidgets.QPushButton('Apply', self)
        btn_2.clicked.connect(self.displayContrast)
        btn_2.resize(btn_2.sizeHint())
        self.cpDlg.vbox.addWidget(btn_2)

        self.cpDlg.show()

    def closeContrast(self):
        self.cpDlg.close()

    def displayContrast(self):
        self.cpDlg.close()

        self.image=self.constrastI
        self.updateColor()

    def sharpDialog(self):
        self.sDlg = QMainWindow(self)
        self.sDlg.setWindowTitle('Sharpening')

        self.sDlg.central_widget = QtWidgets.QWidget()
        self.sDlg.setCentralWidget(self.sDlg.central_widget)
        self.sDlg.grid = QtWidgets.QGridLayout(self.sDlg.central_widget)

        self.sDlg.label = QtWidgets.QLabel(self)
        self.sDlg.label.setStyleSheet("font : 20pt")
        self.sDlg.label.setText('Preview sharpening image')
        self.sDlg.grid.addWidget(self.sDlg.label, 0, 0, 1, 5)

        self.sDlg.label_2 = QtWidgets.QLabel(self)
        self.sDlg.label_2.setText('3x3 kernel :')
        self.sDlg.grid.addWidget(self.sDlg.label_2, 1, 0, 1, 1)

        self.sDlg.label_3 = QtWidgets.QLabel(self)
        self.sDlg.label_3.setText('5x5 kernel :')
        self.sDlg.grid.addWidget(self.sDlg.label_3, 1, 2, 1, 1)

        self.sDlg.label_4 = QtWidgets.QLabel(self)
        k_3 = np.array([[-1, -1, -1],
                       [-1, 10, -1],
                       [-1, -1, -1]])
        self.sharpened_3 = cv2.filter2D(self.image, -1, k_3)
        sharpened3_img = QImage(self.sharpened_3, self.sharpened_3.shape[1], self.sharpened_3.shape[0],
                                self.sharpened_3.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap_1 = QPixmap(sharpened3_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.sDlg.label_4.setPixmap(pixmap_1)
        self.sDlg.grid.addWidget(self.sDlg.label_4, 2, 0, 1, 1)

        self.sDlg.label_5 = QtWidgets.QLabel(self)
        k_5 = np.array([[-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, 25, -1, -1],
                        [-1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1]])
        self.sharpened_5 = cv2.filter2D(self.image, -1, k_5)
        sharpened5_img = QImage(self.sharpened_5, self.sharpened_5.shape[1], self.sharpened_5.shape[0],
                                self.sharpened_5.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap_2 = QPixmap(sharpened5_img)
        pixmap_2 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.sDlg.label_5.setPixmap(pixmap_2)
        self.sDlg.grid.addWidget(self.sDlg.label_5, 2, 2, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply 3x3 kernel', self)
        btn_1.clicked.connect(self.kernel_3)
        btn_1.resize(btn_1.sizeHint())
        self.sDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply 5x5 kernel', self)
        btn_2.clicked.connect(self.kernel_5)
        btn_2.resize(btn_2.sizeHint())
        self.sDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        self.sDlg.label_6 = QtWidgets.QLabel(self)
        self.sDlg.label_6.setText('')
        self.sDlg.grid.addWidget(self.sDlg.label_6, 4, 0, 1, 1)

        btn_3 = QtWidgets.QPushButton('Cancel', self)
        btn_3.clicked.connect(self.closeSharp)
        btn_3.resize(btn_3.sizeHint())
        self.sDlg.grid.addWidget(btn_3, 5, 1, 1, 1)

        self.sDlg.show()

    def closeSharp(self):
        self.sDlg.close()
        self.updateColor()

    def kernel_3(self):
        self.image=self.sharpened_3
        self.closeSharp()

    def kernel_5(self):
        self.image=self.sharpened_5
        self.closeSharp()

    def sobelDialog(self):
        self.sobelDlg = QMainWindow(self)
        self.sobelDlg.setWindowTitle('Sobel Edge Detection')

        self.sobelDlg.central_widget = QtWidgets.QWidget()
        self.sobelDlg.setCentralWidget(self.sobelDlg.central_widget)
        self.sobelDlg.grid = QtWidgets.QGridLayout(self.sobelDlg.central_widget)

        self.sobelDlg.label = QtWidgets.QLabel(self)
        self.sobelDlg.label.setStyleSheet("font : 20pt")
        self.sobelDlg.label.setText('Preview result Sobel operator in ....')
        self.sobelDlg.grid.addWidget(self.sobelDlg.label, 0, 0, 1, 5)

        self.sobelDlg.label_2 = QtWidgets.QLabel(self)
        self.sobelDlg.label_2.setText('x-axis direction :')
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_2, 1, 0, 1, 1)

        self.sobelDlg.label_3 = QtWidgets.QLabel(self)
        self.sobelDlg.label_3.setText('y-axis direction :')
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_3, 1, 2, 1, 1)

        self.sobelDlg.label_4 = QtWidgets.QLabel(self)
        self.sobelDlg.label_4.setText('x and y direction :')
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_4, 1, 4, 1, 1)

        gray_img= cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        self.sobelDlg.label_5 = QtWidgets.QLabel(self)
        self.sobel_X = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
        self.sobel_X = np.uint8(np.absolute(self.sobel_X))
        sobelX_img = QImage(self.sobel_X, self.sobel_X.shape[1], self.sobel_X.shape[0],self.sobel_X.shape[1],
                            QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(sobelX_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.sobelDlg.label_5.setPixmap(pixmap_1)
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_5, 2, 0, 1, 1)

        self.sobelDlg.label_6 = QtWidgets.QLabel(self)
        self.sobel_Y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
        self.sobel_Y = np.uint8(np.absolute(self.sobel_Y))
        sobelY_img = QImage(self.sobel_Y, self.sobel_Y.shape[1], self.sobel_Y.shape[0],self.sobel_Y.shape[1],
                            QImage.Format_Grayscale8)
        pixmap_2 = QPixmap(sobelY_img)
        pixmap_2 = pixmap_2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.sobelDlg.label_6.setPixmap(pixmap_2)
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_6, 2, 2, 1, 1)

        self.sobelDlg.label_7 = QtWidgets.QLabel(self)
        self.sobelCombined = cv2.bitwise_or(self.sobel_X, self.sobel_Y)
        sobelC_img = QImage(self.sobelCombined, self.sobelCombined.shape[1], self.sobelCombined.shape[0],
                            self.sobelCombined.shape[1], QImage.Format_Grayscale8)
        pixmap_3 = QPixmap(sobelC_img)
        pixmap_3 = pixmap_3.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.sobelDlg.label_7.setPixmap(pixmap_3)
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_7, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply sobelX', self)
        btn_1.clicked.connect(self.sobelX)
        btn_1.resize(btn_1.sizeHint())
        self.sobelDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply sobelY', self)
        btn_2.clicked.connect(self.sobelY)
        btn_2.resize(btn_2.sizeHint())
        self.sobelDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Apply sobel', self)
        btn_3.clicked.connect(self.sobel)
        btn_3.resize(btn_3.sizeHint())
        self.sobelDlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.sobelDlg.label_8 = QtWidgets.QLabel(self)
        self.sobelDlg.label_8.setText('')
        self.sobelDlg.grid.addWidget(self.sobelDlg.label_8, 4, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('Cancel', self)
        btn_4.clicked.connect(self.closeSobel)
        btn_4.resize(btn_4.sizeHint())
        self.sobelDlg.grid.addWidget(btn_4, 5, 2, 1, 1)

        self.sobelDlg.show()

    def closeSobel(self):
        self.sobelDlg.close()

    def sobelX(self):
        self.image=self.sobel_X
        self.closeSobel()
        self.updateGray()

    def sobelY(self):
        self.image=self.sobel_Y
        self.closeSobel()
        self.updateGray()

    def sobel(self):
        self.image=self.sobelCombined
        self.closeSobel()
        self.updateGray()

    def prewittDialog(self):
        self.prewittDlg = QMainWindow(self)
        self.prewittDlg.setWindowTitle('Prewitt Edge Detection')

        self.prewittDlg.central_widget = QtWidgets.QWidget()
        self.prewittDlg.setCentralWidget(self.prewittDlg.central_widget)
        self.prewittDlg.grid = QtWidgets.QGridLayout(self.prewittDlg.central_widget)

        self.prewittDlg.label = QtWidgets.QLabel(self)
        self.prewittDlg.label.setStyleSheet("font : 20pt")
        self.prewittDlg.label.setText('Preview result Prewitt operator in ....')
        self.prewittDlg.grid.addWidget(self.prewittDlg.label, 0, 0, 1, 5)

        self.prewittDlg.label_2 = QtWidgets.QLabel(self)
        self.prewittDlg.label_2.setText('x-axis direction :')
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_2, 1, 0, 1, 1)

        self.prewittDlg.label_3 = QtWidgets.QLabel(self)
        self.prewittDlg.label_3.setText('y-axis direction :')
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_3, 1, 2, 1, 1)

        self.prewittDlg.label_4 = QtWidgets.QLabel(self)
        self.prewittDlg.label_4.setText('x and y direction :')
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_4, 1, 4, 1, 1)

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        self.prewittDlg.label_5 = QtWidgets.QLabel(self)
        k_prewittX = np.array ([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
        self.prewitt_X = cv2.filter2D(gray_img, -1, k_prewittX)
        prewittX_img = QImage(self.prewitt_X, self.prewitt_X.shape[1], self.prewitt_X.shape[0], self.prewitt_X.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(prewittX_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.prewittDlg.label_5.setPixmap(pixmap_1)
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_5, 2, 0, 1, 1)

        self.prewittDlg.label_6 = QtWidgets.QLabel(self)
        k_prewittY = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
        self.prewitt_Y = cv2.filter2D(gray_img, -1, k_prewittY)
        prewittY_img = QImage(self.prewitt_Y, self.prewitt_Y.shape[1], self.prewitt_Y.shape[0], self.prewitt_Y.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_2 = QPixmap(prewittY_img)
        pixmap_2 = pixmap_2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.prewittDlg.label_6.setPixmap(pixmap_2)
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_6, 2, 2, 1, 1)

        self.prewittDlg.label_7 = QtWidgets.QLabel(self)
        self.prewitt_XY = cv2.bitwise_or(self.prewitt_X, self.prewitt_Y)
        prewittC_img = QImage(self.prewitt_XY, self.prewitt_XY.shape[1], self.prewitt_XY.shape[0],
                              self.prewitt_XY.shape[1], QImage.Format_Grayscale8)
        pixmap_3 = QPixmap(prewittC_img)
        pixmap_3 = pixmap_3.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.prewittDlg.label_7.setPixmap(pixmap_3)
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_7, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply prewittX', self)
        btn_1.clicked.connect(self.prewittX)
        btn_1.resize(btn_1.sizeHint())
        self.prewittDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply prewittY', self)
        btn_2.clicked.connect(self.prewittY)
        btn_2.resize(btn_2.sizeHint())
        self.prewittDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Apply prewitt', self)
        btn_3.clicked.connect(self.prewitt)
        btn_3.resize(btn_3.sizeHint())
        self.prewittDlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.prewittDlg.label_8 = QtWidgets.QLabel(self)
        self.prewittDlg.label_8.setText('')
        self.prewittDlg.grid.addWidget(self.prewittDlg.label_8, 4, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('Cancel', self)
        btn_4.clicked.connect(self.closePrewitt)
        btn_4.resize(btn_4.sizeHint())
        self.prewittDlg.grid.addWidget(btn_4, 5, 2, 1, 1)

        self.prewittDlg.show()

    def closePrewitt(self):
        self.prewittDlg.close()

    def prewittX(self):
        self.image=self.prewitt_X
        self.closePrewitt()
        self.updateGray()

    def prewittY(self):
        self.image=self.prewitt_Y
        self.closePrewitt()
        self.updateGray()

    def prewitt(self):
        self.image=self.prewitt_XY
        self.closePrewitt()
        self.updateGray()

    def robertsDialog(self):
        self.robertsDlg = QMainWindow(self)
        self.robertsDlg.setWindowTitle('Roberts Cross Edge Detection')

        self.robertsDlg.central_widget = QtWidgets.QWidget()
        self.robertsDlg.setCentralWidget(self.robertsDlg.central_widget)
        self.robertsDlg.grid = QtWidgets.QGridLayout(self.robertsDlg.central_widget)

        self.robertsDlg.label = QtWidgets.QLabel(self)
        self.robertsDlg.label.setStyleSheet("font : 20pt")
        self.robertsDlg.label.setText('Preview result Roberts cross operator in ....')
        self.robertsDlg.grid.addWidget(self.robertsDlg.label, 0, 0, 1, 5)

        self.robertsDlg.label_2 = QtWidgets.QLabel(self)
        self.robertsDlg.label_2.setText('x-axis direction :')
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_2, 1, 0, 1, 1)

        self.robertsDlg.label_3 = QtWidgets.QLabel(self)
        self.robertsDlg.label_3.setText('y-axis direction :')
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_3, 1, 2, 1, 1)

        self.robertsDlg.label_4 = QtWidgets.QLabel(self)
        self.robertsDlg.label_4.setText('x and y direction :')
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_4, 1, 4, 1, 1)

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)

        self.robertsDlg.label_5 = QtWidgets.QLabel(self)
        k_robertsX = np.array([[1, 0],
                               [0, -1]])
        self.roberts_X = cv2.filter2D(gray_img, -1, k_robertsX)
        robertsX_img = QImage(self.roberts_X, self.roberts_X.shape[1], self.roberts_X.shape[0], self.roberts_X.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(robertsX_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.robertsDlg.label_5.setPixmap(pixmap_1)
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_5, 2, 0, 1, 1)

        self.robertsDlg.label_6 = QtWidgets.QLabel(self)
        k_robertsY = np.array([[0, 1],
                               [-1, 0]])
        self.roberts_Y = cv2.filter2D(gray_img, -1, k_robertsY)
        robertsY_img = QImage(self.roberts_Y, self.roberts_Y.shape[1], self.roberts_Y.shape[0], self.roberts_Y.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_2 = QPixmap(robertsY_img)
        pixmap_2 = pixmap_2.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.robertsDlg.label_6.setPixmap(pixmap_2)
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_6, 2, 2, 1, 1)

        self.robertsDlg.label_7 = QtWidgets.QLabel(self)
        self.roberts_XY = cv2.bitwise_or(self.roberts_X, self.roberts_Y)
        robertsC_img = QImage(self.roberts_XY, self.roberts_XY.shape[1], self.roberts_XY.shape[0],
                              self.roberts_XY.shape[1], QImage.Format_Grayscale8)
        pixmap_3 = QPixmap(robertsC_img)
        pixmap_3 = pixmap_3.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.robertsDlg.label_7.setPixmap(pixmap_3)
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_7, 2, 4, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply robertsX', self)
        btn_1.clicked.connect(self.robertsX)
        btn_1.resize(btn_1.sizeHint())
        self.robertsDlg.grid.addWidget(btn_1, 3, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply robertsY', self)
        btn_2.clicked.connect(self.robertsY)
        btn_2.resize(btn_2.sizeHint())
        self.robertsDlg.grid.addWidget(btn_2, 3, 2, 1, 1)

        btn_3 = QtWidgets.QPushButton('Apply roberts cross', self)
        btn_3.clicked.connect(self.roberts)
        btn_3.resize(btn_3.sizeHint())
        self.robertsDlg.grid.addWidget(btn_3, 3, 4, 1, 1)

        self.robertsDlg.label_8 = QtWidgets.QLabel(self)
        self.robertsDlg.label_8.setText('')
        self.robertsDlg.grid.addWidget(self.robertsDlg.label_8, 4, 0, 1, 1)

        btn_4 = QtWidgets.QPushButton('Cancel', self)
        btn_4.clicked.connect(self.closeRoberts)
        btn_4.resize(btn_4.sizeHint())
        self.robertsDlg.grid.addWidget(btn_4, 5, 2, 1, 1)

        self.robertsDlg.show()

    def closeRoberts(self):
        self.robertsDlg.close()

    def robertsX(self):
        self.image = self.roberts_X
        self.closeRoberts()
        self.updateGray()

    def robertsY(self):
        self.image = self.roberts_Y
        self.closeRoberts()
        self.updateGray()

    def roberts(self):
        self.image = self.roberts_XY
        self.closeRoberts()
        self.updateGray()

    def laplacianDialog(self):
        self.lapDlg = QMainWindow(self)
        self.lapDlg.setWindowTitle('Laplacian Edge Detection')

        self.lapDlg.central_widget = QtWidgets.QWidget()
        self.lapDlg.setCentralWidget(self.lapDlg.central_widget)
        self.lapDlg.grid = QtWidgets.QGridLayout(self.lapDlg.central_widget)

        self.lapDlg.label = QtWidgets.QLabel(self)
        self.lapDlg.label.setStyleSheet("font : 20pt")
        self.lapDlg.label.setText('Preview result: ')
        self.lapDlg.grid.addWidget(self.lapDlg.label, 0, 0, 1, 2)

        self.lapDlg.label_2 = QtWidgets.QLabel(self)
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        self.lap=cv2.Laplacian(gray_img,cv2.CV_64F)
        self.lap = np.uint8(np.absolute(self.lap))
        lap_img = QImage(self.lap, self.lap.shape[1], self.lap.shape[0], self.lap.shape[1],QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(lap_img)
        pixmap_1 = pixmap_1.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.lapDlg.label_2.setPixmap(pixmap_1)
        self.lapDlg.grid.addWidget(self.lapDlg.label_2, 2, 0, 1, 1)

        self.lapDlg.label_3 = QtWidgets.QLabel(self)
        self.lapDlg.label_3.setText('')
        self.lapDlg.grid.addWidget(self.lapDlg.label_3, 3, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply Laplacian', self)
        btn_1.clicked.connect(self.laplacian)
        btn_1.resize(btn_1.sizeHint())
        self.lapDlg.grid.addWidget(btn_1, 4, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Cancel', self)
        btn_2.clicked.connect(self.closeLaplacian)
        btn_2.resize(btn_2.sizeHint())
        self.lapDlg.grid.addWidget(btn_2, 5, 0, 1, 1)

        self.lapDlg.show()

    def closeLaplacian(self):
        self.lapDlg.close()

    def laplacian(self):
        self.image = self.lap
        self.closeLaplacian()
        self.updateGray()

    def cannyDialog(self):
        self.cannyDlg = QMainWindow(self)
        self.cannyDlg.setWindowTitle('Canny Edge Detection')

        self.cannyDlg.central_widget = QtWidgets.QWidget()
        self.cannyDlg.setCentralWidget(self.cannyDlg.central_widget)
        self.cannyDlg.grid = QtWidgets.QGridLayout(self.cannyDlg.central_widget)

        self.cannyDlg.label = QtWidgets.QLabel(self)
        self.cannyDlg.label.setText('Insert threshold value for the hysteresis procedure : ')
        self.cannyDlg.grid.addWidget(self.cannyDlg.label, 0, 0, 1, 5)

        self.cannyDlg.label_4 = QtWidgets.QLabel(self)
        self.cannyDlg.label_4.setText('')
        self.cannyDlg.grid.addWidget(self.cannyDlg.label_4, 1, 0, 1, 1)

        self.cannyDlg.label_2 = QtWidgets.QLabel(self)
        self.cannyDlg.label_2.setText('threshold_1 = ')
        self.cannyDlg.grid.addWidget(self.cannyDlg.label_2, 2, 0, 1, 1)

        self.threshold_1 = QtWidgets.QLineEdit(self)
        self.cannyDlg.grid.addWidget(self.threshold_1, 2, 1, 1, 2)

        self.cannyDlg.label_3 = QtWidgets.QLabel(self)
        self.cannyDlg.label_3.setText('threshold_2 = ')
        self.cannyDlg.grid.addWidget(self.cannyDlg.label_3, 3, 0, 1, 1)

        self.threshold_2 = QtWidgets.QLineEdit(self)
        self.cannyDlg.grid.addWidget(self.threshold_2, 3, 1, 1, 2)

        self.cannyDlg.label_4 = QtWidgets.QLabel(self)
        self.cannyDlg.label_4.setText('')
        self.cannyDlg.grid.addWidget(self.cannyDlg.label_4, 4, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomCanny)
        btn_1.resize(btn_1.sizeHint())
        self.cannyDlg.grid.addWidget(btn_1, 5, 1, 1, 1)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.preCanny)
        btn_2.resize(btn_2.sizeHint())
        self.cannyDlg.grid.addWidget(btn_2, 5, 3, 1, 1)

        self.cannyDlg.show()

    def randomCanny(self):
        self.threshold_1.setText(str(random.randint(0, 128)))
        self.threshold_2.setText(str(random.randint(129, 255)))

    def preCanny(self):
        self.cannyDlg.close()
        self.preCannyDlg = QMainWindow(self)
        self.preCannyDlg.setWindowTitle('Canny Edge Detection')

        self.preCannyDlg.central_widget = QtWidgets.QWidget()
        self.preCannyDlg.setCentralWidget(self.preCannyDlg.central_widget)
        self.preCannyDlg.grid = QtWidgets.QGridLayout(self.preCannyDlg.central_widget)

        self.preCannyDlg.label = QtWidgets.QLabel(self)
        self.preCannyDlg.label.setStyleSheet("font : 20pt")
        self.preCannyDlg.label.setText('Preview result : ')
        self.preCannyDlg.grid.addWidget(self.preCannyDlg.label, 0, 0, 1, 2)

        self.preCannyDlg.label_2 = QtWidgets.QLabel(self)
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        self.canny=cv2.Canny(gray_img,int(self.threshold_1.text()),int(self.threshold_2.text()))
        canny_img = QImage(self.canny, self.canny.shape[1], self.canny.shape[0], self.canny.shape[1],
                           QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(canny_img)
        pixmap_1 = pixmap_1.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.preCannyDlg.label_2.setPixmap(pixmap_1)
        self.preCannyDlg.grid.addWidget(self.preCannyDlg.label_2, 2, 0, 1, 1)

        self.preCannyDlg.label_3 = QtWidgets.QLabel(self)
        self.preCannyDlg.label_3.setText('')
        self.preCannyDlg.grid.addWidget(self.preCannyDlg.label_3, 3, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply Canny', self)
        btn_1.clicked.connect(self.apply_canny)
        btn_1.resize(btn_1.sizeHint())
        self.preCannyDlg.grid.addWidget(btn_1, 4, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Cancel', self)
        btn_2.clicked.connect(self.closeCanny)
        btn_2.resize(btn_2.sizeHint())
        self.preCannyDlg.grid.addWidget(btn_2, 5, 0, 1, 1)

        self.preCannyDlg.show()

    def closeCanny(self):
        self.preCannyDlg.close()

    def apply_canny(self):
        self.image = self.canny
        self.closeCanny()
        self.updateGray()

    def threshDialog(self):
        self.threshDlg = QMainWindow(self)
        self.threshDlg.setWindowTitle('Thresholding')

        self.threshDlg.central_widget = QtWidgets.QWidget()
        self.threshDlg.setCentralWidget(self.threshDlg.central_widget)
        self.threshDlg.grid = QtWidgets.QGridLayout(self.threshDlg.central_widget)

        self.threshDlg.label = QtWidgets.QLabel(self)
        self.threshDlg.label.setText('THRESH_BINARY: If pixel intensity is greater than the set threshold, value set to 255, else set to 0 (black).\n\n'
                                    'THRESH_BINARY_INV: Inverted or Opposite case of THRESH_BINARY.\n\n'
                                    'THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the threshold. \n'
                                    '\t\tThe pixel values are set to be the same as the threshold. All other values remain the same.\n\n'
                                    'THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, less than the threshold value.\n\n'
                                    'THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.')
        self.threshDlg.grid.addWidget(self.threshDlg.label, 0, 0, 6, 7)

        self.threshDlg.label_4 = QtWidgets.QLabel(self)
        self.threshDlg.label_4.setText('')
        self.threshDlg.grid.addWidget(self.threshDlg.label_4, 6, 0, 1, 1)

        self.threshDlg.label_2 = QtWidgets.QLabel(self)
        self.threshDlg.label_2.setText('Threshold value = ')
        self.threshDlg.grid.addWidget(self.threshDlg.label_2, 7, 0, 1, 1)

        self.thresholdValue = QtWidgets.QLineEdit(self)
        self.threshDlg.grid.addWidget(self.thresholdValue, 7, 1, 1, 2)

        self.threshDlg.label_3 = QtWidgets.QLabel(self)
        self.threshDlg.label_3.setText('Thresholding technique = ')
        self.threshDlg.grid.addWidget(self.threshDlg.label_3, 8, 0, 1, 1)

        self.threshDlg.comboBox = QtWidgets.QComboBox(self)
        self.threshDlg.comboBox.addItem("THRESH_BINARY")
        self.threshDlg.comboBox.addItem("THRESH_BINARY_INV")
        self.threshDlg.comboBox.addItem("THRESH_TRUNC")
        self.threshDlg.comboBox.addItem("THRESH_TOZERO")
        self.threshDlg.comboBox.addItem("THRESH_TOZERO_INV")
        self.threshDlg.comboBox.activated[str].connect(self.thresTechnique)
        self.threshDlg.grid.addWidget(self.threshDlg.comboBox, 8, 1, 1, 3)

        self.threshDlg.label_4 = QtWidgets.QLabel(self)
        self.threshDlg.label_4.setText('')
        self.threshDlg.grid.addWidget(self.threshDlg.label_4, 9, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomThreshold)
        btn_1.resize(btn_1.sizeHint())
        self.threshDlg.grid.addWidget(btn_1, 10, 1, 1, 2)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.preThreshold)
        btn_2.resize(btn_2.sizeHint())
        self.threshDlg.grid.addWidget(btn_2, 10, 4, 1, 2)

        self.threshDlg.show()

    def thresTechnique(self,tech):
        if str(tech) == "THRESH_BINARY":
            self.technique=cv2.THRESH_BINARY
        elif str(tech) == "THRESH_BINARY_INV":
            self.technique = cv2.THRESH_BINARY_INV
        elif str(tech) == "THRESH_TRUNC":
            self.technique = cv2.THRESH_TRUNC
        elif str(tech) == "THRESH_TOZERO":
            self.technique = cv2.THRESH_TOZERO
        elif str(tech) == "THRESH_TOZERO_INV":
            self.technique = cv2.THRESH_TOZERO_INV

    def randomThreshold(self):
        self.thresholdValue.setText(str(random.randint(0, 128)))
        self.thresTechnique("THRESH_BINARY")

    def preThreshold(self):
        self.threshDlg.close()
        self.preThreshDlg = QMainWindow(self)
        self.preThreshDlg.setWindowTitle('Thresholding')

        self.preThreshDlg.central_widget = QtWidgets.QWidget()
        self.preThreshDlg.setCentralWidget(self.preThreshDlg.central_widget)
        self.preThreshDlg.grid = QtWidgets.QGridLayout(self.preThreshDlg.central_widget)

        self.preThreshDlg.label = QtWidgets.QLabel(self)
        self.preThreshDlg.label.setStyleSheet("font : 20pt")
        self.preThreshDlg.label.setText('Preview thresholding :')
        self.preThreshDlg.grid.addWidget(self.preThreshDlg.label, 0, 0, 1, 5)

        self.preThreshDlg.label_2 = QtWidgets.QLabel(self)
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGRA2GRAY)
        retval, self.threshold = cv2.threshold(gray_img, int(self.thresholdValue.text()), 255, self.technique)
        thresh_img = QImage(self.threshold, self.threshold.shape[1], self.threshold.shape[0], self.threshold.shape[1],
                           QImage.Format_Grayscale8)
        pixmap_1 = QPixmap(thresh_img)
        pixmap_1 = pixmap_1.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.preThreshDlg.label_2.setPixmap(pixmap_1)
        self.preThreshDlg.grid.addWidget(self.preThreshDlg.label_2, 2, 0, 1, 1)

        self.preThreshDlg.label_3 = QtWidgets.QLabel(self)
        self.preThreshDlg.label_3.setText('')
        self.preThreshDlg.grid.addWidget(self.preThreshDlg.label_3, 3, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply Threshold', self)
        btn_1.clicked.connect(self.applyThresh)
        btn_1.resize(btn_1.sizeHint())
        self.preThreshDlg.grid.addWidget(btn_1, 4, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Cancel', self)
        btn_2.clicked.connect(self.closeThresh)
        btn_2.resize(btn_2.sizeHint())
        self.preThreshDlg.grid.addWidget(btn_2, 5, 0, 1, 1)

        self.preThreshDlg.show()

    def closeThresh(self):
        self.preThreshDlg.close()

    def applyThresh(self):
        self.image = self.threshold
        self.closeThresh()
        self.updateGray()

    def colSegDialog(self):
        self.colSegDlg = QMainWindow(self)
        self.colSegDlg.setWindowTitle('Color Segmentation')

        self.colSegDlg.central_widget = QtWidgets.QWidget()
        self.colSegDlg.setCentralWidget(self.colSegDlg.central_widget)
        self.colSegDlg.grid = QtWidgets.QGridLayout(self.colSegDlg.central_widget)

        self.colSegDlg.label = QtWidgets.QLabel(self)
        self.colSegDlg.label.setStyleSheet("font : 15pt")
        self.colSegDlg.label.setText('Range values for the pixels that want to extract : ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label, 0, 0, 1, 5)

        self.colSegDlg.label_2 = QtWidgets.QLabel(self)
        self.colSegDlg.label_2.setText('Minimum HSV value : ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_2, 1, 0, 1, 2)

        self.colSegDlg.label_3 = QtWidgets.QLabel(self)
        self.colSegDlg.label_3.setText('Hue (max 179) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_3, 2, 0, 1, 2)

        self.min_hue = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.min_hue, 2, 2, 1, 2)

        self.colSegDlg.label_4 = QtWidgets.QLabel(self)
        self.colSegDlg.label_4.setText('Saturation (max 255) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_4, 3, 0, 1, 2)

        self.min_sat = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.min_sat, 3, 2, 1, 2)

        self.colSegDlg.label_5 = QtWidgets.QLabel(self)
        self.colSegDlg.label_5.setText('Value (max 255) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_5, 4, 0, 1, 2)

        self.min_val = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.min_val, 4, 2, 1, 2)

        self.colSegDlg.label_6 = QtWidgets.QLabel(self)
        self.colSegDlg.label_6.setText('')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_6, 5, 0, 1, 1)

        self.colSegDlg.label_7 = QtWidgets.QLabel(self)
        self.colSegDlg.label_7.setText('Maximum HSV value : ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_7, 6, 0, 1, 2)

        self.colSegDlg.label_8 = QtWidgets.QLabel(self)
        self.colSegDlg.label_8.setText('Hue (max 179) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_8, 7, 0, 1, 2)

        self.max_hue = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.max_hue, 7, 2, 1, 2)

        self.colSegDlg.label_9 = QtWidgets.QLabel(self)
        self.colSegDlg.label_9.setText('Saturation (max 255) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_9, 8, 0, 1, 2)

        self.max_sat = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.max_sat, 8, 2, 1, 2)

        self.colSegDlg.label_10 = QtWidgets.QLabel(self)
        self.colSegDlg.label_10.setText('Value (max 255) = ')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_10, 9, 0, 1, 2)

        self.max_val = QtWidgets.QLineEdit(self)
        self.colSegDlg.grid.addWidget(self.max_val, 9, 2, 1, 2)

        self.colSegDlg.label_11 = QtWidgets.QLabel(self)
        self.colSegDlg.label_11.setText('')
        self.colSegDlg.grid.addWidget(self.colSegDlg.label_11, 10, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomColSeg)
        btn_1.resize(btn_1.sizeHint())
        self.colSegDlg.grid.addWidget(btn_1, 11, 2, 1, 1)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.preColSeg)
        btn_2.resize(btn_2.sizeHint())
        self.colSegDlg.grid.addWidget(btn_2, 11, 4, 1, 1)

        self.colSegDlg.show()

    def randomColSeg(self):
        chooseCol=random.randint(1,4)
        if chooseCol==1:
            # red color
            self.min_hue.setText("116")
            self.min_sat.setText("155")
            self.min_val.setText("84")
            self.max_hue.setText("179")
            self.max_sat.setText("255")
            self.max_val.setText("255")
        elif chooseCol==2:
            # blue color
            self.min_hue.setText("94")
            self.min_sat.setText("80")
            self.min_val.setText("2")
            self.max_hue.setText("126")
            self.max_sat.setText("255")
            self.max_val.setText("255")
        elif chooseCol==3:
            # green color
            self.min_hue.setText("25")
            self.min_sat.setText("52")
            self.min_val.setText("72")
            self.max_hue.setText("102")
            self.max_sat.setText("255")
            self.max_val.setText("255")
        elif chooseCol==4:
            # every color except white
            self.min_hue.setText("0")
            self.min_sat.setText("42")
            self.min_val.setText("0")
            self.max_hue.setText("179")
            self.max_sat.setText("255")
            self.max_val.setText("255")

    def preColSeg(self):
        self.colSegDlg.close()
        self.preColSeg = QMainWindow(self)
        self.preColSeg.setWindowTitle('Color Segmentation')

        self.preColSeg.central_widget = QtWidgets.QWidget()
        self.preColSeg.setCentralWidget(self.preColSeg.central_widget)
        self.preColSeg.grid = QtWidgets.QGridLayout(self.preColSeg.central_widget)

        self.preColSeg.label = QtWidgets.QLabel(self)
        self.preColSeg.label.setStyleSheet("font : 20pt")
        self.preColSeg.label.setText('Preview Color Segmentation : ')
        self.preColSeg.grid.addWidget(self.preColSeg.label, 0, 0, 1, 5)

        self.preColSeg.label_2 = QtWidgets.QLabel(self)
        hsv_image=cv2.cvtColor(self.image,cv2.COLOR_BGR2HSV)
        low_col = np.array([int(self.min_hue.text()), int(self.min_sat.text()), int(self.min_val.text())])
        high_col = np.array([int(self.max_hue.text()), int(self.max_sat.text()), int(self.max_val.text())])
        seg_mask = cv2.inRange(hsv_image, low_col, high_col)
        self.colorSeg = cv2.bitwise_and(self.image, self.image, mask=seg_mask)
        colSeg_img = QImage(self.colorSeg, self.colorSeg.shape[1], self.colorSeg.shape[0], self.colorSeg.shape[1] * 3,
                           QImage.Format_RGB888).rgbSwapped()
        pixmap_1 = QPixmap(colSeg_img)
        pixmap_1 = pixmap_1.scaled(400, 300, QtCore.Qt.KeepAspectRatio)
        self.preColSeg.label_2.setPixmap(pixmap_1)
        self.preColSeg.grid.addWidget(self.preColSeg.label_2, 2, 0, 1, 1)

        self.preColSeg.label_3 = QtWidgets.QLabel(self)
        self.preColSeg.label_3.setText('')
        self.preColSeg.grid.addWidget(self.preColSeg.label_3, 3, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Apply Color Segmentation', self)
        btn_1.clicked.connect(self.colorSegmentation)
        btn_1.resize(btn_1.sizeHint())
        self.preColSeg.grid.addWidget(btn_1, 4, 0, 1, 1)

        btn_2 = QtWidgets.QPushButton('Cancel', self)
        btn_2.clicked.connect(self.closeColSeg)
        btn_2.resize(btn_2.sizeHint())
        self.preColSeg.grid.addWidget(btn_2, 5, 0, 1, 1)

        self.preColSeg.show()

    def closeColSeg(self):
        self.preColSeg.close()

    def colorSegmentation(self):
        self.image = self.colorSeg
        self.closeColSeg()
        self.updateColor()

    def kMeansDialog(self):
        self.kMDlg = QMainWindow(self)
        self.kMDlg.setWindowTitle('K-Means Clustering')

        self.kMDlg.central_widget = QtWidgets.QWidget()
        self.kMDlg.setCentralWidget(self.kMDlg.central_widget)
        self.kMDlg.grid = QtWidgets.QGridLayout(self.kMDlg.central_widget)

        self.kMDlg.label = QtWidgets.QLabel(self)
        self.kMDlg.label.setStyleSheet("font : 20pt")
        self.kMDlg.label.setText('Define criteria, k and set flags : ')
        self.kMDlg.grid.addWidget(self.kMDlg.label, 0, 0, 1, 5)

        self.kMDlg.label_2 = QtWidgets.QLabel(self)
        self.kMDlg.label_2.setText('Criteria : ')
        self.kMDlg.grid.addWidget(self.kMDlg.label_2, 1, 0, 1, 2)

        self.kMDlg.label_3 = QtWidgets.QLabel(self)
        self.kMDlg.label_3.setText('Maximum number of iterations = ')
        self.kMDlg.grid.addWidget(self.kMDlg.label_3, 2, 0, 1, 2)

        self.max_iter = QtWidgets.QLineEdit(self)
        self.kMDlg.grid.addWidget(self.max_iter, 2, 2, 1, 2)

        self.kMDlg.label_4 = QtWidgets.QLabel(self)
        self.kMDlg.label_4.setText('Accuracy of epsilon (0-100%) = ')
        self.kMDlg.grid.addWidget(self.kMDlg.label_4, 3, 0, 1, 2)

        self.epsilon = QtWidgets.QLineEdit(self)
        self.kMDlg.grid.addWidget(self.epsilon, 3, 2, 1, 2)

        self.kMDlg.label_5 = QtWidgets.QLabel(self)
        self.kMDlg.label_5.setText('')
        self.kMDlg.grid.addWidget(self.kMDlg.label_5, 4, 0, 1, 2)

        self.kMDlg.label_6 = QtWidgets.QLabel(self)
        self.kMDlg.label_6.setText('Number of clusters (k>1) = ')
        self.kMDlg.grid.addWidget(self.kMDlg.label_6, 5, 0, 1, 2)

        self.k = QtWidgets.QLineEdit(self)
        self.kMDlg.grid.addWidget(self.k, 5, 2, 1, 2)

        self.kMDlg.label_7 = QtWidgets.QLabel(self)
        self.kMDlg.label_7.setText('')
        self.kMDlg.grid.addWidget(self.kMDlg.label_7, 6, 0, 1, 2)

        self.kMDlg.label_6 = QtWidgets.QLabel(self)
        self.kMDlg.label_6.setText('     Flags = ')
        self.kMDlg.grid.addWidget(self.kMDlg.label_6, 7, 0, 1, 2)

        self.kMDlg.comboBox = QtWidgets.QComboBox(self)
        self.kMDlg.comboBox.addItem("KMEANS_RANDOM_CENTERS")
        self.kMDlg.comboBox.addItem("KMEANS_PP_CENTERS")
        self.kMDlg.comboBox.activated[str].connect(self.setFlags)
        self.kMDlg.grid.addWidget(self.kMDlg.comboBox, 7, 1, 1, 3)

        self.kMDlg.label_7 = QtWidgets.QLabel(self)
        self.kMDlg.label_7.setText('')
        self.kMDlg.grid.addWidget(self.kMDlg.label_7, 8, 0, 1, 1)

        btn_1 = QtWidgets.QPushButton('Random', self)
        btn_1.clicked.connect(self.randomKMeans)
        btn_1.resize(btn_1.sizeHint())
        self.kMDlg.grid.addWidget(btn_1, 9, 2, 1, 1)

        btn_2 = QtWidgets.QPushButton('OK', self)
        btn_2.clicked.connect(self.preKMeans)
        btn_2.resize(btn_2.sizeHint())
        self.kMDlg.grid.addWidget(btn_2, 9, 4, 1, 1)

        self.kMDlg.show()

    def setFlags(self,flg):
        if str(flg) == "KMEANS_PP_CENTERS":
            self.flag = cv2.KMEANS_PP_CENTERS
        elif str(flg) == "KMEANS_RANDOM_CENTERS":
            self.flag = cv2.KMEANS_RANDOM_CENTERS

    def randomKMeans(self):
        self.max_iter.setText("100")
        self.epsilon.setText(str(random.randint(0,100)))
        self.k.setText(str(random.randint(2,10)))
        self.setFlags("KMEANS_RANDOM_CENTERS")

    def preKMeans(self):
        self.kMDlg.close()
        self.clusterDlg = QMainWindow(self)
        self.clusterDlg.setWindowTitle('K-Means Clustering')

        self.clusterDlg.central_widget = QtWidgets.QWidget()
        self.clusterDlg.setCentralWidget(self.clusterDlg.central_widget)
        self.clusterDlg.grid = QtWidgets.QGridLayout(self.clusterDlg.central_widget)

        self.clusterDlg.label = QtWidgets.QLabel(self)
        k=int(self.k.text())
        if k<=1:
            k=2
        text='Preview K-Means Clustering : k = '+str(k)
        self.clusterDlg.label.setText(text)
        self.clusterDlg.grid.addWidget(self.clusterDlg.label, 0, 0, 1, 1)

        self.clusterDlg.label_2 = QtWidgets.QLabel(self)
        text_2='Disable the cluster number (x: 0-'+str(k-1) +') : \n (turn the pixel into black)'
        self.clusterDlg.label_2.setText(text_2)
        self.clusterDlg.grid.addWidget(self.clusterDlg.label_2, 0, 2, 2, 1)

        self.clusterNum = QtWidgets.QLineEdit(self)
        self.clusterDlg.grid.addWidget(self.clusterNum, 0, 3, 1, 1)

        btn_1 = QtWidgets.QPushButton('Go', self)
        btn_1.clicked.connect(self.clusterDisable)
        btn_1.resize(btn_1.sizeHint())
        self.clusterDlg.grid.addWidget(btn_1, 0, 4, 1, 1)

        self.clusterDlg.label_3 = QtWidgets.QLabel(self)
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = img.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)
        # print(pixel_values.shape)
        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(self.max_iter.text()), (int(self.epsilon.text())/100))
        retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, self.flag)
        # convert back to 8 bit values
        centers = np.uint8(centers)
        # convert all pixels to the color of the centroids
        intermediate_img = centers[labels.flatten()]
        # reshape back to the original image dimension
        self.clustered_img = intermediate_img.reshape(img.shape)
        # cv2.imshow("clustering", clustered_img)
        KM_img = QImage(self.clustered_img, self.clustered_img.shape[1], self.clustered_img.shape[0],
                        self.clustered_img.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
        pixmap_1 = QPixmap(KM_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.clusterDlg.label_3.setPixmap(pixmap_1)
        self.clusterDlg.grid.addWidget(self.clusterDlg.label_3, 3, 0, 1, 1)

        self.clusterDlg.label_4 = QtWidgets.QLabel(self)
        self.clusterDlg.grid.addWidget(self.clusterDlg.label_4, 3, 3, 1, 1)

        btn_2 = QtWidgets.QPushButton('Apply K-Means Clustering', self)
        btn_2.clicked.connect(self.cluster)
        btn_2.resize(btn_2.sizeHint())
        self.clusterDlg.grid.addWidget(btn_2, 4, 0, 1, 1)

        btn_3 = QtWidgets.QPushButton('Cancel', self)
        btn_3.clicked.connect(self.closeCluster)
        btn_3.resize(btn_3.sizeHint())
        self.clusterDlg.grid.addWidget(btn_3, 5, 1, 1, 2)

        self.clusterDlg.show()

    def clusterDisable(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(self.max_iter.text()), (int(self.epsilon.text()) / 100))
        k = int(self.k.text())
        retval, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, self.flag)
        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(img)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        # color (i.e cluster) to disable
        cluster = int(self.clusterNum.text())
        masked_image[labels.flatten() == cluster] = [0, 0, 0]
        # convert back to original shape
        self.maskedImg = masked_image.reshape(img.shape)
        # cv2.imshow("mask cluster",masked_image)
        disableCluster_img = QImage(self.maskedImg, self.maskedImg.shape[1], self.maskedImg.shape[0],
                        self.maskedImg.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
        pixmap_1 = QPixmap(disableCluster_img)
        pixmap_1 = pixmap_1.scaled(200, 300, QtCore.Qt.KeepAspectRatio)
        self.clusterDlg.label_4.setPixmap(pixmap_1)

        btn_4 = QtWidgets.QPushButton('Apply Disable Cluster Number _x_', self)
        btn_4.clicked.connect(self.applyDisableCluster)
        btn_4.resize(btn_4.sizeHint())
        self.clusterDlg.grid.addWidget(btn_4, 4, 3, 1, 1)

    def closeCluster(self):
        self.clusterDlg.close()

    def cluster(self):
        self.image = self.clustered_img
        self.closeCluster()
        self.updateColor()

    def applyDisableCluster(self):
        self.image=self.maskedImg
        self.closeCluster()
        self.updateColor()

def main():
    app = QApplication(sys.argv)
    myWindow = Win()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()