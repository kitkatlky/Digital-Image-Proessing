import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication, QMessageBox,QWidget,QScrollArea
from PyQt5.QtGui import QPixmap, QImage
import cv2
from skimage.util import img_as_ubyte
from skimage.morphology import skeletonize
from scipy.spatial import distance as dist
from imutils import contours,perspective
import numpy as np
import math
import imutils

class ScrollLabel(QScrollArea):

    # contructor
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)

        # making widget resizable
        self.setWidgetResizable(True)

        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QtWidgets.QVBoxLayout(content)

        # creating label
        self.label = QtWidgets.QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def setText(self, text):
        # setting text to the label
        self.label.setText(text)

class Win(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.resize(800, 600)
        self.setFixedSize(800,600)
        self.move(300, 20)
        self.setWindowTitle('Battery Classification Tools')

        self.createMenuBar()
        self.newFile()
        self.showAll=False
        self.lengthObj=False
        self.batteryType=False
        self.calVol=False
        self.show()


    def createMenuBar(self):
        newAct = QAction('New', self)
        newAct.setShortcut('Ctrl+N')
        newAct.triggered.connect(self.newFile)

        openAct = QAction('Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.triggered.connect(self.openFile)

        saveAct = QAction('Save',self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.triggered.connect(self.saveImg)

        exitAct = QAction('Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.triggered.connect(self.closeEvent)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        fileMenu.addAction(newAct)
        fileMenu.addAction(openAct)
        fileMenu.addAction(saveAct)
        fileMenu.addAction(exitAct)

        showAct = QAction('Show Detected Object',self)
        showAct.triggered.connect(self.showAllObj)

        measureAct = QAction ('Measure Detected Object',self)
        measureAct.triggered.connect(self.measure)

        classifyAct = QAction('Classification of Battery', self)
        classifyAct.triggered.connect(self.classify)

        volumeAct = QAction('Volume of Battery',self)
        volumeAct.triggered.connect(self.volume)

        funcMenu = menubar.addMenu('Function')
        funcMenu.addAction(showAct)
        funcMenu.addAction(measureAct)
        funcMenu.addAction(classifyAct)
        funcMenu.addAction(volumeAct)


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
            self.closeEvent()

    def closeEvent(self, event):

        reply = QMessageBox.question(self, 'Close',"Are you sure to quit?",
                                     QMessageBox.Yes |QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:

            event.accept()
        else:

            event.ignore()

    def newFile(self):
        self.name='white_81.jpg'
        self.image = cv2.imread(self.name)
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.grid = QtWidgets.QGridLayout(self.central_widget)

        self.imagePreview = self.image
        self.label = QtWidgets.QLabel(self)
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                     QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)
        self.grid.addWidget(self.label,1,0,1,13)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setStyleSheet("font : 15pt")
        self.label_2.setText('(Draw a 2cm*2cm box on top left corner to get more accurate measurement result.) ')
        self.grid.addWidget(self.label_2, 0, 0, 1, 13)

        btn1 = QtWidgets.QPushButton('Remove All Label', self)
        btn1.clicked.connect(self.resetPic)
        btn1.resize(btn1.sizeHint())
        self.grid.addWidget(btn1, 2, 10, 1, 2)

        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setText('Output Detail : ')
        self.grid.addWidget(self.label_3, 3, 0, 1, 2)

        # creating scroll label
        self.label_4 = ScrollLabel(self)
        # setting text to the label
        self.text = ''
        self.label_4.setText(self.text)
        self.grid.addWidget(self.label_4, 3, 2, 3, 8)

        btn2 = QtWidgets.QPushButton('Clear Output Detail ', self)
        btn2.clicked.connect(self.resetBox)
        btn2.resize(btn2.sizeHint())
        self.grid.addWidget(btn2, 6, 8, 1, 2)

    def openFile(self):
        self.name = QtWidgets.QFileDialog.getOpenFileName(self, "Select Image")[0]
        self.image = cv2.imread(self.name)
        self.imgHeight = self.image.shape[0]
        self.imgWidth = self.image.shape[1]

        self.imagePreview = self.image
        self.pixmap = QPixmap(self.name)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

    def keepRatio(self):
        if self.imgHeight>400 and self.imgWidth<=800 :
            self.pixmap = self.pixmap.scaled(self.imgWidth, 400, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight<=400 and self.imgWidth>800 :
            self.pixmap = self.pixmap.scaled(800, self.imgHeight, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight > 400 and self.imgWidth > 800:
            self.pixmap = self.pixmap.scaled(800, 400, QtCore.Qt.KeepAspectRatio)
        elif self.imgHeight <= 400 and self.imgWidth <= 800:
            self.pixmap = self.pixmap.scaled(self.imgWidth, self.imgHeight, QtCore.Qt.KeepAspectRatio)


    def saveImg(self):
        fname, fliter = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', 'C:\\Users\\user\\Desktop\\',
                                                              "Image Files (*.jpg);;Image Files (*.tiff);;Image Files (*.bmp)")
        self.image=self.imagePreview
        if fname:
            cv2.imwrite(fname, self.image)
        else:
            print('Error')

    def edgeDialog(self):
        self.dlg = QMainWindow(self)
        self.dlg.setWindowTitle('Edge Detection')

        self.dlg.central_widget = QtWidgets.QWidget()
        self.dlg.setCentralWidget(self.dlg.central_widget)
        self.dlg.grid = QtWidgets.QGridLayout(self.dlg.central_widget)

        self.dlg.label = QtWidgets.QLabel(self)
        self.dlg.label.setStyleSheet("font : 14pt")
        self.dlg.label.setText('Select one type of edge detection')
        self.dlg.grid.addWidget(self.dlg.label, 0, 0, 1, 5)

        self.dlg.label_2 = QtWidgets.QLabel(self)
        self.dlg.label_2.setText(' ')
        self.dlg.grid.addWidget(self.dlg.label_2, 1, 0, 1, 2)

        self.dlg.label_3 = QtWidgets.QLabel(self)
        self.dlg.label_3.setText('Edge detection by using : ')
        self.dlg.grid.addWidget(self.dlg.label_3, 2, 0, 1, 2)

        self.edgeType=1
        self.dlg.comboBox = QtWidgets.QComboBox(self)
        self.dlg.comboBox.addItem("Canny")
        self.dlg.comboBox.addItem("Sobel")
        self.dlg.comboBox.addItem("Prewitt")
        self.dlg.comboBox.activated[str].connect(self.edgeDetect)
        self.dlg.grid.addWidget(self.dlg.comboBox, 2, 2, 1, 2)

        self.dlg.label_4 = QtWidgets.QLabel(self)
        self.dlg.label_4.setText(' ')
        self.dlg.grid.addWidget(self.dlg.label_4, 3, 3, 1, 2)

        btn1 = QtWidgets.QPushButton('OK', self)
        btn1.clicked.connect(self.closeEdge)
        btn1.resize(btn1.sizeHint())
        self.dlg.grid.addWidget(btn1, 4, 4, 1, 1)

        self.dlg.show()

    def edgeDetect(self,type):
        if str(type) == "Canny":
            self.edgeType=1
        elif str(type) == "Sobel":
            self.edgeType=2
        elif str(type) == "Prewitt":
            self.edgeType=3

    def closeEdge(self):
        self.dlg.close()
        self.runProgram()

    def runProgram(self):
        image = self.image.copy()

        # convert the image to grayscale, blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.edgeType == 1:
            edged = cv2.Canny(gray, 50, 100)
            self.text += "Canny edge detection : \n"
            self.label_4.setText(self.text)

        elif self.edgeType == 2:
            sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            sobelX = np.uint8(np.absolute(sobelX))
            sobelY = np.uint8(np.absolute(sobelY))
            img_sobel=cv2.bitwise_or(sobelX,sobelY)
            _,threshold=cv2.threshold(img_sobel,60,1,cv2.THRESH_BINARY)
            skeleton = skeletonize(threshold)
            edged = img_as_ubyte(skeleton)

            self.text += "Sobel edge detection : \n"
            self.label_4.setText(self.text)

        elif self.edgeType == 3:
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            img_prewittx = cv2.filter2D(blurred, -1, kernelx)
            img_prewitty = cv2.filter2D(blurred, -1, kernely)
            prewitt = cv2.bitwise_or(img_prewittx, img_prewitty)
            _, threshold = cv2.threshold(prewitt, 20, 1, cv2.THRESH_BINARY)
            skeleton = skeletonize(threshold)
            edged = img_as_ubyte(skeleton)

            self.text += "Prewitt edge detection : \n"
            self.label_4.setText(self.text)

        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        noiseLeft=0

        for (i, c) in enumerate(cnts):
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 200:
                noiseLeft+=1
                continue

            # compute the rotated bounding box of the contour
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
            # cv2.drawContours(image, [c.astype("int")], -1, (0, 255, 0), 2)

            # # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # compute the center of the contour
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))

            cv2.putText(image, "#{}".format(i + 1-noiseLeft), (cX-20, cY-5),cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # if the pixels per metric has not been initialized, then
            # compute it as the ratio of pixels to supplied metric
            # (in this case, 1cm=38 pixels)
            if pixelsPerMetric is None:
                if i==0 and -5<dA-dB<5:
                    pixelsPerMetric = dB / 2
                else:
                    pixelsPerMetric = 38

            # compute the size of the object
            dimA = dA * 10 / pixelsPerMetric
            dimB = dB * 10 / pixelsPerMetric

            dimA,dimB=self.roundNum(dimA,dimB,1)
            if dimA > dimB:
                length = dimA
                diameter = dimB
            else:
                diameter = dimA
                length = dimB

            type = "unidentified"
            if 8.5 < diameter < 12.5 and 35.3 < length < 55.3:
                type = "AAA"
            elif 12.5 < diameter < 16.5 and 41.3 < length < 61.3:
                type = "AA"
            elif 24.2 < diameter < 28.2 and 40.8 < length < 60.8:
                type = "C"
            elif 32.2 < diameter < 36.2 and 52.3 < length < 72.3:
                type = "D"
            elif 19 < diameter < 21 and 19 < length < 21:
                type = " "

            if self.showAll:
                self.text += "#{} object found\n".format(i + 1 - noiseLeft)
                self.label_4.setText(self.text)

            elif self.lengthObj:
                # draw the object sizes on the image
                cv2.putText(image, "{:.1f}mm".format(dimB),
                            (int(tltrX-30), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2)
                cv2.putText(image, "{:.1f}mm".format(dimA),
                            (int(trbrX), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 2)

                self.text += "Object #{}\n".format(i + 1 - noiseLeft)
                self.text += "Width = {:.1f}mm".format(dimB) + "\theight = {:.1f}mm\n \n".format(dimA)
                self.label_4.setText(self.text)

            elif self.batteryType:
                cv2.putText(image, type, (cX - 20, cY + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
                if type==" ":
                    self.text += "Type Object #{} : reference square box".format(i + 1 - noiseLeft) +"\n \n"
                elif type=="unidentified":
                    self.text += "Type Object #{} :".format(i + 1 - noiseLeft) + str(type) + "\n \n"
                else:
                    self.text += "Type Object #{} :".format(i + 1 - noiseLeft)+str(type)+" battery\n \n"
                self.label_4.setText(self.text)

            elif self.calVol:
                if 19<diameter<21 and 19<length<21:
                    vol=" "
                else:
                    if not(type=="unidentified"):
                        height=length-0.8
                        vol=math.pi*math.pow(diameter/2,2)*height

                        cv2.putText(image, "{:.2f}mm2".format(vol), (cX - 50, cY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        self.text += "Volume Object #{} : {:.2f} mm\u00b2 \n \n".format(i + 1 - noiseLeft,vol)

                self.label_4.setText(self.text)


            self.imagePreview=image
            img = QImage(self.imagePreview, self.imagePreview.shape[1], self.imagePreview.shape[0],
                         self.imagePreview.shape[1] * 3,QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QPixmap(img)
            self.keepRatio()
            self.label.setPixmap(self.pixmap)

            # show the output image
            # cv2.imshow("Output", image)
            # cv2.waitKey(0)

    def roundNum(self,num1,num2,dp):
        num1=round(num1,dp)
        num2 = round(num2, dp)
        return num1,num2

    def showAllObj(self):
        self.edgeDialog()
        self.showAll=True
        self.lengthObj=False
        self.batteryType = False
        self.calVol = False

    def midpoint(self,ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def measure(self):
        self.edgeDialog()
        self.showAll=False
        self.lengthObj = True
        self.batteryType = False
        self.calVol = False

    def classify(self):
        self.edgeDialog()
        self.showAll = False
        self.lengthObj = False
        self.batteryType=True
        self.calVol=False

    def volume(self):
        self.edgeDialog()
        self.showAll = False
        self.lengthObj = False
        self.batteryType = False
        self.calVol = True

    def resetBox(self):
        self.text=' '
        self.label_4.setText(self.text)

    def resetPic(self):
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3,
                     QImage.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap(img)
        self.keepRatio()
        self.label.setPixmap(self.pixmap)

def main():
    app = QApplication(sys.argv)
    myWindow = Win()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
