import cv2,sys,logging
import numpy as np
from PyQt5 import QtGui, QtCore, Qt
#from ui import Ui_MainWindow

class Video():
    def __init__(self,filename):
        self.filename = filename
        self.capture = cv2.VideoCapture(self.filename)

        if not self.capture:
          raise ValueError("Could not open %s" % self.filename)

        self.currentFrame = np.array([])
        self.rawFrame = np.array([])
        self.frameNumber = 0


    def captureNextFrame(self):
        """
        capture frame and reverse RBG BGR and return opencv image
        """
        ret, readFrame=self.capture.read()
        if(ret==True):
            self.currentFrame=cv2.cvtColor(readFrame,cv2.COLOR_BGR2RGB)
            self.currentFrame = readFrame
            self.rawFrame = readFrame
            self.frameNumber += 1


    def provideRawFrame(self,filter):
      filter.rawFrame = self.rawFrame

    def provideFrameCount(self,filter):
      filter.frameCount = self.getFrameCount()

    def convertFrame(self):
        """                                                    converts frame to format suitable for QtGui            """
        self.currentFrame = cv2.cvtColor(self.currentFrame,cv2.COLOR_BGR2RGB)
        try:
          height,width=self.currentFrame.shape[:2]
          img=QtGui.QImage(self.currentFrame,
                           width,
                           height,
                           QtGui.QImage.Format_RGB888)
          img=QtGui.QPixmap.fromImage(img)
          self.previousFrame = self.currentFrame
          return img


        except:
            return None

    def getFrameCount(self):
      try:
        fc = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
      except AttributeError:
        fc = self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
      return fc

    def videoPlay(self):
      while(self.capture.isOpened()):
        ret,frame=cap.read()
        grapy=cv.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
        cap.release()
        cv2.destroyAllWindows()
    

