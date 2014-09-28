import sys
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsItem, QGraphicsScene, QGraphicsView, QStyle)
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
import numpy as numpy
import cv2
from TargetView import TargetView
from ui import Ui_MainWindow
from Video import Video
class jaabaGUI(QMainWindow):
    """ controller for the blob labeling GUI"""
    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #add new slider
        # self.positionSlider=QSlider(Qt.Horizontal)
        # self.positionSlider.setGeometry (800,800,100,30)
        # self.positionSlider.setRange(0, 0)
        # self.positionSlider.sliderMoved.connect(self.setPosition)

        #setup Video
        #video player
        self.mediaPlayer1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        #self.mediaPlayer.metaDataChanged.connect(self.metaDataChanged)
        self.mediaPlayer1.durationChanged.connect(self.durationChanged)
        self.mediaPlayer1.positionChanged.connect(self.positionChanged)
        self.mediaPlayer2.positionChanged.connect(self.positionChanged)
        #self.mediaPlayer2.positionChanged.connect(self.paintEvent)
        

        #visualizetion
        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)
        #self.scene.setBackgroundBrush(Qt.black)
        self.videoItem1 = QGraphicsVideoItem()
        self.videoItem2 = Video()
        self.scene.addItem(self.videoItem1)
        self.scene.addItem(self.videoItem2)
        self.mediaPlayer1.setVideoOutput(self.videoItem1)
        self.mediaPlayer2.setVideoOutput(self.videoItem2)

        #mouse event 

        


        #slider bar
        self.ui.horizontalSlider.setRange(0, 0)
        self.ui.horizontalSlider.sliderMoved.connect(self.setPosition)
        # self.ui.horizontalSlider.sliderPressed.connect(self.sliderPressed)

        #draw on video
        self.flyCanvas= TargetView()
        self.scene.addItem(self.flyCanvas)

        
   


        #print self.ui.graphicsView.width()/2,self.ui.graphicsView.height()
        #self.videoItem1.setSize(QSizeF(self.ui.graphicsView.width()/2,self.ui.graphicsView.height()))
        #self.videoItem2.setSize(QSizeF(self.ui.graphicsView.width()*10,self.ui.graphicsView.height()*10))
       # self.videoItem2.setSize(graphicsView.size())
        #self.videoItem2.setOffset(QPointF(500,500))
        #self.videoItem2.setOffset(QPointF(self.ui.graphicsView.width()/2,0))   
        #self.videoItem2.setPos(QPointF(0,0))
        # print self.ui.graphicsView.width(), self.ui.graphicsView.height()
        # print self.ui.graphicsView.size()
        # print self.videoItem2.boundingRect().width(), self.videoItem2.boundingRect().height()
        # print self.ui.graphicsView.sceneRect()
        #self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)

        #callbacks
        self.ui.actionQuit.triggered.connect(self.quit)
        self.ui.actionLoad_Project.triggered.connect(self.loadVideo)
        #self.ui.buttonPlay.clicked[bool].connect(self.setToggleText)
        self.ui.buttonPlay.clicked.connect(self.play)
        #print self.ui.graphicsView.sizeHint()


        #initialization
        self.loaded = False
        self.videoFilename = None
        self.frame_count=None
        self.width=None
        self.height=None
        self.frame_trans=None



        
    # ###actions starts from here###
    def quit(self):
        QApplication.quit()

    def loadVideo(self):
        self.writeLog("Loading video...")

        self.videoFilename = QFileDialog.getOpenFileName(self, 'Open File', '.')[0]
        if not self.videoFilename:
            self.writeLog("User cancelled - no video loaded")
            return
        else:
       		cap=cv2.VideoCapture(self.videoFilename)
	    	self.frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
	    	self.width=cap.get(3)
	    	self.height=cap.get(4)
	        self.mediaPlayer2.setMedia(QMediaContent(QUrl.fromLocalFile(self.videoFilename )))
	        self.mediaPlayer1.setMedia(QMediaContent(QUrl.fromLocalFile(self.videoFilename )))
	        self.ui.buttonPlay.setEnabled(True)
            # self.mediaPlayer2.setVideoOutput(self.videoItem2)
            # self.mediaPlayer1.setVideoOutput(self.videoItem1)
            # size= self.videoItem2.nativeSize()
            # print size
            #print self.mediaPlayer.duration()
          
            #print self.mediaPlayer.metaData()
        self.writeLog("Video loaded!")

    def play(self):
    	
        self.videoItem1.setAspectRatioMode(0)
        self.videoItem2.setAspectRatioMode(0)
        self.scene.setSceneRect(0,0,self.ui.graphicsView.width(),self.ui.graphicsView.height())
        self.videoItem1.setSize(QSizeF(self.ui.graphicsView.width()/2,self.ui.graphicsView.height()))
        self.videoItem2.setSize(QSizeF(self.ui.graphicsView.width()/2,self.ui.graphicsView.height()))
        self.videoItem1.setPos(QPointF(0,0))
        self.videoItem2.setPos(QPointF(self.ui.graphicsView.width()/2,0))
        self.flyCanvas.setPos(QPointF(self.ui.graphicsView.width()/2,0))
        #self.ui.graphicsView.setGeometry(0,0, 600,800)
        #print 'graphicsView size', self.ui.graphicsView.size()
        #print 'graphicsScene size', self.scene.sceneRect()
        #self.videoItem2.setSize(QSizeF(1000,300))
        #print 'graphicsVideoItem size',self.videoItem2.size()
        # print 'item x',self.videoItem2.scenePos().x()
        # print 'item y', self.videoItem2.scenePos().y()
        # print 'item x',self.videoItem1.scenePos().x()
        # print 'item y', self.videoItem1.scenePos().y()

        if self.mediaPlayer1.state() == QMediaPlayer.PlayingState:
        	self.ui.buttonPlay.setIcon(self.ui.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPlay))
        	self.ui.buttonPlay.setText("Play")
        	self.mediaPlayer1.pause()
        	self.writeLog("Video paused")
        else: 
        	self.ui.buttonPlay.setIcon(self.ui.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPause))
	        self.ui.buttonPlay.setText("Stop")
	        self.mediaPlayer1.play()
	        self.writeLog("Playing video")

        if self.mediaPlayer2.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer2.pause()
        else: 
            self.mediaPlayer2.play()




        
        #size= self.videoItem2.nativeSize()
        # print self.mediaPlayer.duration()
      
        #print self.mediaPlayer.metaData()
      

        # print self.ui.graphicsView.width(), self.ui.graphicsView.height()
        # print self.ui.graphicsView.size()
        # print self.videoItem2.boundingRect().width(), self.videoItem2.boundingRect().height()
        # print self.ui.graphicsView.sceneRect()
        # print self.scene.sceneRect()
        # print self.ui.graphicsView.sizeHint()

    

    def setPosition(self, position):
    	self.mediaPlayer1.setPosition(position) 
    	self.mediaPlayer2.setPosition(position)  

    # when position of media changed, set slider and text box accordingly.
    def positionChanged(self, position):
        self.ui.horizontalSlider.setValue(position)
        if isinstance(self.frame_trans,float):
	        # print type(position),position
	        # print type(self.frame_trans),self.frame_trans 
	        # print position/self.frame_trans
	     	self.ui.lineEdit.setText(str(int(round(position/self.frame_trans,0))))
	       
        self.writeLog(str(position))    
    
    def durationChanged(self, duration):
	    self.ui.horizontalSlider.setRange(0, duration) 
	    self.frame_trans=self.mediaPlayer1.duration()/self.frame_count
	    #print self.frame_trans

	#def eventFilter(self,source,event):
		#if (event.type()==PyQt5.QtCore.QEvent.MousePress and source is self.videoItem2):
		# 	pos=event.pos()
		# 	print('mouse position: (%d,%d)' % (pos.x(),pos.y()))
	 #    return PyQt5.QtGui.QWidget.eventFilter(self, source, event)

    def writeLog(self,text):
        self.ui.log.setText(text)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = jaabaGUI()
    gui.show()
    sys.exit(app.exec_())



    #good example:http://hasanaga.info/simple-avi-player-with-opencv-qt-4-8-playing-video-file-with-slider-position/