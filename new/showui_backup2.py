import sys
import PyQt5
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem

from ui import Ui_MainWindow
class jaabaGUI(QMainWindow):
    """ controller for the blob labeling GUI"""
    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #setup Video
        #video player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        # self.mediaPlayer.metaDataChanged.connect(self.metaDataChanged)

        #visualizetion
        self.scene = QGraphicsScene()
        

        self.ui.graphicsView.setScene(self.scene)
       
        self.scene.setBackgroundBrush(Qt.black)
        self.scene.setSceneRect(0,0,800,600)

      
       # self.videoItem1 = QGraphicsVideoItem()

        self.videoItem2 = QGraphicsVideoItem()
        #print self.ui.graphicsView.width()/2,self.ui.graphicsView.height()
        #self.videoItem1.setSize(QSizeF(self.ui.graphicsView.width()/2,self.ui.graphicsView.height()))
        #self.videoItem2.setSize(QSizeF(self.ui.graphicsView.width()*10,self.ui.graphicsView.height()*10))
       # self.videoItem2.setSize(graphicsView.size())
        #self.videoItem2.setOffset(QPointF(500,500))
        #self.videoItem2.setOffset(QPointF(self.ui.graphicsView.width()/2,0))
        self.videoItem2.setSize(QSizeF(800,600))
        #self.videoItem2.setPos(QPointF(0,0))
        #self.scene.addItem(self.videoItem1)
        self.scene.addItem(self.videoItem2)

        # print self.ui.graphicsView.width(), self.ui.graphicsView.height()
        # print self.ui.graphicsView.size()
        # print self.videoItem2.boundingRect().width(), self.videoItem2.boundingRect().height()
        # print self.ui.graphicsView.sceneRect()

        #self.mediaPlayer.setVideoOutput(self.videoItem1)
        self.mediaPlayer.setVideoOutput(self.videoItem2)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)

        #callbacks
        self.ui.actionQuit.triggered.connect(self.quit)
        self.ui.actionLoad_Project.triggered.connect(self.loadVideo)
        #self.ui.buttonPlay.clicked[bool].connect(self.setToggleText)
        self.ui.buttonPlay.clicked.connect(self.play)
        #print self.ui.graphicsView.sizeHint()


        #initialization
        self.loaded = False
        self.videoFilename = None

        
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
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.videoFilename )))
            self.ui.buttonPlay.setEnabled(True)
            self.mediaPlayer.setVideoOutput(self.videoItem2)
            # size= self.videoItem2.nativeSize()
            # print size
            print self.mediaPlayer.duration()
          
            #print self.mediaPlayer.metaData()



    def play(self):
        print 'graphics size', self.ui.graphicsView.size()

        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()
        
        size= self.videoItem2.nativeSize()
        print self.mediaPlayer.duration()
      
        #print self.mediaPlayer.metaData()
      

        print self.ui.graphicsView.width(), self.ui.graphicsView.height()
        print self.ui.graphicsView.size()
        print self.videoItem2.boundingRect().width(), self.videoItem2.boundingRect().height()
        print self.ui.graphicsView.sceneRect()
        print self.scene.sceneRect()
        print self.ui.graphicsView.sizeHint()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.ui.buttonPlay.setIcon(
                self.ui.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPause))
            self.ui.buttonPlay.setText("Stop")
        else:
            self.ui.buttonPlay.setIcon(
                self.ui.style().standardIcon(PyQt5.QtWidgets.QStyle.SP_MediaPlay))
            self.ui.buttonPlay.setText("Stop")

        # def setToggleText(self,pressed):
        #     source = self.sender()

        #     if source.text()=="Play":
        #         self.ui.buttonPlay.setText("Stop")
        #     else:
        #         self.ui.buttonPlay.setText("Play")

    def writeLog(self,text):
        self.ui.log.setText(text)


    # def metaDataChanged(self):
    #     QSize resolution = self.mediaPlayer.metaData()
    #     #resolution= self.mediaPlayer.metaData().get(QtCore.QString('Resolution'),[QtCore.Qvariant()])
    #     self.mediaPlayer.GetMetadata()
    #     resolution = QmediaMetaData.Resolution
    #     videoframerate= self.mediaPlayer.metaData('VideoFrameRate')
    #     print resolution, videoframerate

        # def play(self):
        #     try:
        #       self.scene.clear()
        #       self.video1.captureNextFrame()
        #       self.video2.captureNextFrame()

        #       self.image1 = self.video1.convertFrame()
        #       self.image2 = self.video2.convertFrame()

        #       self.image1 = self.image1.scaled(self.ui.graphicsView.width()/2,self.ui.graphicsView.height())
        #       self.image2 = self.image2.scaled(self.ui.graphicsView.width()/2,self.ui.graphicsView.height())

        #       self.pm1 = self.scene.addPixmap(self.image1)
        #       self.pm2 = self.scene.addPixmap(self.image2)

        #       self.pm1.setOffset(0,0)
        #       self.pm2.setOffset(self.image1.width(),0)

        #       self.scene.update()

        #       # Memory profiling.
        #       #print h.heap()

        #     except TypeError:
        #       self.writeLog("No frame")
        #       raise   
        
        # def playVideo(self):
        #     if not self._timer:
        #       if not self.loaded:
        #         self.loadVideo()
        #       self.writeLog("Playing video...")
        #       self.ui.buttonPlay.setText("Stop")
        #       self._timer = QTimer(self)
        #       self._timer.timeout.connect(self.play)
        #      # self._timer.start(self.timerDelay)
        #     else:
        #       self.ui.buttonPlay.setText("Play")
        #       self.writeLog("Video paused")
        #       self._timer.stop()
        #       self._timer = None



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = jaabaGUI()
    gui.show()
    sys.exit(app.exec_())



    #good example:http://hasanaga.info/simple-avi-player-with-opencv-qt-4-8-playing-video-file-with-slider-position/