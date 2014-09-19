import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from tryVideo import Video

from ui import Ui_MainWindow
class jaabaGUI(QMainWindow):
    """ controller for the blob labeling GUI"""
    def __init__(self,parent=None):
        QMainWindow.__init__(self,parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionQuit.triggered.connect(self.quit)
        self.ui.actionLoad_Project.triggered.connect(self.loadVideo)
       # self.ui.buttonPlay.clicked[bool].connect(self.setToggleText)
        self.ui.buttonPlay.clicked[bool].connect(self.playVideo)
        self.ui.buttonPlay.clicked[bool].connect(self.playVideo)

        self.scene = QGraphicsScene()
        self.ui.graphicsView.setScene(self.scene)

        self._timer = None
        self.loaded = False
        self.videoFilename = None

        
###actions starts from here###
    def quit(self):
        QApplication.quit()
    def loadVideo(self):
        self.writeLog("Loading video...")

        self.videoFilename = QFileDialog.getOpenFileName(self, 'Open File', '.')[0]
        if not self.videoFilename:
            self.writeLog("User cancelled - no video loaded")
            return

        self.video1= Video(self.videoFilename)
        self.video2= Video(self.videoFilename)
        self.loaded = True
        self.play()


    def setToggleText(self,pressed):
        source = self.sender()

        if source.text()=="Play":
            self.ui.buttonPlay.setText("Stop")
        else:
            self.ui.buttonPlay.setText("Play")


    def writeLog(self,text):
        self.ui.log.setText(text)

    def play(self):
        try:
          self.scene.clear()
          self.video1.captureNextFrame()
          self.video2.captureNextFrame()

          self.image1 = self.video1.convertFrame()
          self.image2 = self.video2.convertFrame()
          
          self.image1 = self.image1.scaled(self.ui.graphicsView.width()/2,self.ui.graphicsView.height())
          self.image2 = self.image2.scaled(self.ui.graphicsView.width()/2,self.ui.graphicsView.height())
          print self.ui.graphicsView.width(),self.ui.graphicsView.height()

          self.pm1 = self.scene.addPixmap(self.image1)
          self.pm2 = self.scene.addPixmap(self.image2)
          
          self.pm1.setOffset(0,0)
          self.pm2.setOffset(self.image1.width(),0)
          
          self.scene.update()

          # Memory profiling.
          #print h.heap()

        except TypeError:
          self.writeLog("No frame")
          raise   
    
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
    def playVideo(self):
        self.writeLog("Playing video...")
        self.ui.buttonPlay.setText("Stop")
        self.video1.videoPlay
        self.video2.videoPlay
   
 
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = jaabaGUI()
    gui.show()
    sys.exit(app.exec_())



    #good example:http://hasanaga.info/simple-avi-player-with-opencv-qt-4-8-playing-video-file-with-slider-position/