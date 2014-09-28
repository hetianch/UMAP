import PyQt5
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem

class Video(QGraphicsVideoItem):
	def __init__(self,parent=None):
		super(Video, self).__init__()
		self.installEventFilter(self)

	def eventFilter(self,source,event):
		if (event.type()==PyQt5.QtCore.QEvent.GraphicsSceneMousePress):
			pos=event.pos()	
			print event
			print event.type()
			print('mouse position: (%d,%d)' % (pos.x(),pos.y()))	
			return False

		return True