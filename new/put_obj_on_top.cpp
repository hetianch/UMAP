int main(int argc, char **argv) 
{
	QApplication app(argc, argv);
 
	QMainWindow			mainWin;
	QGraphicsView *	view = new QGraphicsView;
	mainWin.setCentralWidget( view );
 
	QGraphicsScene scene;
	for ( int i = 0; i < 7; ++i )
	{
		QGraphicsRectItem *	pItem1 = new QGraphicsRectItem( 0, 0, 25, 25 );
		pItem1->setPos( 0, i * 50 );
		pItem1->setBrush( Qt::red );
		pItem1->setZValue( 10 );
 
		QGraphicsRectItem *	pItem2 = new QGraphicsRectItem( 0, 0, 25, 25 );
		pItem2->setPos( 100, i * 50 );
		pItem2->setBrush( Qt::yellow );
		pItem2->setZValue( 0 );
 
		scene.addItem( pItem1 );
		scene.addItem( pItem2 );
 
		// Connect item1 and item2 with a line
		QGraphicsItem * pLine = scene.addLine( 
			QLineF(	pItem1->mapToScene( pItem1->boundingRect().center() ), 
					pItem2->mapToScene( pItem2->boundingRect().center() ) ) );
		pLine->setZValue( 5 );
	}
 
	view->setScene( &scene );
 
	mainWin.show();
	return app.exec();
}