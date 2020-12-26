from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import os

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        color = self.palette().color(QtGui.QPalette.Window)
        styles = {'color': 'r', 'font-size': '20px'}

        self.graphWidget = pg.PlotWidget()
        hour = range(1,11)
        temperature = [30,32,34,32,33,31,29,32,35,45]
        pen = pg.mkPen(color='r', width=5, style=Qt.DashLine)
        self.graphWidget.setBackground(color)
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setTitle('My Title', **styles)
        self.graphWidget.setLabel('bottom', 'Hour', **styles)
        self.graphWidget.setLabel('left', 'Temperature', **styles)
        self.graphWidget.addLegend()
        self.graphWidget.plot(hour, temperature, name='Test', pen=pen, symbol='+', symbolSize=15, symbolBrush=('b'))

        self.setCentralWidget(self.graphWidget)


def main():
    app = QtWidgets.QApplication([])
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())



if __name__ == '__main__':
    main()