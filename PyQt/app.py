# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 09:09:55 2018

This file loads a Qt Designer file and executes the program.

@author: Hristo Goumnerov
"""
#%% Import modules

# import sys for basic operations
import sys

# import PyQt5 for GUI development
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic

#%% Load file

Ui_MainWindow, QtBaseClass = uic.loadUiType("tax_calc.ui")

# create basic class
class MyApp(QMainWindow):
    
    def __init__(self):
        
        super(MyApp, self).__init__()
        
        self.ui = Ui_MainWindow()
        
        self.ui.setupUi(self)
        
        self.ui.calc_tax_button.clicked.connect(self.CalculateTax)

#%% Define functions
    
    # function to perform tax calculation
    def CalculateTax(self):
        
        price = int(self.ui.price_box.toPlainText())
        
        tax = (self.ui.tax_rate.value())
        
        total_price = price + ((tax/100)*price)
        
        total_price_string = "The total price with tax is: " + str(total_price)
        
        self.ui.results_window.setText(total_price_string)

#%% Main setup

# standard app setup
if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    
    window = MyApp()
    
    window.show()
    
    app.exec_()