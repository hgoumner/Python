# -*- coding: utf-8 -*-
"""

This file fits data from an Origin worksheet using a spline fit and stores the
output in a new worksheet.

@author: Hristo Goumnerov
"""
#%% Import modules

import sys
 
# Replace the value of string variable py_ext_path 
# with the actual Python extension installation path in your computer
 
py_ext_path = "R:/166/TRANSFER/Goumnerov/Python"
 
if py_ext_path not in sys.path:
    sys.path.append(py_ext_path)

import PyOrigin
import numpy as np
from scipy.interpolate import UnivariateSpline as spl

#%% Perform computation

wks_in = PyOrigin.ActiveLayer()

if wks_in.IsValid():
    ### Load data
        
    cx = wks_in.Columns(0)
    cy = wks_in.Columns(1)
    
    x_in = cx.GetData(0,-1)
    y_in = cy.GetData(0,-1)
    
    x_in = np.int32(x_in)
    y_in = np.float64(y_in)

    ### Run spline fitting
    
    # Spline fit function (x input, y input, min x, max x, number of points) 
    def spline_fit(x,y,xmin,xmax,n):
    	
        # Slice out region
        xc = x[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]]
        yc = y[np.nonzero(x==xmin)[0][0]:np.nonzero(x==xmax)[0][0]] 
    
        # Compute spline fit
        if (xmax+n) > len(x):
            
            print('\n')
            print('NUMBER OF POINTS EXCEEDS INPUT SIZE')
            print('\n')
        
        else:
        
            # Initialize output variables
            xcur = np.zeros(n)
            ycur = np.zeros(n)
            yspl = []
        
        # Compute fit
        for i in np.arange(0,len(xc)+n,n):
                
            # Slice out n points per step, compute third order cubic spline
            xcur = x[np.nonzero(x==xmin)[0][0]+i:np.nonzero(x==xmin)[0][0]+i+n]
            ycur = y[np.nonzero(x==xmin)[0][0]+i:np.nonzero(x==xmin)[0][0]+i+n]
            ys = spl(xcur,ycur, k=3)
            if i == len(xc)-1:
                yspl.extend(ys(xcur)[0])
                break
            else:
                yspl.extend(ys(xcur))
        
        # Store data
        yspl = np.float64(yspl[:len(xc)])
        
        # Return output
        return xc, yc, yspl
    
    ### Run Noise computation
    
    # Noise function (y original, y fitted, conversion factor)
    def noise(y1,y2,c):
        
        # Initialize output variable
        out = []
        
        # Compute noise
        for i in range(len(y1)):
            out.append((y1[i]-y2[i])*c)
        
        # Store data and standard deviation
        out = np.float64(out)
        std = np.std(out)
        
        # Return output
        return out, std
    
    ### Run code
    
    # X, Y, min X, max X, number of points for spline
    x = x_in
    y = y_in
    xmin = 1
    xmax = 1300
    n = 5
    c = 1014
    
    # Run function with given output and store output data to specified x range, original y, fitted y
    x_sl, y_sl, y_fit = spline_fit(x,y,xmin,xmax,n)
    
    # Noise function (y original, y fitted, conversion factor)
    noi, std = noise(y_sl,y_fit,c)
    
    ### Export output to worksheet
    
    # Create worksheet page named 'Output' using template named 'Origin'.
    pgName = PyOrigin.CreatePage(PyOrigin.PGTYPE_WKS, "Output", "Origin", 1)
    wp = PyOrigin.Pages(str(pgName)) 			# Get page
    wks_out = PyOrigin.ActiveLayer()     		# Get sheet
    
    # Setup worksheet
    wks_out.SetData([x_sl,y_sl,y_fit,noi,std],-1)
    
    wks_out.Columns(0).SetLongName("X");
    wks_out.Columns(1).SetLongName("Y Original");
    wks_out.Columns(2).SetLongName("Y Fitted");
    wks_out.Columns(3).SetLongName("Noise");
    wks_out.Columns(4).SetLongName("Standard Deviation");
    
#// Begin of Script
#string str$ = "R:\166\TRANSFER\Goumnerov\TOM_air\Origin\spline_fit.py";
#run -pyf "%(str$)";
#// End of Script;