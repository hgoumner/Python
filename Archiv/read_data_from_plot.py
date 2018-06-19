# -*- coding: utf-8 -*-
"""

******************
*** INCOMPLETE ***
******************

This file reads in a plot and allows the user to manually extract the data.

@author: Hristo Goumnerov
"""
#%% Import modules

# pylint: disable=I0011,C0103,E1101

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy as sp

# import matplotlib for plotting
import matplotlib.pyplot as plt

# import tkinter for file selection dialog
import tkinter
tkinter.Tk().withdraw()
from tkinter.filedialog import askopenfilename as ask

#%% Import image

event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))
xcl = []
ycl = []

if __name__ == "__main__":
    root = tkinter.Tk()
    root.geometry('{}x{}'.format(800, 600))
#    root.withdraw()
    
    #setting up a tkinter canvas with scrollbars
    frame = tkinter.Frame(root, bd=2, relief=tkinter.SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    canvas = tkinter.Canvas(frame, bd=0)
    canvas.grid(row=0, column=0, sticky=tkinter.N+tkinter.S+tkinter.E+tkinter.W)
    frame.pack(fill=tkinter.BOTH,expand=1)

    #adding the image
    File = ask(parent=root, initialdir="M:/",title='Choose an image.')
    print("opening %s" % File)
    img = tkinter.PhotoImage(file=File)
    w = img.width
    h = img.height
    canvas.create_image(0,0,image=img,anchor="nw")
    
    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        cx, cy = event2canvas(event, canvas)
        print ("(%d, %d)" % (cx,cy))
        return xcl.append(cx), ycl.append(cy)
    
    #mouseclick event
    canvas.bind("<ButtonPress-1>",printcoords)
    canvas.bind("<ButtonRelease-1>",printcoords)
    
    root.mainloop()

#%% Select axis ranges

# Origin X axis
x1 = [xcl[0],ycl[0]]
# Max X axis
x2 = [xcl[2],ycl[0]]
# Origin Y axis
y1 = x1
# Max Y axis
y2 = [xcl[-1],ycl[-1]]

# Selected points
x0 = np.int32([x1[0],x2[0],y1[0],y2[0]])
y0 = np.int32([x1[1],x2[1],y1[1],y2[1]])

I = sp.misc.imread(File)

# Plot Image with selected points
plt.figure(1)
plt.imshow(I)
plt.plot(x0,y0,'ro')

#%% Process Data

# Crop Image
xori = int(x1[0])
yori = int(y1[1])
w = int(x2[0]-x1[0])
h = int(y2[1]-y1[1])

II = np.int32(I[:,:,0])

area = (xori,yori+h,xori+w,yori)
Ic = II[yori+h-1:yori-1,xori-1:xori+w-1]

# Extract data points in pixel values

xoff = 3
yoff = 1

Ic = Ic[:-yoff,xoff:]

xn = np.arange(0,np.shape(Ic)[1],1)
yn = np.zeros([np.shape(Ic)[1],1],dtype=np.int32)

for i in range(len(yn)):
    if np.nonzero(Ic[:,i]==0)[0] != []:
        yn[i] = int(np.median(np.nonzero(Ic[:,i]==0)))
    else:
        yn[i] = np.shape(Ic)[0]

# Convert pixel to real world coordinates
xmin = 0
xmax = 6
ymin = 0
ymax = 1

xout = np.zeros([len(xn),1],dtype=np.float64)
yout = np.zeros([len(yn),1],dtype=np.float64)

for i in range(len(xn)):
    xout[i] = (xn[i])*(xmax-xmin)/(np.shape(Ic)[1])
    yout[i] = (np.shape(Ic)[0]-yn[i])*(ymax-ymin)/(np.shape(Ic)[0])

#%% Plot Data

plt.figure(2)
plt.imshow(I)

plt.figure(3)
plt.plot(xout,yout)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])

#%% Export Data

output = np.c_[xout,yout]
