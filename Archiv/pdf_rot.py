# -*- coding: utf-8 -*-
"""

This file rotates and/or merges PDF files.

@author: Hristo Goumnerov
"""
#%%% Import modules

# import os for directory information
import os

# import fitz for pdf processing
import fitz

# import tkinter for file selection dialog
import tkinter
tkinter.Tk().withdraw()
from tkinter.filedialog import askopenfilenames as asks

#%%% Rotate file(s)

# function to rotate PDF
def pdf_rot(inp,rot):

    # open the PDF
    doc = fitz.open(inp)
    
    # rotate each page in PDF
    for page in doc:
        page.setRotation(rot)

    # update and save the file
    doc.save(inp, incremental = True)
    doc.close()

    # print completion statement to console
    fn = os.path.basename(inp)[:-4]

    return print(fn+'      DONE!')

#%% Merge file(s)

# function to merge multiple PDFs
def merge(inp):
    
    # get filenames
    fn = os.path.dirname(inp[0])
    
    # initialize output file
    out = fitz.open()

    # add each PDF to output file
    for i in range(len(inp)):
        cur = fitz.open(inp[i])
        out.insertPDF(cur)
    
    # save output file
    out.save(fn+'/output.pdf')
    out.close()

    # print completion statement to console
    return print('DONE!')
#%% Rotate file(s)

# Filename(s)
inp = asks()

# rotation angle
rot = 90

# run function for one or multiple PDFs
if len(inp) == 1:
    print('\n')
    pdf_rot(inp[0],rot)
else:
    print('\n')
    for i in len(inp):
        pdf_rot(inp[i],rot)

#%% Merge files

# Filename(s)
inp = asks()

# merge PDFs
merge(inp)
