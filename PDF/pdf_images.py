#%% import modules

from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import fitz


#%% extract images

def extract(files, outpath):

    count = 0
    for i in files:
        doc = fitz.open(i)
        for f in range(len(doc)):

            for img in doc.getPageImageList(f):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:       # this is GRAY or RGB
                    pix.writePNG(outpath + '/' + i.split('/')[-1][:-4] + "_p_%s.png" % (f+1))
                else:               # CMYK: convert to RGB first
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.writePNG(outpath + '/' + i.split('/')[-1][:-4] + "_p_%s.png" % (f+1))
                    pix1 = None
                pix = None
        count += 1
        curprog = (100*count/len(files))

        sys.stdout.write('\r[%.0f] %s' % (curprog, '%'))
        sys.stdout.flush()


if __name__ == '__main__':

    app = QApplication([])
    files = QFileDialog.getOpenFileNames(None, "Select PDF file", '', "PDF File (*.pdf)")[0]

    outdir = '/'.join(files[0].split('/')[:-1])
    os.mkdir(outdir + '/Images')

    extract(files, outdir+'/Images')