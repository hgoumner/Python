## import modules

from PyQt5.QtWidgets import QApplication, QFileDialog
import sys
import os
import fitz
import re


## extract images

def extract(outpath):

    files = QFileDialog.getOpenFileNames(None, "Select PDF file", '', "PDF File (*.pdf)")[0]

    for i in files:

        fname = os.path.splitext(os.path.basename(i))[0]

        with fitz.open(i) as doc:

            # export image text
            pages = []
            count_t = 0
            for page in doc:
                text =  page.getText()
                if re.search('(figure|fig\.|abbildung|abb\.)', text, re.IGNORECASE):
                    pages.append(count_t+1)
                count_t += 1
            with open(outpath + fname + '.txt', 'w') as text_out:
                for p in pages:
                    text_out.write(str(p))
                    text_out.write('\n')


            # export images
            for f in range(len(doc)):

                tot_ims = len(doc.getPageImageList(f))

                if tot_ims > 5:
                    for img in doc.getPageImageList(f):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n < 2:  # this is GRAY or RGB
                            pix.writePNG(outpath + fname + "_p_%s.png" % (f + 1))
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            pix1.writePNG(outpath + fname + "_p_%s.png" % (f + 1))
                            pix1 = None
                        pix = None
                else:
                    count_i = 1
                    for img in doc.getPageImageList(f):
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n < 2:       # this is GRAY or RGB
                            pix.writePNG(outpath + fname + "_p_%s_%s.png" % (f+1, count_i))
                        else:               # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            pix1.writePNG(outpath + fname + "_p_%s_%s.png" % (f+1, count_i))
                            pix1 = None
                        pix = None
                        count_i += 1

            curprog = (100*(f+1)/len(doc))

            print('\r' + os.path.basename(i) + ' %.0f %s' % (curprog, '%'))


def main():
    app = QApplication([])
    outdir = r'C:\Users\gou\Documents\Transfer_C\Images\\'
    extract(outdir)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


'''
def get_text_percentage(file_name: str) -> float:
    """
    Calculate the percentage of document that is covered by (searchable) text.

    If the returned percentage of text is very low, the document is
    most likely a scanned PDF
    """
    total_page_area = 0.0
    total_text_area = 0.0

    doc = fitz.open(file_name)

    for page_num, page in enumerate(doc):
        total_page_area = total_page_area + abs(page.rect)
        text_area = 0.0
        for b in page.getTextBlocks():
            r = fitz.Rect(b[:4])  # rectangle where block text appears
            text_area = text_area + abs(r)
        total_text_area = total_text_area + text_area
    doc.close()
    return total_text_area / total_page_area


if __name__ == "__main__":
    text_perc = get_text_percentage("my.pdf")
    print(text_perc)
    if text_perc < 0.01:
        print("fully scanned PDF - no relevant text")
    else:
        print("not fully scanned PDF - text is present")
'''

