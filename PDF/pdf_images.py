"""
PDF processing script
"""

# import modules
import os, sys, re
import fitz

from PyQt5.QtWidgets import QApplication, QFileDialog

# body
class Document( object ):
    """document object"""

    def __init__( self, file, outpath ):
        """object constructor"""
        self.file     = file
        self.filename = os.path.splitext( os.path.basename( self.file ) )[0]
        self.outpath  = outpath
        self.extract()

    def ext_text( self, doc ):
        """method to extract text from document"""

        pages = []
        for idx, page in enumerate(doc, start=1):
            text = page.getText()
            if re.search( '(figure|fig\.|abbildung|abb\.)', text, re.IGNORECASE ):
                pages.append( idx )

        return pages

    def sav_text( self, pages ):
        """method to save extracted text"""

        with open( self.outpath + self.filename + '.txt', 'w' ) as text_out:
            for p in pages:
                text_out.write( str( p ) )
                text_out.write( '\n' )

    def ext_img( self, doc ):
        """method to extract image(s) from document"""

        # loop over pages
        for f in range( len( doc ) ):

            tot_ims = len( doc.getPageImageList( f ) )

            if tot_ims > 5:
                for img in doc.getPageImageList( f ):
                    xref = img[0]
                    pix = fitz.Pixmap( doc, xref )
                    if pix.n < 2:  # this is GRAY or RGB
                        pix.writePNG( self.outpath + self.filename + "_p_%s.png" % (f + 1) )
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap( fitz.csRGB, pix )
                        pix1.writePNG( self.outpath + self.filename + "_p_%s.png" % (f + 1) )
                        pix1 = None
                    pix = None
            else:
                count_i = 1
                for img in doc.getPageImageList( f ):
                    xref = img[0]
                    pix = fitz.Pixmap( doc, xref )
                    if pix.n < 2:  # this is GRAY or RGB
                        pix.writePNG( self.outpath + self.filename + "_p_%s_%s.png" % (f + 1, count_i) )
                    else:  # CMYK: convert to RGB first
                        pix1 = fitz.Pixmap( fitz.csRGB, pix )
                        pix1.writePNG( self.outpath + self.filename + "_p_%s_%s.png" % (f + 1, count_i) )
                        pix1 = None
                    pix = None
                    count_i += 1

    def extract( self ):
        """method to process file"""

        with fitz.open( self.file ) as doc:

            # text
            pages = self.ext_text( doc )
            if pages:
                self.sav_text( pages )

            # images
            try:    self.ext_img( doc )
            except: pass


def main():
    app = QApplication( [] )
    files = QFileDialog.getOpenFileNames( None, "Select PDF file", '', "PDF File (*.pdf)" )[0]
    outdir = r'C:\Users\gou\Documents\Transfer_C\Images\\'

    for idx, f in enumerate(files,start=1):
        Document( f, outdir )
        curprog = (100 * idx) / len(files)
        print( '\r' + os.path.basename(f) + ' %.0f %s' % (curprog, '%') )

    sys.exit( app.exec_() )


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