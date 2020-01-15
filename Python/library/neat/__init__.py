# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import textwrap
from pprint import pprint
from neat import config
import prep

def shorten_line(line, max_width=config.MAX_WIDTH):
    """For more attractive printing on the screen. Shortens the line 
    displayed. Function breaks a line into two parts. The first part
    accepts the first 2/3 of characters. The second part accepts the
    last 2/3 of characters counting from the end of the string."""
    newline = [round(x, config.SIG_DIGITS) if isinstance(x, float) else x 
                for x in line]
    linetext = ", ".join(str(x) for x in newline)
    line_length = len(linetext)
    linedict = {k:v for k, v in enumerate(newline)}
    if line_length > max_width:
        width_tally = int(2/3*max_width+4)
        a = textwrap.shorten(linetext, width= width_tally, placeholder='....')
        
        keep = []
        for k in sorted(linedict, reverse=True):
            width_tally += len(str(linedict[k]))
            if width_tally <= max_width:
                keep.append(k)
        keep.sort()

        b = ", ".join(str(linedict[k]) for k in keep)
        return(a + b)
    else:
        return(newline)

def wrap(text, width=75):
    wrapper = textwrap.TextWrapper(width)
    word_list = wrapper.wrap(text)
    row = ' '.join([str(elem + '\n') for elem in word_list])
    return('%s' % row)
    
def print_sheet(sheet, header, sheetname='', cut=config.ROWS_TO_DISPLAY ):
    """Function prints lists in matrix format for specified number of rows
    """
    num_rows = len(sheet)
    num_cols = prep.num_cols(sheet)
    string_list = []
    
    if len(sheetname) > 0:
        string_list.append("%s" % sheetname)
    string_list.append("%d Rows X %d Columns\n" % (num_rows, num_cols))
    string_list.append("Header: %s" % shorten_line(header))
    
    for row in range(num_rows):
        line = shorten_line(sheet[row])
        if row <= cut or row >= num_rows-cut:
            string_list.append("Row %02d: %s" % (row, line))
        elif row == cut + 1:
            string_list.append(" ".join("..." for x in sheet[row]))
            
    string_list.append("\n")
    return("\n".join(string_list))

def print_list(data, cut=config.ROWS_TO_DISPLAY ):
    """Function prints lists in matrix format for specified number of rows
    """
    num_rows = len(data)
    string_list = []
    
    for row in range(num_rows):
        line = shorten_line(data[row])[0]
        if row <= cut or row >= num_rows-cut:
            string_list.append(line)
        elif row == cut + 1:
            string_list.append("....")
                  
    return(pprint(string_list))