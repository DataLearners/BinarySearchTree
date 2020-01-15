# -*- coding: utf-8 -*-
"""
Functions for cleaning worksheets
"""
import prep

def scrub(sheet):
    """ Function cleans an individual csv sheet. 
    1. Convert strings to numeric where appropriate
    2. Remove blank rows
    3. Find the header row
    """
    new_sheet = str_to_num(sheet)
    new_sheet = remove_blank_rows(new_sheet)
    new_sheet = new_sheet[find_header(new_sheet):]
    return(new_sheet)
      
def find_header(sheet):
    """ 
    The idea is that the header row is made up of only strings and it is a
    complete row, i.e. number of string columns match the maximum number of
    columns in the dataset. Rows could be blank or incomplete so the function
    measures the length of each row.
    """
    cols = prep.num_cols(sheet)
    for row_i in range(len(sheet)):
        row = sheet[row_i]
        row_length = range(len(row))
        col_cnt = sum(1 for ii in row_length if type(row[ii]) == str)
        if col_cnt == cols:
            header_row = row_i
            break
        else: 
            header_row = 0
    return(header_row)
               
def remove_blank_rows(sheet):
    """ Blank rows are determined by the length of the row. Even a single
    digit will still register as a positive length.
    """
    del_rows = [ii for ii in range(len(sheet)) if len(sheet[ii]) == 0]
    new_sheet = [sheet[ii] for ii in range(len(sheet)) if ii not in del_rows]       
    return(new_sheet)

def str_to_num(sheet):
    """ 
    Turn strings into numbers including percents wherever possible. 
    Element cannot be empty string. -1 means last element of list
    """
    for ii in range(len(sheet)):
        for jj in range(len(sheet[ii])):
            elm = sheet[ii][jj]
            if type(elm) == str and len(elm) > 0:
                try:
                    sheet[ii][jj] = float(elm.replace(",",""))
                except ValueError:
                    pass
                if elm[-1] == "%": 
                    sheet[ii][jj] = float(elm[:-1])/100. 
    return(sheet)
    
def delete_cols(sheet, select_cols):
    """Function deletes the columns of a list. Sorting the select_cols in 
    reverse order is essential to the success of the function. Once a column
    is deleted the number of columns available changes.
    """
    try:
        select_cols.sort(reverse=True)
        for row in sheet:
            for ii in select_cols:
                del row[ii]
    except:
        print("Columns selected for deletion must be a list")
    return(sheet)

def remove_blank_cols(sheet):
    """Function determines how sparse the column is across all rows. Columns
    that have greater sparseness than the percentage threshold are deleted.
    """
    cols = prep.num_cols(sheet)
    rows = len(sheet)
    columnloss = [0 for ii in range(cols)]
    
    for ii in range(cols):
        for row in sheet:
            if len(str(row[ii])) == 0:
                columnloss[ii] += 1/rows            
    emptycols = [ii for ii in range(cols) if columnloss[ii] > 0.9]
    cleanedsheet = delete_cols(sheet, emptycols)
    return(cleanedsheet)

def unstack(sheet, col_to_unstack, grp_col):
    """Function unstacks a column using the grp_col. Each unique 
    element of the grp_col becomes a new column. The values in the
    column to unstack are spread to each new group column. 
    """
    groups = [row[grp_col] for row in sheet[1:]]
    colnames = list(set(groups))

    for colname in colnames:
        sheet[0].append(colname)
        for row in sheet[1:]:
            if row[grp_col] == colname:
                row.append(row[col_to_unstack])
            else:
                row.append('')
                
    delete_cols(sheet, [col_to_unstack, grp_col])
    return(sheet)
    