# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import os, csv
import prep
from prep import clean
import neat

class Folder:
    """Ingests all csv files in a specified folder. The object then stores
    multiple attributes of every file. """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.files = [filename for filename in os.listdir(dirpath)]
        self.sheets = [loadcsv(dirpath, filename) \
                       for filename in os.listdir(dirpath)]
        self.filecount = len(self.files)
        self.sheets = [clean.scrub(sheet) for sheet in self.sheets]
        self.cols = [prep.num_cols(sheet) for sheet in self.sheets]
        self.rows = [len(sheet) for sheet in self.sheets]
        self.headers = [sheet[0] for sheet in self.sheets]
            
    def __call__(self, fileindex=0):
        sheet = self.sheets[fileindex][1:]
        header = self.headers[fileindex]
        sheetname = self.files[fileindex]
        print("%d files in the folder %s" % (self.filecount, self.dirpath))
        print(neat.print_sheet(sheet, header, sheetname))
        
    def collate(self):
        collated = collate_sheets(self.sheets)
        collated = clean.remove_blank_cols(collated)
        return(collated)
        
    def export(self):
        exportcsv(filename=self.files[0], data=self.sheets, 
                  folder=os.getcwd())
    
    def toListDF(self, fileindex=0):
        """Send a file to the ListDF class"""
        df = ListDF(self.sheets[fileindex][1:], self.headers[fileindex],
                    self.files[fileindex])
        return(df)

class ListDF:
    """a single data table object"""
    def __init__(self, rows, header, sheetname=''):
        self.data = rows
        self.header = header
        self.num_rows = len(rows)
        self.num_cols = prep.num_cols(rows)
        self.doctitle = sheetname 
        for i in range(len(header)):
            title = header[i]
            column = [row[i] for row in rows]
            setattr(self, title, column)
    
    def __call__(self, col_idx=0):
        return([row[col_idx] for row in self.data])
    
    def add_col(self, col_data, col_name, insert_spot=-1):
        """Insert a new column into the data table"""
        for i, row in enumerate(self.data):
            row.insert(insert_spot, col_data[i])       
        self.header.insert(insert_spot, col_name)
        self.__init__(self.data, self.header)

    def paste(self, data, names):
        """Paste data in front of data frame"""
        rows = [pair[0] + pair[1] for pair in zip(data, self.data)]
        header = names + self.header
        self.__init__(rows, header)
       
    def subset(self, col):
        """Create a subset ListDF from a column list of numbers"""
        rows = [[row[idx] for idx in col] for row in self.data]
        header = [self.header[idx] for idx in col]
        return(ListDF(rows, header))
    
    def transform_col(self, fct, col_name):
        ii = self.header.index(col_name)
        for row in self.data:
            row[ii] = fct(row[ii])
        self.__init__(self.data, self.header)
        
    def del_col(self, col=None, colname=''):
        """Allow input of a column number or a column title"""
        if colname in self.header:
            col = self.header.index(colname)
        [row.pop(col) for row in self.data]
        self.header.pop(col)
        self.__init__(self.data, self.header)
        
    def sort_col(self, col=None, colname='', descending=False):
        """Sort data based on column number or a column title"""
        if colname in self.header:
            col = self.header.index(colname)
        data = sorted(self.data, key=lambda x: x[col], reverse=descending)
        self.__init__(data, self.header)

    def export(self, filename='ListDataframe.csv', folder=os.getcwd()):
        rows = self.data.copy()
        if rows[0] == self.header:
            rows = self.data[1:]
        else:
            rows.insert(0, self.header)
        exportcsv(filename, rows, folder)
        n_ = self.num_rows
        print("{} rows exported to {}\n as {}\n".format(n_, folder, filename))
        
    def __repr__(self):
        return(neat.print_sheet(self.data, self.header, self.doctitle))

def collate_sheets(sheets):
    """Combine sheets together provided that the header rows all match.
    Function input is the class object loadedfiles
    """
    header = sheets[0][0]
    row_count = 0
    print("Sheets merging...")
    for ii in range(len(sheets)):
        sheet = sheets[ii]
        if ii == 0:
            rows = len(sheet)
            combined = sheet
            row_count += rows
        else:
            if sheet[0] == header:
                rows = len(sheet[1:])
                combined.extend(sheet[1:])
                row_count += rows
            else: 
                print("Sheet %d does not match" % ii)
        print("Sheet %03d Rows %03d %03d" % (ii, rows, row_count) )
    print("Sheet All Rows %03d\n" % row_count)
    return(combined)
    
def loadcsv(folder, filename):
    """folder is a filepath. filename includes the extension of the file.
    The function reads in csv files and returns a list.
    """
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, newline='') as csvfile:
            csv_list = list(csv.reader(csvfile))
            return(csv_list)
    except:
        print("Something went wrong with %s" % filename)
              
def exportcsv(filename, data, folder=os.getcwd()):
    """folder is a filepath. filename includes the extension of the file.
    The function exports a list as a csv.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, dialect = 'excel')
        wr.writerows(data)