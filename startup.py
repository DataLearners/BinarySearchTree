"""root refers to a folder path
startup is for finding, cleaning, and loading csv files into the workspace."""
import os
import sys
import csv
import numbers
MY_DIR = os.environ['USERPROFILE']


def find_root(name):
    for root, dirs, files in os.walk(MY_DIR):
        if name in root:
            print("{} Found".format(root))
            return(root)


def find_subroot(partial_path):
    """Returns the root containing all of the names in the partial path"""
    try:
        names = partial_path.split('/')
    except SyntaxError:
        print("Oops! Only use forwardslash('\\') for your input string.")
    possible_roots = []
    for name in names:
        possible_roots.append(find_root(name))
    for root in possible_roots:
        if all(name in root for name in names):
            return(root)


def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def make_folder(path):
    try:
        os.mkdir(path)
    except OSError:
#        print("Creation of the directory %s failed" % path)
        pass
    else:
        print("Successfully created the directory %s " % path)


def add_library(partial_path):
    """Adds the custom library folder to the sys.path"""
    my_library = find_root(partial_path)
    sys.path.append(my_library)
    sys.path = list(set(sys.path))
    sys.path.sort()


class Clean:
    def __init__(self, sheet, colsparseness=0.1, rowsparseness=0.5):
        self.colsparseness = colsparseness
        self.rowsparseness = rowsparseness
        self.sheet = self.scrub(sheet)
        self.header = self.find_header(self.sheet)
        self.numcols = self.find_numcols(self.sheet)
        self.numrows = len(self.sheet) - 1

    def scrub(self, sheet):
        """ Function cleans an individual csv sheet. Removes blank rows and
        blank columns; and converts strings to numeric where appropriate.
        """
        goodrows = self.remove_blank_rows(sheet)
        goodcols = self.remove_blank_cols(goodrows)
        scrubbed = self.str_to_num(goodcols)
        return scrubbed

    def find_numcols(self, sheet):
        """ Find the number of columns in a sheet"""
        cols = [len(sheet[row]) for row in range(len(sheet))]
        return max(cols, default=0)

    def fillsheet(self, sheet):
        """Fill missing values for the sheet so each row is the same length"""
        longest_row = max([len(row) for row in sheet])
        fill = lambda x: [None for i in range(longest_row - len(x))]
        filledsheet = [row + fill(row) for row in sheet]  # fill empties
        return filledsheet

    def find_header(self, sheet):
        """Function assumes that the header row is made up of only strings.
        The function finds the row with highest string content
        """
        filledsheet = self.fillsheet(sheet)
        stringdensity = lambda x: self.type_counts(x)['String']
        densities = [stringdensity(row) for row in filledsheet]
        header_row = densities.index(max(densities))
        return filledsheet[header_row]

    def remove_blank_rows(self, sheet):
        """Rows with sparseness above the accepted threshold are removed. """
        filledsheet = self.fillsheet(sheet)
        empties = self.find_blanks(filledsheet, threshold=self.rowsparseness)
        [filledsheet.pop(index) for index in empties]
        return filledsheet

    def remove_blank_cols(self, sheet):
        """Function calculates sparseness on the transpose of the sheet.
        Columns with sparesness above the threshold are removed."""
        filledsheet = self.fillsheet(sheet)
        rows = [list(tup) for tup in zip(*filledsheet)]

        empties = self.find_blanks(rows, threshold=self.colsparseness)
        [row.pop(index) for row in filledsheet for index in empties]
        return filledsheet

    def find_blanks(self, rows, threshold):
        """Function determines the sparseness of each row. Rows above the
        threshold are returned"""
        sparseness = lambda x: self.type_counts(x)['None']
        sparse = {index: sparseness(row) for index, row in enumerate(rows)}
        metrics = sparse.items()
        empties = [col for col, metric in metrics if metric > threshold]
        empties.sort(reverse=True)
        return empties

    def type_counts(self, simplelist):
        counts = {'String':0, 'None':0, 'Numeric':0}
        for elm in simplelist:
            if isinstance(elm, numbers.Number):
                counts['Numeric'] += 1
            if elm == None or len(str(elm)) == 0:
                counts['None'] += 1
            if elm == 'NULL':
                counts['None'] += 1
            elif type(elm) == str and len(str(elm)) > 0:
                counts['String'] += 1
        total = sum(counts.values())
        prop = {}
        for key in counts:
            prop[key] = counts[key]/total
        return prop

    def str_to_num(self, sheet):
        """
        Turn strings into numbers including percents wherever possible.
        Element cannot be empty string. -1 means last element of list
        """
        for i in range(len(sheet)):
            for j in range(len(sheet[i])):
                elm = sheet[i][j]
                if type(elm) == str and len(elm) > 0:
                    try:
                        sheet[i][j] = float(elm.replace(",", ""))
                    except ValueError:
                        pass
                    if elm[-1] == "%":
                        sheet[i][j] = float(elm[:-1])/100.
        return sheet


class Load:
    """Ingests all csv files in a specified folder. The object then stores
    multiple attributes of every file.
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.sheets = self.collate_sheets(dirpath)
        self.files = list(self.sheets.keys())
        self.rows = [len(sheet) - 1 for sheet in self.sheets.values()]
        self.headers = [sheet[0] for sheet in self.sheets.values()]

    def __call__(self, fileindex=0):
        """Method for exporting the data and header for the file"""
        file = self.files[fileindex]
        sheet = self.sheets[file][1:]
        header = self.sheets[file][0]
        return sheet, header

    def __repr__(self):
        filecount = len(self.files)
        txt = "{} files in the folder {}".format(filecount, self.dirpath)
        stringlist = [txt, self.files]
        return "\n".join([str(item) for item in stringlist])

    def loadcsv(self, folder, filename):
        """folder is a filepath. filename includes the extension of the file.
        The function reads in csv files and returns a list.
        """
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, newline='') as csvfile:
                csv_list = list(csv.reader(csvfile))
                return Clean(csv_list).sheet
        except:
            print("Something went wrong with {}".format(filename))

    def collate_sheets(self, dirpath):
        """Combine sheets together provided that the header rows all match.
        Function input is the class object loadedfiles
        """
        combined_sheets = []
        row_count = 0
        files = [filename for filename in os.listdir(dirpath)]
        print(files)
        sheets = {file: self.loadcsv(dirpath, file) for file in files}

        header = sheets[files[0]][0]
        collated = []
        combined_sheets.append(header)
        print("Sheets attempting merge...")

        zeros = lambda x, y: str(x).zfill(len(str(y)))  # leading zeros display
        for file, sheet in sheets.items():
            rows = len(sheet[1:])
            if sheet[0] == combined_sheets[0]:

                combined_sheets.extend(sheet[1:])
                row_count += rows
                collated.append(file)
            else:
                print("{} not merged".format(file))
                row_count += rows
            str_rows = zeros(rows, row_count)
            print("{} {} Rows {} Total Rows".format(file, str_rows, row_count))

        filelist = [file for file in sheets if file not in collated]
        new_sheets = {file: sheets[file] for file in filelist}
        collated_name = collated[0]
        new_sheets[collated_name] = combined_sheets
        print("{} Sheets {} Total Rows \n".format(len(new_sheets), row_count))
        return new_sheets
