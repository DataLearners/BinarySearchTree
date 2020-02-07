# -*- coding: utf-8 -*-
from prep import ingest
from prep import clean
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import random

def is_empty(any_structure):
    if any_structure:
        print('Structure is not empty.')
        return False
    else:
        print('Structure is empty.')
        return True

def num_cols(sheet):
    """ Find the number of columns in a sheet"""
    cols = [len(sheet[row]) for row in range(len(sheet))]       
    return(max(cols, default=0)) 
    
def colmin(xlist, colnum):
    """Find the row containing the minimum of a column"""
    col = [row[colnum] for row in xlist]
    the_min = min(col)
    return([row for row in xlist if row[colnum]==the_min])
 
def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    vals = set([row[col] for row in rows])
    unique = sorted(list(vals))
    return(unique)  
    
def is_numeric(value):
    """Test if a value is numeric"""
    import numbers
    return(isinstance(value, numbers.Number))

def class_counts(rows, col):
    """Counts the number of each type in the column. Categorical data only"""
    counts = {}
    for row in rows:
        label = row[col]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def dec_counts(rows, col):
    """Provides proportions for each category of a column"""
    counts = class_counts(rows, col)
    total = sum(counts.values()) * 1.0
    prop = {}
    for lbl in counts.keys():
        prop[lbl] = counts[lbl] / total
    return prop

def flatten(nested_list):
    """Generator to flatten an arbitrarily nested list. To extract 
    results use list(flatten(x))"""
    for i in nested_list:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def get_col_ids(sheet): 
    """function to return the column indices of a list"""
    col_list = []
    for row in sheet:
        for col, row in enumerate(row):
            if col not in col_list:
                col_list.append(col)
    return(col_list)
    
def get_key(val, my_dict): 
    """function to return key for any value"""
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

def sortdict(my_dict, descending=True, sortbykeys=False):
    """Sorts dictionary by key or values"""
    if sortbykeys:
        keys = sorted(my_dict.keys(), reverse=descending) 
        return({k:my_dict[k] for k in keys})
        
    values = sorted(my_dict.values(), reverse=descending)
    return({get_key(v, my_dict):v for v in values})

def first_n_pairs(index, my_dict, descending=True):
    """get the first k,v pairs of a dictionary"""   
    sorted_dict = sortdict(my_dict, descending, sortbykeys=False)
    keys = list(sorted_dict.keys())
    pairs = {k:sorted_dict[k] for k in keys[:index]}
    return(pairs) 

def dup_rows(rows):
    """Find and return all of the duplicate rows in a data set"""
    seen, dupes = {}, []
    for row in rows:
        if str(row) not in seen:
            seen[str(row)] = 1
        else:
            if seen[str(row)] == 1:
                dupes.append(row)
            seen[str(row)] += 1    
    return(dupes)
    
def left_outer_exclude_b(a, b):
    """Similar to a sql left outer join excluding B this function finds all
    rows in a that are not in b."""
    in_a = {str(row):0 for row in a}
    for row in b:
        if str(row) in in_a:
            in_a[str(row)] +=1
    exclusive = [row for row in a if in_a[str(row)] == 0]
    return(exclusive)
    
def bootstrap(rows):
    """Create a bootstrapped replica of a data set. Function returns the
    same number of rows as the original data set. Bootstrapping samples 
    rows from the original data set with replacement."""
    output = []
    for i in range(len(rows)):
        rand_data = [(random.random(), row) for i, row in enumerate(rows)]
        the_min_row = colmin(rand_data, 0)[0][1]
        output.append(the_min_row)  
    return(output)
        
def split_train_test(data, tst_prop=0.5):
    """Split a data set on its rows into test and train data sets."""
    random.shuffle(data)
    train_size = int(len(data) - tst_prop*len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    return(train_data, test_data)    
    
def to_tuple(x):
    """Convert object to its tuple equivalent"""
    if isinstance(x, list):
        result = tuple(flatten(x))
        return(result)
    if isinstance(x, dict):
        result = flatten(x.items())
        return(tuple(result))
    if isinstance(x, int):
        return(x)
    