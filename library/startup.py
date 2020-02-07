# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:31:26 2019
root refers to a folder path
@author: Douglas Brown
"""
import os
MY_DIR = ''

def find_root(name, path):
    for root, dirs, files in os.walk(path):
        if name in root:
            return(root)
        
def find_subroot(partial_path, path):
    """Returns the root containing all of the names in the partial path"""
    try:
        names = partial_path.split('/')
    except SyntaxError:
        print("Oops! Only use forwardslash('\\') for your input string.")
    possible_roots = []
    for name in names:
        possible_roots.append(find_root(name, path))
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
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)
