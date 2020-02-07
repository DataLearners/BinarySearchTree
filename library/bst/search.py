# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import math, random
import bst, prep, times
from bst import lossfuncs
from statistics import mean
from memo import Memo

class SplitClass(Memo):
    """Split rows at a specific value of a feature"""
    
    def __init__(self, col_idx, value, resp_col, data, settings):
        Memo.__call__(self, col_idx, value, resp_col, 
                      data=data, settings=settings)
        self.resp = resp_col
        self.data = data
        self.form_question(col_idx, value, settings.header)
        self.split(data, self.question)
        self.calc_gain(self.true_rows, self.false_rows, resp_col)
        
    def form_question(self, col_idx, value, header):
        """Formulate the question for the split"""
        question = bst.Question(col_idx, value, header)
        self.question = question
        self.col_name = question.col_name
    
    def split(self, data, question):
        """Split the dataset into left and right pieces"""
        left, right = partition(data, question)
        self.true_rows, self.false_rows = left, right
        self.min_rows = min(len(self.true_rows), len(self.false_rows))
    
    def calc_gain(self, left, right, y):
        """Determine the information gained from the split"""
        redux = lossfuncs.info_gain(left, right, y, self.settings.loss_func)
        score = lossfuncs.Score(self.settings, self.data, y)
        self.info_gain = score.weight * (score.value - redux)

class BestSplit(Memo):
    """Find the optimal split based on the features selected"""
    
    def __init__(self, settings, features, data, resp_col):
        Memo.__call__(self, resp_col, features=features, 
                      data=data, settings=settings)
        self.gain = settings.min_gain # keep track of the best info gain
        self.question = None # keep track of the value that produced it
        self.n = len(data)
        self.min_gain = settings.min_gain
        self.min_rows = settings.min_rows
        self.search(features, data, resp_col)
     
    def search(self, features, data, resp_col):
        """Search for the optimal split amongst all the features"""
        for fcol in features:
            values = indices(data, fcol)
            if len(values) == 0:
                continue
            for val in values:
                split = SplitClass(fcol, val, resp_col, data, self.settings)
                big_enough = (split.min_rows >= self.settings.min_rows)
                gain_enough = (split.info_gain >= self.settings.min_gain)
                if not all([big_enough, gain_enough]):
                    continue
                if split.info_gain > self.gain:
                    self.gain, self.question = split.info_gain, split.question 

@times.func_timer
def find_best_split(settings, data, resp_col):
    """Find the best question to ask by iterating over every feature
    and calculating the information gain. The loss function is gini in the 
    case of Classification Trees and variance in the case of Regression Trees
    """
    curr_score = lossfuncs.Score(settings, data, resp_col).value
    
    #feature subset applies to Random Forest
    features = random.sample(settings.xcols, settings.n_features)
    features.sort()
          
    best = BestSplit(settings, features, data, resp_col)
    return(best.gain, best.question, curr_score)

@times.func_timer
def build_tree(settings, data, resp_col, idx=1):
    """Builds the tree.
    Base case: no further info gain. Since we can ask no further questions,
    we'll return a leaf.
    Otherwise: Continue recursively down both branches. Return a Decision
    node. The Decision node records the question and both branches.
    """
    gain, question, score = find_best_split(settings, data, resp_col)

    if question == None:
        leaf = bst.Leaf(resp_col, idx, score, data, settings)
        return(leaf)
    
    true_rows, false_rows = partition(data, question)
    child_true, child_false = 2*idx , 2*idx + 1 #Left child, Right child
    
    left = build_tree(settings, true_rows, resp_col, child_true)
    right = build_tree(settings, false_rows, resp_col, child_false)
    
    decision = bst.Decision(question, data, left, right, idx, score, gain)
    decision.tree = settings.tree_idx
    return(decision)

@times.func_timer
def partition(rows, question):
    """Partition dataset by answer to the class: question into 
    true rows and false rows"""
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return(true_rows, false_rows)

@times.func_timer
def indices(df, colnum):
    """Find the average of each unique pair of values for numeric variables.
    Return k-1 categories for string variables."""
    unique = prep.unique_vals(df, colnum)
    
    if any([type(elm)==str for elm in unique]):
        results = unique[1:]
    elif all([prep.is_numeric(elm) for elm in unique]):
        cuts = range(0, len(unique), 2)
        pair = lambda x: [x[i:i + 2] for i in cuts if len(x[i:i + 2]) == 2]
        pairs = pair(unique)
        results = [mean(group) for group in pairs]
    return(results)  

@times.func_timer     
def height(indices):
    """Size of a binary tree of height, h is 2^h - 1. The max index will be
    close to the size of the array. """
    idx = max(indices, default=1)
    h = round(math.log2(idx + 1))
    return(h)

@times.func_timer
def pull_nodes(tree_idx):
    nodes = {}
    for key in bst.Leaf.instances:
        for index in bst.Leaf.instances[key]:
            leaf = bst.Leaf.instances[key][index]
            if hasattr(leaf, 'tree') and leaf.tree == tree_idx:
                nodes[leaf.index] = leaf
            
    for key in bst.Decision.instances:
        for index in bst.Decision.instances[key]:
            decision = bst.Decision.instances[key][index]
            if hasattr(decision, 'tree') and decision.tree == tree_idx:
                nodes[decision.index] = decision
                
    treenodes = prep.sortdict(nodes, descending=False, sortbykeys=True)   
    return(treenodes)
    
def featDF(tree, data):
    feat_matrix = [[row[x] for x in tree.configs.xcols] for row in data]
    return(prep.ingest.ListDF(feat_matrix, tree.configs.xnames))    