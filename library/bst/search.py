# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import math, random
import bst, prep
from bst import config, loss
from statistics import mean

class SplitClass:
    """Split rows at a specific value of a feature"""
    instances = {}
    counter = 0
    
    def __init__(self, data, col_idx, value, resp_col):
        SplitClass.counter += 1
        self.resp = resp_col
        self.data = data
        
        question = bst.Question(col_idx, value)
                
        self.question = question
        self.col_name = question.col_name
        self.true_rows, self.false_rows = partition(data, question)
        self.min_rows = min(len(self.true_rows), len(self.false_rows))
        
        redux = loss.Gain.memo(self.true_rows, self.false_rows, self.resp)
        score_ = loss.Score.memo(data, self.resp)
        self.info_gain = score_.weight * (score_.value - redux.value)
        
        key = (col_idx, value, resp_col, len(data), score_.value)
        try:
            index = len(SplitClass.instances[key])
            SplitClass.instances[key].update({index: self})
        except KeyError:
            SplitClass.instances[key] = {0: self}
    
    def memo(data, col_idx, value, resp_col):
        score = loss.Score.memo(data, resp_col).value
        
        key = (col_idx, value, resp_col, len(data), score)
        if key in SplitClass.instances:
            for index in SplitClass.instances[key]:
                instance = SplitClass.instances[key][index]
                if(instance.data == data):
                    return(instance)
                    
        return(SplitClass(data, col_idx, value, resp_col))

@config.func_timer
def build_tree(rows, resp_col, idx=1):
    """Builds the tree.
    Base case: no further info gain. Since we can ask no further questions,
    we'll return a leaf.
    Otherwise: Continue recursively down both branches. Return a Decision
    node. The Decision node records the question and both branches.
    """
    memo_split = memoize(find_best_split)
    gain, question, score = memo_split(rows, resp_col)
    if question == None:
        leaf = bst.Leaf.memo(rows, resp_col, idx, score)
        leaf.tree = config.TREE_ID
        return(leaf)
    
    true_rows, false_rows = partition(rows, question)
    child_true, child_false = 2*idx , 2*idx + 1 #Left child, Right child
    
    left = build_tree(true_rows, resp_col, child_true)
    right = build_tree(false_rows, resp_col, child_false)
    
    decision = bst.Decision.memo(question, rows, left, right, idx, score, gain)
    decision.tree = config.TREE_ID
    return(decision)

def memoize(f):
    memo = {}
    def helper(*args):
        key = tuple(map(prep.to_tuple, args))
        if key not in memo:            
            memo[key] = f(*args)
        return memo[key]
    return helper

@config.func_timer    
def find_best_split(rows, resp_col):
    """Find the best question to ask by iterating over every feature
    and calculating the information gain. The loss function is gini in the 
    case of Classification Trees and variance in the case of Regression Trees
    """
    best_gain = config.MIN_GAIN  # keep track of the best information gain
    best_question = None  # keep track of the value that produced it
    curr_score = loss.Score.memo(rows, resp_col).value
    
    #feature subset applies to Random Forest
    feature_idx = random.sample(config.Xcols, config.N_FEATURES)
    feature_idx.sort()
          
    for idx in feature_idx:
        values = indices(rows, idx)
        if len(values) == 0:
            continue
        for val in values:
            split = SplitClass.memo(rows, idx, val, resp_col)
            big_enough = (split.min_rows >= config.MIN_ROWS)
            gain_enough = (split.info_gain >= config.MIN_GAIN)
            if not all([big_enough, gain_enough]):
                continue
            if split.info_gain > best_gain:
                best_gain, best_question = split.info_gain, split.question

    return(best_gain, best_question, curr_score)

@config.func_timer
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

@config.func_timer
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

@config.func_timer     
def height(indices):
    """Size of a binary tree of height, h is 2^h - 1. The max index will be
    close to the size of the array. """
    idx = max(indices)
    h = round(math.log2(idx + 1))
    return(h)

@config.func_timer
def pull_nodes(tree_idx):
    nodes = {}
    for key in bst.Leaf.instances:
        for index in bst.Leaf.instances[key]:
            leaf = bst.Leaf.instances[key][index]
            if leaf.tree == tree_idx:
                nodes[leaf.index] = leaf
            
    for key in bst.Decision.instances:
        for index in bst.Decision.instances[key]:
            decision = bst.Decision.instances[key][index]
            if decision.tree == tree_idx:
                nodes[decision.index] = decision
            
    return(nodes)
    
    