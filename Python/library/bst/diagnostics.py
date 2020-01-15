# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import bst, prep
from bst import config
from sklearn.metrics import r2_score, mean_squared_error

flat = lambda x: list(prep.flatten(x))

@config.func_timer   
def score_features(tree, standardize=True):
    """Find the feature importance in the tree"""
    importance = {x:0 for x in config.HEADER}
    for idx, node in tree.nodes.items():
        if isinstance(node, bst.Decision):
            feature = node.question.col_name
            importance[feature] += node.gain
    
    scores = {}
    for key, val in importance.items():
        if val > 0:
            scores[key] = val
    
    if standardize:
        vals = sorted(scores.values(), reverse=True)
        key = lambda x: prep.get_key(x, scores)
        percents = {key(v):v/sum(vals) for v in vals}
        return(percents)

    return(scores)
        
@config.func_timer 
def indicators(tree, idx):
    """Identifies the features used for making predictions on a leaf. Function
    returns an indicator vector for all of the features"""
    if isinstance(tree.nodes[idx], bst.Leaf):
        feature_row = [0 for i in config.HEADER]
        indices = bst.display.find_branch(idx)[:-1]
        for key in indices:
            feature_row[tree.nodes[key].question.col_num] = 1
    
        feature_row.pop(config.RESP)
        return(feature_row) 

@config.func_timer 
def leaf_summary(tree):
    """Summary of the features in each leaf"""
    header = flat(["Leaf", "Rows", "Score", config.xnames])
    matrix = []
    for idx, node in tree.nodes.items():
        if isinstance(node, bst.Leaf):
            row = flat([node.index, node.n, node.score, node.feature_row])
            matrix.append(row)
            
    return(prep.ingest.ListDF(matrix, header))

@config.func_timer  
def mark(tree):
    """Add column to the training data that specifies the Leaf for each row"""
    
    header = flat(["Leaf_idx", bst.config.HEADER])
    marked_data = []
    for idx, node in tree.nodes.items():
        if isinstance(node, bst.Leaf):
            for row in node.data:
                labeled_row = flat([idx, row])
                marked_data.append(labeled_row)
            
    return(prep.ingest.ListDF(marked_data, header)) 

@config.func_timer 
def tree_summary(tree):
    """Summary of the nodes in the tree"""
    row = lambda x:[x.index, "Decision", str(x.question), x.n, x.score, x.gain]
    header = ["Index", "Type", "Question", "Rows", "Score", "Gain"]
    
    data = []
    for idx, node in tree.nodes.items():
        if isinstance(node, bst.Leaf):
            data.append([node.index, "Leaf", "", node.n, node.score, 0])
        if isinstance(node, bst.Decision):
            data.append(row(node))
    
    summaryDF = prep.ingest.ListDF(data, header)
    summaryDF.sort_col(colname="Index")
    return(summaryDF)

@config.func_timer 
def classify_nodes(tree):
    """Find all indices for Decision and Leaf nodes"""
    leaves, decisions = [], []
    
    for idx, node in tree.nodes.items():
        if isinstance(node, bst.Leaf):
            leaves.append(idx)
        if isinstance(node, bst.Decision):
            decisions.append(idx)

    return(sorted(leaves), sorted(decisions))

@config.func_timer    
def regression(y_hat, y_true, intervals, predictors):
        n_ = len(y_hat)
        r_sq = r2_score(y_true, y_hat)
        adj_rsq = 1 - (1 - r_sq)*(n_ - 1)/(n_ - predictors - 1)
        mse = mean_squared_error(y_true, y_hat)
        
        btwn = lambda x, pair: pair[0] <= x <= pair[1]
        accuracy = [btwn(y_true[i], pair) for i, pair in enumerate(intervals)]
        error = accuracy.count(False)/n_
        
        txt = "Testing on {0[0]} rows Accuracy {0[1]:.2%}"
        msg_basic = txt.format([n_, 1-error])
        txt = "RSq {0[0]:.0%} AdjRSq {0[1]:.0%} MSE {0[2]:.2f}\n"
        msg_reg = txt.format([r_sq, adj_rsq, mse])
        msg = msg_basic + "\n" + msg_reg
        
        return(error, r_sq, adj_rsq, mse, msg)
        
@config.func_timer    
def classification(y_hat, y_true, intervals, predictors):
        n_ = len(y_hat)
        accuracy = [y_hat[i] == y_true[i] for i in range(n_)]
        error = accuracy.count(False)/n_
        
        msg = "Testing on {} rows Accuracy {:.2%}\n".format(n_, 1-error)
        return(error, msg)
        
      