# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:48:14 2019

@author: Douglas Brown

Module contains the loss function(s) to be optimized for the algorithm
"""
import statistics
import prep
import times
from memo import Memo

class Score(Memo):
    """Store the loss function score of the node"""
    def __init__(self, settings, data, resp_col):
        Memo.__call__(self, resp_col, settings=settings, data=data)
        self.resp = resp_col
        self.weight = len(data)/settings.N
        self.value = score_node(data, resp_col, settings.loss_func)

@times.func_timer       
def gini(data, resp_col):
    """Calculate the Gini Impurity for a list of rows"""
    counts = prep.class_counts(data, resp_col)
    impurity = 1   
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(data)
        impurity -= prob_of_lbl**2  
    return(impurity)

@times.func_timer 
def gini_wgtd(left, right, resp_col):
    """Weighted Gini impurity based on size of child nodes"""
    p = float(len(left)) / (len(left) + len(right))
    g_left = gini(left, resp_col)
    g_right = gini(right, resp_col)
    return(p * g_left + (1 - p) * g_right)
 
@times.func_timer   
def var(data, resp_col):
    y = [row[resp_col] for row in data]   
    if len(y) > 1:
        return(statistics.variance(y))
    else:
        return(0)

@times.func_timer    
def var_wgtd(left, right, resp_col):
    """Weighted sum of squared residuals based on size of child nodes"""
    p = len(left) / (len(left) + len(right))
    g_left = var(left, resp_col)
    g_right = var(right, resp_col)
    return(p * g_left + (1 - p) * g_right)

@times.func_timer         
def score_node(data, resp_col, loss):
    if loss == 'gini':
        return(gini(data, resp_col))
    elif loss == 'var':
        return(var(data, resp_col))

@times.func_timer   
def info_gain(true_data, false_data, resp_col, loss):
    if loss == 'gini':
        return(gini_wgtd(true_data, false_data, resp_col))
    elif loss == 'var':
        return(var_wgtd(true_data, false_data, resp_col))  

