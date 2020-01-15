# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:48:14 2019

@author: Douglas Brown

Module contains the loss function(s) to be optimized for the algorithm
"""
import statistics
import prep
from bst import config

class Score:
    """Store the loss function score of the node"""
    instances = {}
    counter = 0
    
    def __init__(self, data, resp_col):
        Score.counter += 1
        self.resp = resp_col
        self.data = data
        self.loss = config.LOSS_FUNC
        self.weight = len(data)/config.N
        self.value = score_node(data, resp_col)
        
        key = (self.loss, self.resp, len(self.data))
        try:
            index = len(Score.instances[key])
            Score.instances[key].update({index: self})
        except KeyError:
            Score.instances[key] = {0: self}
            
    def memo(data, resp_col):
        key = (config.LOSS_FUNC, resp_col, len(data))
        if key in Score.instances:
            for index in Score.instances[key]:
                instance = Score.instances[key][index]
                if(instance.data == data):
                    return(instance)
                    
        return(Score(data, resp_col))

@config.func_timer       
def gini(rows, resp_col):
    """Calculate the Gini Impurity for a list of rows"""
    counts = prep.class_counts(rows, resp_col)
    impurity = 1   
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(rows)
        impurity -= prob_of_lbl**2  
    return(impurity)

@config.func_timer 
def gini_wgtd(left, right, resp_col):
    """Weighted Gini impurity based on size of child nodes"""
    p = float(len(left)) / (len(left) + len(right))
    g_left = gini(left, resp_col)
    g_right = gini(right, resp_col)
    return(p * g_left + (1 - p) * g_right)
 
@config.func_timer   
def var(rows, resp_col):
    y = [row[resp_col] for row in rows]   
    if len(y) > 1:
        return(statistics.variance(y))
    else:
        return(0)

@config.func_timer    
def var_wgtd(left, right, resp_col):
    """Weighted sum of squared residuals based on size of child nodes"""
    p = len(left) / (len(left) + len(right))
    g_left = var(left, resp_col)
    g_right = var(right, resp_col)
    return(p * g_left + (1 - p) * g_right)

@config.func_timer         
def score_node(rows, resp_col):
    if config.LOSS_FUNC == 'gini':
        return(gini(rows, resp_col))
    elif config.LOSS_FUNC == 'var':
        return(var(rows, resp_col))

@config.func_timer   
def info_gain(true_data, false_data, resp_col):
    if config.LOSS_FUNC == 'gini':
        return(gini_wgtd(true_data, false_data, resp_col))
    elif config.LOSS_FUNC == 'var':
        return(var_wgtd(true_data, false_data, resp_col))  

class Gain:
    instances = {}
    counter = 0
    calc = {'gini': gini_wgtd, 'var': var_wgtd}
    
    def __init__(self, true_data, false_data, resp_col):
        Gain.counter += 1
        self.resp = resp_col
        self.left = true_data
        self.right = false_data
        self.loss = config.LOSS_FUNC
        self.value = Gain.calc[self.loss](true_data, false_data, resp_col)
        
        key = (len(true_data), len(false_data), resp_col)
        try:
            index = len(Gain.instances[key])
            Gain.instances[key].update({index: self})
        except KeyError:
            Gain.instances[key] = {0: self}
            
    def memo(true_data, false_data, resp_col):

        key = (len(true_data), len(false_data), resp_col)   
        if key in Gain.instances:
            for index in Gain.instances[key]:
                instance = Gain.instances[key][index]
                check = []
                check.append(instance.left == true_data)
                check.append(instance.right == false_data)
                if(all(check)):
                    return(instance)
                    
        return(Gain(true_data, false_data, resp_col))