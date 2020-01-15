# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:04:49 2019

@author: Douglas Brown
"""
import bst, prep, neat
from bst import config
import scipy.stats

mode = lambda x: max(set(x), key=x.count)
estimate = {'gini': mode, 'var': scipy.mean}
frst_types = {'var':'RegressionForest', 'gini':'ClassificationForest'}
tree_types = {'var':'RegressionTree', 'gini':'ClassificationTree'}

class Forest:
    """Random Forest class. Constructs either Classification Trees or 
    Regression Trees based on the loss function. Classification Forests return
    the mode of the Classification Trees. Regression Forests return the mean
    of the Regression Trees."""
    
    def __init__(self, data, resp, header, loss, 
                 min_gain=0, min_rows=1, n_features=1, n_trees=10):
        self.n_features = n_features
        self.y = resp
        self.loss = loss
        self.type = frst_types[loss]
        self.tree_type = tree_types[loss]
        self.trees = {}
        
        for idx in range(n_trees):
            boot = prep.bootstrap(data)
            spill = prep.left_outer_exclude_b(data, boot)
            tree = bst.Tree(boot, resp, header, loss, min_gain, min_rows)
            self.trees[idx] = {'data': boot, 'out_of_bag': spill, 'tree': tree}

        self.out_bag_error = calc_out_of_bag_error(self)
        
    def __call__(self, index=0):
        return(self.trees[index]['tree'])
   
    def __str__(self):
        attr = "{}".format([x for x in dir(self) if "__" not in x])
        return("Forest Class attributes\n {}".format(neat.wrap(attr)))
    
    @config.func_timer 
    def predict(self, test_data, conf=0.95):
        """Generate a set of predictions on a data set."""
        leaf = lambda tree: bst.classify(row, self.trees[tree]['tree'].model)
                
        y_hat = []
        y_ = []
        ranges = []
        without_y = []
        for row in test_data:
            leaves = [leaf(tree) for tree in self.trees]
            y_hats_row = [node.prediction for node in leaves]
            y_hat.append(estimate[self.loss](y_hats_row))
            y_.append(row[self.y])
            interval = bst.conf_interval(y_hats_row, conf, self.tree_type)
            ranges.append(interval)
            feature_row = [row[i] for i in range(len(row)) if i != self.y]
            without_y.append(feature_row)
        
        p_ = self.n_features
        if self.tree_type == 'RegressionTree':
            diagnosis = bst.diagnostics.regression(y_hat, y_, ranges, p_)
            self.error, self.r_sq, self.adj_rsq, self.mse, message = diagnosis
        elif self.tree_type == 'ClassificationTree':
            diagnosis = bst.diagnostics.classification(y_hat, y_, ranges, p_)
            self.error, message = diagnosis

        print(message)
        row = lambda i: [y_hat[i], y_[i], ranges[i], without_y[i]]
        features = range(len(bst.config.HEADER))
        head_minus_y = [bst.config.HEADER[i] for i in features if i != self.y]
        titles = ["Prediction", "Y_true", "Interval", head_minus_y]
        header = list(prep.flatten(titles))
        
        pred = [list(prep.flatten(row(i))) for i in range(len(y_))]
        pDF = prep.ingest.ListDF(pred, header)
        
        return(pDF)    

@config.func_timer        
def calc_out_of_bag_error(forest):
    """With classification forests the function returns the error rate on 
    out of bag samples. With regression forests the function returns the 
    mean square error rate, RSquare value, and Adj RSquare value."""
    leaf = lambda tree: bst.classify(row, forest.trees[tree]['tree'].model)
    
    y_hat = []
    y_true = []
    for idx in forest.trees:
        sub_frst = {k:forest.trees[k] for k in forest.trees if k != idx}
        bag_data = forest.trees[idx]['out_of_bag']
        for row in bag_data:
            leaves = [leaf(tree) for tree in sub_frst]
            y_hats_row = [node.prediction for node in leaves]
            y_hat.append(estimate[forest.loss](y_hats_row))
            y_true.append(row[forest.y])
    
    p_ = forest.n_features
    ranges = [(0, 0) for x in range(len(y_hat))]    
    if forest.tree_type == 'RegressionTree':
        return(bst.diagnostics.regression(y_hat, y_true, ranges, p_)[0])
    
    return(bst.diagnostics.classification(y_hat, y_true, ranges, p_)[0])
 