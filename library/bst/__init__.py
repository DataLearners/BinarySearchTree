# -*- coding: utf-8 -*-
#package setup
from bst import config
from bst import search
from bst import diagnostics
from bst import display
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown

Loss refers to the loss function used in the model. Classification Trees use 
a Gini Impurity loss function. Regression Trees use Variance.
"""
import prep, neat
import scipy.stats

import neat.config as neat_config
neat_config.MAX_WIDTH = 70
neat_config.ROWS_TO_DISPLAY = 10
neat_config.SIG_DIGITS = 3

def configure(data, resp, header, loss, min_gain, min_rows):
    config.RESP = resp
    config.HEADER = header
    config.LOSS_FUNC = loss
    config.MIN_GAIN = min_gain
    config.MIN_ROWS = min_rows
    
    config.TREE_ID += 1
    config.N = len(data)
    config.RESP_NAME = header[resp]
    
    config.PredHead = ["Leaf", "y_hat", header[resp], "Interval"]
    
    config.Xcols = [i for i in range(len(data[0])) if i != resp]
    config.xnames = [header[i] for i in config.Xcols]
    config.N_FEATURES = len(config.Xcols)

class Question:
    """A question is used to partition a dataset into 
    true and false answers"""
    
    def __init__(self, col_num, value):
        self.col_num = col_num
        self.value = value
        self.col_name = config.HEADER[col_num]
               
    def match(self, row):
        """Compare the feature value in a row to the 
        feature value in question."""
        val = row[self.col_num]
        if prep.is_numeric(val):
            return(val >= self.value)
        elif type(val) == str:
            return(val == self.value)

    def __repr__(self):
        condition = "=="
        val = self.value
        if prep.is_numeric(self.value):
            val = round(self.value, config.SIG_DIGITS)
            condition = ">="
            
        str_val = str(val)
        return("Is {} {} {}?".format(self.col_name, condition, str_val))

class Leaf:
    """A Leaf node predicts the response based on the data. It predicts
    mode in the case of Classification Tree and mean in the case of 
    Regression Tree"""
    instances = {}
    counter = 0
    mode = lambda x: max(set(x), key=x.count)
    predict = {'gini': mode, 'var': scipy.mean}
    
    def __init__(self, data, resp_col, idx, score):
        Leaf.counter += 1
        self.data = data
        self.resp_col = resp_col
        self.y_true = [row[resp_col] for row in data]
        self.n = len(data)
        self.index = idx
        self.score = score
        self.prediction = self.predict[config.LOSS_FUNC](self.y_true)
        
        key = (resp_col, idx, score, len(data))    
        try:
            index = len(Leaf.instances[key])
            Leaf.instances[key].update({index: self})
        except KeyError:
            Leaf.instances[key] = {0: self}
        
    def memo(data, resp_col, idx, score):
        key = (resp_col, idx, score, len(data))
        
        if key in Leaf.instances:
            for index in Leaf.instances[key]:
                instance = Leaf.instances[key][index]
                if data == instance.data:
                    return(instance)
                   
        return(Leaf(data, resp_col, idx, score))
             
class Decision:
    """A Decision Node asks a question. The question results in a true branch
    and a false branch in response to the question."""
    instances = {}
    counter = 0

    def __init__(self, question, data, true_, false_, idx, score, gain):
        Decision.counter += 1
        self.question = question
        self.true_branch = true_
        self.false_branch = false_
        self.data = data
        self.n = len(data)
        self.index = idx
        self.score = score
        self.gain = gain
        
        key = (question.col_name, question.value, len(data), idx, score, gain)
        try:
            index = len(Decision.instances[key])
            Decision.instances[key].update({index: self})
        except KeyError:
            Decision.instances[key] = {0: self}
        
    def memo(question, data, true_, false_, idx, score, gain):
        key = (question.col_name, question.value, len(data), idx, score, gain)
        if key in Decision.instances:
            for index in Decision.instances[key]:
                check = []
                instance = Decision.instances[key][index]
                check.append(instance.data == data)
                check.append(vars(instance.true_branch) == vars(true_))
                check.append(vars(instance.false_branch) == vars(false_))
                if(all(check)):
                    return(instance)
                    
        return(Decision(question, data, true_, false_, idx, score, gain))

tree_types = {'var':'RegressionTree', 'gini':'ClassificationTree'}
class Tree:
    """Decision Tree class. Performs binary search. Binary refers to how the
    features are seperated using Questions. Classification Trees and 
    Regression Trees depend on the loss function type."""
    nodes = {}
    
    def __init__(self, data, resp, header, loss, min_gain=0, min_rows=1):
        configure(data, resp, header, loss, min_gain, min_rows)
        self.index = config.TREE_ID
        self.type = tree_types[loss]
        self.y = resp
        self.x = config.Xcols
        self.train_size = len(data)
        
        self.model = search.build_tree(data, resp)
        self.nodes = search.pull_nodes(self.index)
        self.feature_importance = diagnostics.score_features(self)
        
        self.tree_array = diagnostics.tree_summary(self)
        self.leaves, self.decisions = diagnostics.classify_nodes(self)
        self.marked_data = diagnostics.mark(self)
        
        for idx, node in self.nodes.items():
            node.feature_row = diagnostics.indicators(self, idx)
            
        self.leaf_matrix = diagnostics.leaf_summary(self)
        
    def __call__(self, index=1):
        return(self.nodes[index])
   
    def __str__(self):
        attr = "{}".format([x for x in dir(self) if "__" not in x])
        return("Tree Class attributes\n {}".format(neat.wrap(attr)))
    
    def traverse_to(self, index=0):
        return(display.traverse(self, index))
        
    def predict(self, test_data, conf=0.95):
        """Generate a set of predictions on a data set."""
        node = lambda row: classify(row, self.model)
        int_ = lambda row: conf_interval(node(row).y_true, conf, self.type)
        
        ranges = [int_(row) for row in test_data]
        y_ = [row[self.y] for row in test_data]
        y_hat = [node(row).prediction for row in test_data]
        leaves = [node(row).index for row in test_data]
        p_ = len(self.feature_importance)
        
        if self.type == 'RegressionTree':
            diagnosis = diagnostics.regression(y_hat, y_, ranges, p_)
            self.error, self.r_sq, self.adj_rsq, self.mse, message = diagnosis
        elif self.type == 'ClassificationTree':
            diagnosis = diagnostics.classification(y_hat, y_, ranges, p_)
            self.error, message = diagnosis

        print(message)
        feat_matrix = [[row[x] for x in self.x] for row in test_data]
        names = [config.HEADER[x] for x in self.x]
        pDF = prep.ingest.ListDF(feat_matrix, names)

        row = lambda i: [leaves[i], y_hat[i], y_[i], ranges[i]]
        pred = [row(i) for i in range(len(test_data))]
        pDF.paste(pred, config.PredHead)
        
        return(pDF)

@config.func_timer 
def classify(row, node):
    """Recursive function for returning Leaf node.
    Base case: reach a Leaf node
    Recursion: traverse through Decision Nodes along branches until Base Case
    """
    if isinstance(node, Leaf):
        return(node)
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

@config.func_timer   
def conf_interval(y_true, conf, tree_type='RegressionTree'):
    """For the variance loss function, return the mean of the y feature 
    and a confidence interval around the mean."""
    if tree_type != 'RegressionTree': 
        return(None, None)
    
    mu, se = scipy.mean(y_true), 0
    if len(y_true) > 1:
        se = scipy.stats.sem(y_true)
    if se == 0:
        return(0, 0)
    
    low, high = scipy.stats.norm.interval(conf, mu, se)
    return(low, high)
