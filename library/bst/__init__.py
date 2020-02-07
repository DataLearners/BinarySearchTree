# -*- coding: utf-8 -*-
#package setup
from bst import search
from bst import diagnostics
from bst import display
from bst import lossfuncs
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown

Loss refers to the loss function used in the model. Classification Trees use 
a Gini Impurity loss function. Regression Trees use Variance.
"""
import prep, neat
import scipy.stats
from memo import Memo

neat.config.MAX_WIDTH = 70
neat.config.ROWS_TO_DISPLAY = 10
neat.config.SIG_DIGITS = 3
        
class Question:
    """A question is used to partition a dataset into true and false answers"""
   
    def __init__(self, col_num, value, header):   
        self.col_num = col_num
        self.value = value
        self.col_name = header[col_num]
               
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
            val = round(self.value, 2)
            condition = ">="
            
        str_val = str(val)
        return("Is {} {} {}?".format(self.col_name, condition, str_val))

class Leaf(Memo):
    """A Leaf node predicts the response based on the data. It predicts
    mode in the case of Classification Tree and mean in the case of 
    Regression Tree"""
    
    def __init__(self, resp_col, idx, score, data, settings):
        Memo.__call__(self, resp_col, idx, score, data=data)
        
        self.resp_col = resp_col
        self.y_true = [row[resp_col] for row in data]
        self.n = len(data)
        self.index = idx
        self.score = score
        self.loss_func = settings.loss_func
        self.tree = settings.tree_idx
        
        mode = lambda x: max(set(x), key=x.count)
        estimate = {'gini': mode, 'var': scipy.mean}
        self.prediction = estimate[settings.loss_func](self.y_true)
             
class Decision(Memo):
    """A Decision Node asks a question. The question results in a true branch
    and a false branch in response to the question."""

    def __init__(self, question, data, true_, false_, idx, score, gain):
        Memo.__call__(self, idx, score, gain, question=question, 
                      true_branch=true_, false_branch=false_, data=data)
        self.n = len(data)
        self.index = idx
        self.score = score
        self.gain = gain

class Tree:
    """Decision Tree class. Performs binary search. Binary refers to how the
    features are seperated using Questions. Classification Trees and 
    Regression Trees depend on the loss function type."""
    nodes = {}
#    instances = {}
    counter = 0
    
    def __init__(self, data, resp_col, header, loss, 
                 min_gain=0, min_rows=1, forest=False, n_features=1):
        Tree.counter += 1
        self.configs = Settings(data, resp_col, header, loss, 
                           min_gain, min_rows, forest, n_features)        
        
        self.model = search.build_tree(self.configs, data, resp_col)
        self.nodes = search.pull_nodes(self.configs.tree_idx)
        self.height = search.height(self.nodes)
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
        settings = self.configs
        
        node = lambda row: classify(row, self.model)
        ytrue = lambda row: node(row).y_true
        loss = settings.loss_func
        interval = lambda row: conf_interval(ytrue(row), conf, loss)
        
        ranges = [interval(row) for row in test_data]
        y_actual = [row[settings.y] for row in test_data]
        y_hat = [node(row).prediction for row in test_data]
        leaves = [node(row).index for row in test_data]
        numpredictors = len(self.feature_importance)
        
        diagnostics.diagnose(self, y_hat, y_actual, ranges, numpredictors)

        pDF = search.featDF(self, test_data)
        row = lambda i: [leaves[i], y_hat[i], y_actual[i], ranges[i]]
        pred = [row(i) for i in range(len(test_data))]
        pDF.paste(pred, settings.predhead)
        
        return(pDF)

class Settings:
    """Settings for the Binary Search Tree that persist between all of the 
    modules in the library"""
    
    def __init__(self, data, resp, header, loss, min_gain, min_rows, 
                 forest, n_features):
        self.y = resp
        self.data = data
        self.loss_func = loss
        self.min_gain = min_gain
        self.min_rows = min_rows
        tree_types = {'var':'RegressionTree', 'gini':'ClassificationTree'}
        self.tree_type = tree_types[loss]
        self.tree_idx = Tree.counter
        self.N = len(data)
        self.xcols = [i for i in range(len(data[0])) if i != resp]
        self.header = header
        self.yname = header[resp]
        self.predhead = ["Leaf", "y_hat", self.yname, "Interval"]
        self.xnames = [header[i] for i in self.xcols]
        
        if not forest:
            self.n_features = len(self.xcols)
        else:
            self.n_features = n_features
            self.predhead = ["y_hat", self.yname, "Interval"]

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

def conf_interval(y_list, conf, loss_func='var'):
    """For the variance loss function, return the mean of the y feature 
    and a confidence interval around the mean."""
    if loss_func != 'var': 
        return(None, None)
    
    mu, se = scipy.mean(y_list), 0
    if len(y_list) > 1:
        se = scipy.stats.sem(y_list)
    if se == 0:
        return(0, 0)
    
    low, high = scipy.stats.norm.interval(conf, mu, se)
    return(low, high)

    
    
    

