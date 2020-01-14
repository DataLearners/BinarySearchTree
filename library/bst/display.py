# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:25:53 2019

@author: Douglas Brown
"""
import math, bst
from bst import config

@config.func_timer 
def find_branch(i, ancestors = None):
    """Recursive function to find the ancestors of a particular node
    ancestors = None is necessary to keep the recursive function from 
    storing ancestors between function calls which resuts in duplicates"""
    if ancestors == None:
        ancestors = []
    parent = math.floor(i/2)
    if parent == 0:
        ancestors.append(i)
        return sorted(ancestors)
    ancestors.append(i) #Due to recursion its i and not parent
    return find_branch(parent, ancestors)

@config.func_timer 
def get_answers(position):
    """Record all of the answers to questions that occur on a branch.
    Yes -> 2k, No -> 2k+1 where k is the preceding node
    """
    nxtnode = lambda x: [x[(i_ + 1) % len(x)] for i_ in range(len(x))]
    decide = lambda a, b:'Yes' if b == 2*a else ('No' if b == 2*a + 1 else '')
    
    branch = find_branch(position)
    outcomes = nxtnode(branch)  
    return(list(map(decide, branch, outcomes)))

@config.func_timer 
def print_tree(node, spacing="" ):
    """print_tree(my_tree.model)"""
    if isinstance(node, bst.Leaf):
        print (spacing + "Predict", node.prediction)
        return
    
    print (spacing + str(node.question))
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ") 

@config.func_timer 
def print_counts(counts):
    """A nicer way to print count values in a dictionary."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = '{:.0%}'.format(counts[lbl] / total)
    return probs 

@config.func_timer  
def traverse(tree, position):
    """Given a node in the tree recreate the decisions that
    occur to reach the node. """
    path, answers = find_branch(position), get_answers(position)
    pairs = [pair for pair in zip(path,answers)]
    
    print("Traversing the tree to node {}...".format(position))
    for pair in pairs:
        idx, answer = pair[0], pair[1]
        score = round(tree(idx).score, config.SIG_DIGITS)
        print("Node({}) Rows {} Score {}".format(idx, tree(idx).n, score))
        if isinstance(tree(idx), bst.Decision):
            print("\t{} {}".format(tree(idx).question, answer))
        elif isinstance(tree(idx), bst.Leaf):  
            print("\tLeaf Node Prediction {}\n".format(tree(idx).prediction))
   
