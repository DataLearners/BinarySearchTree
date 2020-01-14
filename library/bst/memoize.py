# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 21:56:52 2020

@author: Douglas Brown
"""

import bst, prep, random
from bst import config

@config.func_timer
def split(rows, col_idx, value):
    resp = config.RESP_NAME
    lookup = {'Y': resp, 'Rows': rows[0], 'Col': col_idx, 'Val':value}
    key = prep.get_key(lookup, bst.search.SplitClass.inputs)
    if key in bst.search.SplitClass.instances.keys():
        return(bst.search.SplitClass.instances[key])

    return(bst.search.SplitClass(rows, col_idx, value))

@config.func_timer
def leaf(rows, resp_col, idx, score):
    """Memoization of Leaf Class. Returns leaf if it exists or generates
    a new Leaf instance."""
    lookup = {'Y': resp_col, 'Rows': rows, 'Index': idx, 'Score': score}
    key = prep.get_key(lookup, bst.Leaf.inputs)
    if key in bst.Leaf.instances.keys():
        leaf = bst.Leaf.instances[key]
#        print("\tLeaf Memoized")
    else:
        leaf = bst.Leaf(rows, resp_col, idx, score)
            
    leaf.tree = config.TREE_ID
    return(leaf)

@config.func_timer
def decision(query, rows, true_, false_, idx, score, gain):
    """Memoization of Decision Class. Returns decision if it exists or 
    generates a new Decision instance."""
    lookup = {'Y': config.RESP_NAME, 'Q': query, 'Rows': rows,
              'T': true_, 'F': false_, 'Index': idx, 
              'Score': score, 'Gain': gain}
    key = prep.get_key(lookup, bst.Decision.inputs)
    if key in bst.Decision.instances.keys():
        decision = bst.Decision.instances[key]
    else:
        decision = bst.Decision(query, rows, true_, false_, idx, score, gain)
            
    decision.tree = config.TREE_ID
    return(decision)