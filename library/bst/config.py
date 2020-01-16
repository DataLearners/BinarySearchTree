# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 01:16:29 2019

@author: Douglas Brown
"""
#SEED = 0
SIG_DIGITS = 3
CONFIDENCE = 0
MIN_ROWS = 1
MIN_GAIN = 0
LOSS_FUNC = None
HEADER = ''
RESP = None
RESP_NAME = ''
N_FEATURES = 0
TREE_ID = 0
N = 0
FOREST = False

import time, prep, matplotlib.pyplot as plt, numpy as np
sort_func = lambda x: sorted(x, reverse=True)
sort_dict = lambda d: {prep.get_key(v, d):v for v in sort_func(d.values())}

class Timer:
    def __init__(self):
        self.func_times = {}
        
    def __call__(self, index=0):
        self.sorted = sort_dict(self.func_times)
        items = list(self.sorted.items())
        self.longest_func = items[0]
        msg = "{} is the {} longest function".format(items[index], index + 1)
        return(msg)
    
    def plot_times(self, index=5):
        to_plot = prep.first_n_pairs(index, self.func_times)
        categories = to_plot.keys()
        pos = np.arange(len(categories))
        vals = to_plot.values()
        
        plot_size(12, 4)
        plt.bar(pos, vals, align='center', alpha=0.5)
        plt.xticks(pos, categories)
        plt.xlabel('Function')
        plt.ylabel('Times (sec)')
        plt.title('{} Longest functions'.format(index))
        plt.show()

TIMES = Timer()

def func_timer(func):
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
#        TIMES.func_times[func.__name__] = round(run_time, SIG_DIGITS)   
        try:
            TIMES.func_times[func.__name__] += round(run_time, SIG_DIGITS)
        except KeyError:
            TIMES.func_times[func.__name__] = round(run_time, SIG_DIGITS)
        return(value) #run the function as usual 
    return wrapper_timer

def plot_size(w_, h_):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = w_
    fig_size[1] = h_
    plt.rcParams["figure.figsize"] = fig_size
    
    
    
    
    
    
    
    
    
    
    