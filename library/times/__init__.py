# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 01:16:29 2019

@author: Douglas Brown
"""
SIG_DIGITS = 3
import time, matplotlib.pyplot as plt, numpy as np
import prep

def func_timer(func):
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
#        TIMES.func_times[func.__name__] = round(run_time, SIG_DIGITS)   
        try:
            Timer.func_times[func.__name__] += round(run_time, SIG_DIGITS)
        except KeyError:
            Timer.func_times[func.__name__] = round(run_time, SIG_DIGITS)
        return(value) #run the function as usual 
    return wrapper_timer

def plot_size(w_, h_):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = w_
    fig_size[1] = h_
    plt.rcParams["figure.figsize"] = fig_size
    
class Timer:
    func_times = {}
    
    def __init__(self):
        self.func_times = prep.sortdict(Timer.func_times)
        
    def __call__(self, index=0):
        items = list(self.func_times.items())
        Timer.longest_func = items[0]
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
    
    
    
    
    
    
    
    
    