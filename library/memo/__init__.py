# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:57:48 2020

@author: Douglas Brown
"""
import times

class Memo:
    """Memoization class. Stores class instances in memory at the time of 
    intialization. If the class is not stored in memory then it performs
    initialization of the instance.
    args evaluate as a list. kwargs evaluate as a dict. args become the
    key or unique identifier. kwargs become attributes of the class object 
    and resolve cases where the key is not truly unique.
    
    Memo.__init__(self, x, y, data= datalist)
    results in: key = (x, y) self.data = datalist"""
    
    def __init__(self):
        pass
           
    def __call__(self, *args, **kwargs):
        """Determine if the key has been previously created. 
        Instance is added to dictionary provided that kwargs are duplicative
        on the key searched."""
        classname = type(self)
        self.setcache(classname)
        key = tuple([arg for arg in args if not isinstance(arg, list)])
        self.keyexists = key in classname.instances
        memoized = self.does_exist(key, classname.instances, **kwargs)
        self.tag(**kwargs)
        
        if memoized:
            return(classname.instances[key][self.index])
            
        self.add_instance(key, classname)
        return(classname.instances[key][self.index])
        
    @times.func_timer
    def add_instance(self, key, classname):
        """Add the instance of self to instances"""
        classname.counter += 1
        if self.keyexists:
            self.index = len(classname.instances[key])
            classname.instances[key].update({self.index: self})
        else:
            self.index = 0
            classname.instances[key] = {self.index: self}
            
    @times.func_timer
    def does_exist(self, key, instances, **kwargs):
        """kwargs are additional attributes that make a key non-unique in 
        Memo.instances. Function determines whether kwargs are contained on 
        the second level (index) of instances[key][index]"""
        if not self.keyexists:
            return(False)
        for index in instances[key]:
            instance = instances[key][index]
            check = []
            for arg in kwargs:
                check.append(kwargs[arg] == instance.__dict__[arg])
                if(all(check)):
                    self.index = index
                    return(True)
                    
    @times.func_timer          
    def setcache(self, classname):
        """Initalize the class dictionary and instance counter if it does
        not already exist"""
        try:
            classname.instances
        except:
            classname.instances = dict()
            classname.counter = 0        
     
    @times.func_timer                                  
    def tag(self, **kwargs):
        """kwargs become attributes of the instance"""
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])

                