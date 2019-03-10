#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:09:51 2019

@author: egod
"""

import threading
import time
from queue import Queue
from numpy import random
   
def job(l,q):
    name = threading.currentThread().getName()
    print('this is {} working..'.format(name))
    for i in range(len(l)):
        l[i] = l[i]**2
        time.sleep(random.uniform())
    q.put(l)
    print('{}\'s job finished!'.format(name))

 
def multi_thread():

    q = Queue()
    threads=[]
    data=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
    
    for i in range(4):
        t = threading.Thread(target=job,
                             name='T{}'.format(i),
                             args=(data[i],q),
                             )
        t.start()
        threads.append(t)
    
    for thread in threads:
        thread.join()
        
    results = []
    for _ in range(4):
        results.append(q.get())
    
    print(results)
    
if __name__ == '__main__':
    multi_thread()


