# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:26:07 2021

@author: Bill
"""
import glob
import shutil, os
print(os.path.dirname(os.path.abspath("__file__")))
print(os.path.pardir)
print(os.path.join(os.path.dirname("__file__"),os.path.pardir))
print(os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir)))

myfiles = glob.glob('../1' + '/*.JPG') 
print(myfiles)