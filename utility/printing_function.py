'''
Created on Aug 23, 2018

@author: en
'''
import os
from sklearn.externals import joblib
import numpy as np


def print_results(path):    
    for each in os.listdir(path):
        try:
            print(each, joblib.load('%s/%s/0/result_test.pkl'%(path, each)))
        except:
            print(each)

def print_results_rpnet(path):    
    for each in os.listdir(path):
        try:
            print(each, np.median(joblib.load('%s/%s/0/result_validation.pkl'%(path, each)), axis=1))
        except:
            print(each)

def write_done_to_config():
    for each in os.listdir('.'):
        if os.path.exists('%s/0/done.txt'%(each)): continue
        with open('%s/0/done.txt'%each, 'w') as f:
            f.write('')
            
    