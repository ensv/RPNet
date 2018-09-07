'''
Created on Jul 18, 2018

@author: sovann
'''

import os
import socket


class Setting():
    
    def __init__(self, folder_name):
        
        # path to store all the data, train log and other results
        log_dir = '/data/chercheurs/en/RPNet/'
        # dataset
        self.dataset_shuffle_size = 2500 # dataset shuffle buffle size 
        self.hostname = socket.gethostname()
        self.projectname = folder_name
        
        self.data_path = '%s/%s'%(log_dir, folder_name)
        print('data path is set to be ', self.data_path)
        try: os.makedirs(self.data_path)
        except: pass
        

if __name__ == '__main__':
    pass
