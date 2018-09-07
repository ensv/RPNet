'''
Created on Jul 23, 2018

@author: en
'''
import sys
import os
import numpy as np

if __name__ == '__main__':
    path = sys.argv[1]
    
    for each in os.listdir(path):
        for e in os.listdir(path + '/' + each):
            try: 
                int(e)
            except: 
                continue

            try:
                cpt = [e for e in open('%s/%s/%s/checkpoint'%(path, each, e), 'r').readlines() if e.startswith('all_model_checkpoint_paths')]
            except:
                continue
            cpt = [e.split('"')[1] for e in cpt]
            cpt = np.array([(int(e.split('model.ckpt-')[1]), e) for e in cpt if len(e.split('model.ckpt-')) == 2])
            
            index = np.argsort(cpt[:, 0].astype('i4'))
            cpt = cpt[index]
            
            for _, file_path in cpt[:-30]:
                try:
                    os.remove('%s.data-00000-of-00001'%file_path)
                    os.remove('%s.index'%file_path)
                    os.remove('%s.meta'%file_path)
                except:
                    pass