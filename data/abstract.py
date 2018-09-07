'''
Created on Jul 18, 2018

@author: sovann
'''
import os
import tensorflow as tf
import numpy as np

class AbstractData():
    '''
    classdocs
    '''

    def _maybe_download(self):
        raise NotImplementedError 
    
    
    def generate_annotation(self):
        raise NotImplementedError 
    
    
    def verify_annotation(self):
        # time to check if valdiation example is accidentally included in test or train set or not
        # other verification to be defined
        raise NotImplementedError 
    
        
    def save_to_tfrecord(self, train_test_phase, suffix = '', **data):
        
        # verify shape
        shapes = list(set([v.shape[0] for _, v in data.items()]))
        assert len(shapes) == 1
        
        # get all keys
        keys = [e for e in data.keys()]
        
        if suffix == '':
            filename = '%s/%s.tfrecord'%(self.path, train_test_phase)        
        else:
            filename = '%s/%s_%s.tfrecord'%(self.path, train_test_phase, suffix)

        with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
            for j in range(0, shapes[0]):
                tmp_data = {e:data[e][j] for e in keys}
                tmp_data = {k:tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.tostring()])) for k, v in tmp_data.items()}
                example = tf.train.Example(features=tf.train.Features(feature =tmp_data))
                tfrecord_writer.write(example.SerializeToString())

    
    def save_other_information(self, **data):
        raise NotImplementedError 
    
    
    def load_annotation(self, train_test_phase):
        raise NotImplementedError     
    
    def __init__(self, name, dir_, suffix = ''):
        '''
        Constructor
        '''
        self.name = name
        
        self.path = dir_
        
        try: os.mkdir(self.path)
        except: pass
        
        self._maybe_download()

        self.generate_annotation()
        self.verify_annotation()

        for train_test_phase in ['train', 'test', 'validation']:
            
            filename = '%s/%s.tfrecord'%(self.path, train_test_phase)
            
            if suffix != '':
                filename = '%s/%s_%s.tfrecord'%(self.path, train_test_phase, suffix)
            if os.path.exists(filename): 
                print('%s already exists ... '%filename)
                print('the program will skip ....')
                continue
            
            self.annot = self.load_annotation(train_test_phase)    
            self.data = self.process(self.annot)
            self.save_to_tfrecord(train_test_phase, suffix, **self.data)
            if train_test_phase == 'train':
                self.save_other_information(**self.data)
