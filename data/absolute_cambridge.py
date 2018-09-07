'''
Created on Jul 19, 2018

@author: sovann
'''
import sys, os
import cv2
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from data.abstract import AbstractData
import numpy as np
from abstract_network.setting import Setting

class AbsoluteCambridge(AbstractData):
    
    def __init__(self, name_):
        self.name = name_
        
        self.setting = Setting('absolute_cambridge')
        dir_ = self.setting.data_path + '/' + name_
        AbstractData.__init__(self, name_, dir_)

    def _maybe_download(self):
        
        if os.path.exists('%s/KingsCollege/dataset_train.txt'%self.setting.data_path): return
        
        current_dir = os.getcwd()
        os.chdir(self.setting.data_path)
        
        links = [
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip']

        for each in links:
            os.system('wget %s'%each)
        os.system("unzip '*.zip'")
        os.system(" find . -name '*.mp4' -type f -delete")
        os.system("rm -rf *.zip")

        os.chdir(current_dir)


    def write_2_text(self, annot, name):
        f = open('%s/%s_set.txt'%(self.path, name), 'w')
        f.write('Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R]\n\n')
        for each in annot:
            f.write(' '.join(each.tolist()) + '\n')
        f.close()

    def save_other_information(self, **data):
        pass
    
    def generate_annotation(self):
        # train and validation set
        annot = open('%s/dataset_train.txt'%(self.path)).readlines()[3:]
        names = np.array([e.split('/')[-2] for e in annot])
        annot = [e.split() for e in annot]

        unique_names, unique_counts = np.unique(names, return_counts = True)

        if unique_names.shape[0] >= 3:
            index = np.argsort(unique_counts)
            unique_names = unique_names[index][0]
            valid_index = np.where(names == unique_names)[0]
        else:
            index = np.arange(names.shape[0])
            np.random.shuffle(index)  # @UndefinedVariable
            valid_index = index[:int(index.shape[0]*0.2)]

        train_index = np.delete(np.arange(names.shape[0]), valid_index)

        for each in annot:
            each[0] = '%s/%s/'%(self.setting.data_path, self.name) + each[0]

        annot = np.array(annot)

        valid_annot = annot[valid_index]
        train_annot = annot[train_index]
        
        self.write_2_text(valid_annot, 'validation')
        self.write_2_text(train_annot, 'train')

        annot = open('%s/dataset_test.txt'%(self.path)).readlines()[3:]
        annot = [e.split() for e in annot]
        for each in annot:
            each[0] = '%s/%s/'%(self.setting.data_path, self.name) + each[0]
        annot = np.array(annot)
        self.write_2_text(annot, 'test')
        
        
    def verify_annotation(self):

        train = open('%s/train_set.txt'%(self.path)).readlines()[3:]
        train = [e.split()[0] for e in train]
        
        test = open('%s/test_set.txt'%(self.path)).readlines()[3:]
        test = [e.split()[0] for e in test]
        
        valid = open('%s/validation_set.txt'%(self.path)).readlines()[3:]
        valid = [e.split()[0] for e in valid]
        
        for e in train:
            assert e not in test
            assert e not in valid
            
        for e in valid:
            assert e not in test
            assert e not in train
        
    def load_annotation(self, train_test_phase):
        annot = open('%s/%s_set.txt'%(self.path, train_test_phase)).readlines()[3:]
        annot = np.array([e.split() for e in annot])
        return annot
    
    def process(self, annot):
        images = np.zeros((len(annot), 256, 455, 3), 'uint8')
        for index, (path, _, _, _, _, _, _, _) in enumerate(annot):
            im = cv2.imread(path)  # @UndefinedVariable
            im = cv2.resize(im, (455, 256))  # @UndefinedVariable
            images[index] = im
        images = images.astype('f4')
        images = images - np.mean(images, axis=0)
        return {'image': images, 'pose': annot[:, 1:].astype('f4')}


if __name__ == '__main__':
    for subset in  ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
        AbsoluteCambridge(subset)
