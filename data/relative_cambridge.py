'''
Created on Jul 19, 2018

@author: sovann
'''
import sys, os
import cv2
import ctypes
import multiprocessing
from multiprocessing.pool import Pool
from pyquaternion.quaternion import Quaternion
import shutil
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from data.abstract import AbstractData
import numpy as np
from abstract_network.setting import Setting

images = None

def create_share_ndarray(shape, dtype, cdtype):

    total_shape = 1
    for ele in shape:
        total_shape *= ele
    shared_base = multiprocessing.Array(cdtype, total_shape)  # @UndefinedVariable
    shared_array = np.frombuffer(shared_base.get_obj(), dtype=dtype)
    shared_array = np.reshape(shared_array, shape)
    assert shared_array.base.base is shared_base.get_obj()
    return shared_array

def create_share_from_arr(arr, cdtype):

    shared_arr = create_share_ndarray(arr.shape, arr.dtype, cdtype)
    shared_arr[:] = arr[:]
    return shared_arr


def _read_image(param):
    index, each = param
    
    if not os.path.exists(each[0]):
    
        im1 = cv2.imread(each[0].replace('/data/chercheurs/en', '/home/2017025/sen01'))  # @UndefinedVariable
        im1 = cv2.resize(im1, (455, 256))  # @UndefinedVariable
        
        im2 = cv2.imread(each[8].replace('/data/chercheurs/en', '/home/2017025/sen01'))  # @UndefinedVariable
        im2 = cv2.resize(im2, (455, 256))  # @UndefinedVariable
    else:
        
        im1 = cv2.imread(each[0])  # @UndefinedVariable
        im1 = cv2.resize(im1, (455, 256))  # @UndefinedVariable
        
        im2 = cv2.imread(each[8])  # @UndefinedVariable
        im2 = cv2.resize(im2, (455, 256))  # @UndefinedVariable
    
    images[index] = np.array([im1, im2])



class RelativeCambridge(AbstractData):
    
    def __init__(self, name_):
        self.name = name_
        
        self.setting = Setting('relative_cambridge')
        dir_ = self.setting.data_path + '/' + name_
        AbstractData.__init__(self, name_, dir_)

    def _maybe_download(self):
        
        if os.path.exists('%s/KingsCollege'%self.setting.data_path): return
        
        current_dir = os.getcwd()
        os.chdir(self.setting.data_path)
        
        links = ['https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
                'https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip']

        shutil.copy('%s/rpnet.zip'%current_dir, './')
        os.system('unzip rpnet.zip')
        
        for each in links:
            os.system('wget %s'%each)
        os.system("unzip '*.zip'")
        os.system(" find . -name '*.mp4' -type f -delete")
        os.system("rm -rf *.zip")

        os.chdir(current_dir)


    def write_2_text(self, annot, name):
        if os.path.exists('%s/%s_set.txt'%(self.path, name)):
            print('file %s/%s_set.txt exists, gonna skip ... '%(self.path, name))
            return 
        f = open('%s/%s_set.txt'%(self.path, name), 'w')
        f.write('Visual Landmark Dataset V1\nImageFile, Camera Position [X Y Z W P Q R] ImageFile, Camera Position [X Y Z W P Q R]\n\n')
        for each in annot:
            f.write(' '.join(each.tolist()) + '\n')
        f.close()

    def save_other_information(self, **data):
        pass
    

    def _pair_image_in_sequence(self, annot, nb_pairs=8):
        pair_index = []
        
        for i in np.arange(annot.shape[0]):
            j = np.array([e for e in np.arange(-20, 21) if e not in  np.arange(-5, 6)]) + i
            np.random.shuffle(j)
            j = j[:nb_pairs]
            for each in j:
                if each >=0 and each < annot.shape[0]:
                    pair_index.append((i, each))
        pair_index = np.array(pair_index)

        assert np.sum(pair_index[:, 0] - pair_index[:, 1] == 0) == 0
        return pair_index
    
    
    def _pair_image(self, annot):
        
        names = np.array([e.split('/')[-2] for e in annot[:, 0]])
        unique_names = np.unique(names)
        
        for e in unique_names: 
            assert 'seq' in e or e in ['img_south', 'img_west', 'img_east', 'img_north', 'img']

        pairs = []
        for e in unique_names:
            tmp_annot = annot[names == e]
            index = np.argsort(tmp_annot[:, 0])
            tmp_annot = tmp_annot[index]
            tmp_pair = self._pair_image_in_sequence(tmp_annot)
            pairs.append(tmp_annot[tmp_pair])
        pairs = np.vstack(pairs).reshape((-1, 16))
        
        return pairs             
    
    
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

        valid_annot = self._pair_image(valid_annot)
        train_annot = self._pair_image(train_annot)

        np.random.shuffle(valid_annot)
        np.random.shuffle(train_annot)
        
        self.write_2_text(valid_annot, 'validation')
        self.write_2_text(train_annot, 'train')

        annot = open('%s/dataset_test.txt'%(self.path)).readlines()[3:]
        annot = [e.split() for e in annot]
        for each in annot:
            each[0] = '%s/%s/'%(self.setting.data_path, self.name) + each[0]
        annot = np.array(annot)
        annot = self._pair_image(annot)
        np.random.shuffle(annot)
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
    

    def get_relative_T_in_cam2_ref(self, R2, t1, t2):
        new_c2 = - np.dot(R2, t2)
        return np.dot(R2, t1) + new_c2    
    
    def _rel_pose_after_rotation(self, each_pose):
        assert each_pose.shape[0] == 14
        
        rotat0 = Quaternion(axis=[0, 0, 1], degrees=0)
        rotat1 = Quaternion(axis=[0, 0, 1], degrees=-90)
        rotat2 = Quaternion(axis=[0, 0, 1], degrees=-180)
        rotat3 = Quaternion(axis=[0, 0, 1], degrees=-270)
        
        rotats = [rotat0, rotat1, rotat2, rotat3]
        
        q1 = Quaternion(each_pose[3:7])
        q2 = Quaternion(each_pose[10:14])
        t1 = each_pose[:3]
        t2 = each_pose[7:10]
        relative_rotation, relative_translation = [], []

        pose1 = []
        pose2 = []
        for i in range(4):
            new_q1 =  rotats[i] * q1
            pose1.append(np.concatenate((t1, new_q1.elements)))
            for j in range(4):
                new_q2 = rotats[j] * q2

                if i == 0:
                    pose2.append(np.concatenate((t2, new_q2.elements)))
                
                relative_rotation.append((new_q2 * new_q1.inverse).elements)
                relative_translation.append(self.get_relative_T_in_cam2_ref(new_q2.rotation_matrix, t1, t2))

        relative_rotation = np.array(relative_rotation)
        relative_translation = np.array(relative_translation)
        
        pose1 = np.array(pose1)
        pose2 = np.array(pose2)
        
        assert pose1.shape[0] == pose2.shape[0] == 4 and pose2.shape[1] == pose1.shape[1] == 7

        rel_pose = np.hstack((relative_translation, relative_rotation)).reshape((16, 7))
        rel_pose = np.vstack((rel_pose, pose1, pose2, each_pose.reshape((2, 7)) ))    
        return rel_pose.reshape((1, 26, 7))
    
    
    def _get_relative_pose(self, pose):
        print(pose.shape)
        assert pose.shape[1] == 14
        pose = np.vstack([self._rel_pose_after_rotation(each) for each in pose])
        print(pose.shape)
        assert pose.shape[1] == 26 and pose.shape[2] == 7
        return pose
    
    
    def process(self, annot):
        global images
        
        images = create_share_ndarray((len(annot), 2, 256, 455, 3), 'uint8', ctypes.c_uint8)
        pool = Pool(processes = multiprocessing.cpu_count() - 10)  # @UndefinedVariable
        pool.map(_read_image, enumerate(annot))
        pool.close()
        pool.join()
        
#         images = np.zeros((len(annot), 2, 256, 455, 3), 'uint8')
#         for index, each in enumerate(annot):
# 
#             im1 = cv2.imread(each[0])  # @UndefinedVariable
#             im1 = cv2.resize(im1, (455, 256))  # @UndefinedVariable
#             
#             im2 = cv2.imread(each[8])  # @UndefinedVariable
#             im2 = cv2.resize(im2, (455, 256))  # @UndefinedVariable
# 
#             images[index] = np.array([im1, im2])
        images = images.astype('f4')
        images = images - images.mean(axis=0).mean(axis=0)
        
        relative_pose = self._get_relative_pose(np.hstack((annot[:, 1:8], annot[:, 9:])).astype('f4'))
        return {'image': images, 
                'pose': relative_pose.astype('f4'), 
                'extra':np.random.randint(0, 4, 2 * images.shape[0]).reshape((-1, 2)).astype('int32')}


if __name__ == '__main__':
    for subset in  ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch', 'Street'][1:]:
        RelativeCambridge(subset)