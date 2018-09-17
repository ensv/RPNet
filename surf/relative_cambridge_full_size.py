'''
Created on Jul 25, 2018

@author: sovann
'''

import matplotlib  # @UnusedImport
import ctypes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2  # @UnresolvedImport
from multiprocessing.pool import Pool
import multiprocessing
from pyquaternion.quaternion import Quaternion
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
cv2.setNumThreads(0)  # @UndefinedVariable

path = '/data/chercheurs/en/'


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


def get_Rerror_in_degree(vec1, vec2):
    d = np.abs(np.sum(np.multiply(vec1, vec2), axis=1))
    np.putmask(d, d> 1, 1)
    np.putmask(d, d< -1, -1)
    return 2 * np.arccos(d) * 180/math.pi

def get_Terror_in_degree(vec1, vec2):
    d = np.sum(np.multiply(vec1, vec2), axis=1)
    np.putmask(d, d> 1, 1)
    np.putmask(d, d< -1, -1)
    return np.arccos(d) * 180/math.pi


def get_cameraMatrix(f, cx = 455, cy=256):
    camera_mat = np.zeros((3, 3), 'f4')
    camera_mat[0, 0] = f
    camera_mat[1, 1] = f
    camera_mat[:2, 2] = (cx/2, cy/2)
    camera_mat[2, 2] = 1
    return camera_mat

def normalize_point(point, mat):
    point = np.hstack((point, np.ones((point.shape[0], 1))))
    return np.dot(np.linalg.inv(mat), point.T).T

def get_relative_pose(idx):
    img1, img2 = annot[idx, 0], annot[idx, 8]

    f1 = txt['/'.join(img1.split('/')[-2:]).replace('.png', '.jpg')][0]
    f2 = txt['/'.join(img2.split('/')[-2:]).replace('.png', '.jpg')][0]
    
    im1 = cv2.imread(img1, 0)  # @UndefinedVariable
    im2 = cv2.imread(img2, 0)  # @UndefinedVariable

    assert im1.shape[0] < im1.shape[1]

    sift = cv2.xfeatures2d.SIFT_create()  # @UndefinedVariable
    
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # @UndefinedVariable

    # Match descriptors.
    try:
        matches = bf.match(des1, des2)
    except cv2.error:  # @UndefinedVariable
        return np.array([0.333, 0.333, 0.3334, 0.25, 0.25, 0.25, 0.25], 'f4')


    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)[:100]

    # Draw first 10 matches.
#     draw_params = dict(#matchColor = (0,255,0), # draw matches in green color
#                            singlePointColor = None,
#                             flags = 2)
#     img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:20], None, **draw_params)
#     plt.figure(figsize=(20,20))
#     plt.imshow(img3)
#     plt.show()

    mat1 = get_cameraMatrix(f1, cx = 1920, cy = 1080)
    mat2 = get_cameraMatrix(f2, cx = 1920, cy = 1080)        
    
    src = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)
    
    src = normalize_point(src, mat1)
    dst = normalize_point(dst, mat2)
    
    src = src[:, :2]
    dst = dst[:, :2]

    tmp_results = []
    
    for _threshold in [1e-2, 1e-3, 1e-4]:
        
        try:
            E, mask = cv2.findEssentialMat(np.copy(src), np.copy(dst), cameraMatrix = np.identity(3, 'f4'), threshold = _threshold)  # @UndefinedVariable
            _, R, t, mask, _ = cv2.recoverPose(E, np.copy(src), np.copy(dst), mask = mask, cameraMatrix = np.identity(3, 'f4'), distanceThresh = 50)  # @UndefinedVariable
        except cv2.error:  # @UndefinedVariable
            tmp_results.append(np.array([0.333, 0.333, 0.3334, 0.25, 0.25, 0.25, 0.25], 'f4'))
            continue
        tmp_results.append(np.concatenate((t.flatten(), np.array(Quaternion(matrix=R).elements).astype('f4').flatten())))

    return np.array(tmp_results)

def _save_plot(arr, name):
    
    cumulative = np.array([np.where(arr <= e)[0].shape[0] for e in range(180)]) / arr.shape[0]
        
    plt.figure(figsize=(8, 6))
    plt.plot(range(180), cumulative)
    plt.grid(True)
    plt.title(name.split('_')[0])
    plt.savefig(name, dpi = 300)
    plt.close()
#     plt.show(True)
 
def save_plot(R_error, T_error, name):
    _save_plot(R_error, '%s/R.png'%name)
    _save_plot(T_error, '%s/T.png'%name)


def get_Terror_in_meter(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2, axis=1 ))


def get_relative_pose_np(r1, r2, t1, t2):
    
    r1, r2, t1, t2 = r1.astype('f4'), r2.astype('f4'), t1.astype('f4'), t2.astype('f4')
    pose = []
    for i in range(t1.shape[0]):
        
        rr = (Quaternion(r2[i]) * Quaternion(r1[i]).inverse).elements
        R2 = Quaternion(r2[i]).rotation_matrix
        new_c2 = - np.dot(R2, t2[i])
        rt = np.dot(R2, t1[i]) + new_c2   
        
        pose.append(np.concatenate((rt / np.linalg.norm(rt), rr)))
    return np.array(pose)

if __name__ == '__main__':
    

    for dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
        print('\n====================================\n')
        print('dataset: ', dataset)

        train_test_set = 'test'

        base = '%s/DRP/relative_cambridge/%s'%(path, dataset)

        annot = open('%s/DRP/relative_cambridge/%s/test_set.txt'%(path, dataset)).readlines()[3:]
        annot = np.array([e.split() for e in annot])
        gt_pose = get_relative_pose_np(annot[:, 4:8], annot[:, 12:16], annot[:, 1:4], annot[:, 9:12])

        txt = [e for e in open('%s/DRP/relative_cambridge/%s/reconstruction.nvm'%(path, dataset)).readlines()[3:] if 'seq' in e]
        txt = {e.split('\t')[0]:np.array(e.split('\t')[1].split(), 'f4') for e in txt if len(e.split('\t')) >= 2}
        
        try:
            os.makedirs('%s/DRP/baseline/%s/SURF/full/'%(path, dataset))
        except:
            pass

            
        pool = Pool(processes=32)  # @UndefinedVariable
        est_pose = pool.map(get_relative_pose, range(annot.shape[0]), chunksize=1)
        pool.close()
        pool.join()
    
        est_pose = np.vstack(est_pose)
        est_pose = est_pose.reshape((gt_pose.shape[0], 3, 7))
        
        for i, _threshold in enumerate([1e-2, 1e-3, 1e-4]):
            
            filename = '%s/DRP/baseline/%s/SURF/test_full_%f/'%(path, dataset, _threshold)

            try:
                os.makedirs(filename)
            except:
                pass


            xyz, wpqr = est_pose[:, i, :3], est_pose[:, i, 3:]                                  
    
            T = gt_pose[:, :3]
            R = gt_pose[:, 3:]
    
            R_error = get_Rerror_in_degree(wpqr, R)
    
            T_error = get_Terror_in_degree(xyz, T)
            T_error_meter = get_Terror_in_meter(xyz, T)
    
            print('Threshold:', _threshold)                
            print('Rotation: ', np.median(R_error))
            print('Translation: ', np.median(T_error))
            print('Translation (meter):', np.median(T_error_meter))
            print()
            
#             np.save('%sResults'%filename, np.array([np.median(R_error), np.median(T_error), np.median(T_error_meter)]))
#             np.save('%sRerror'%filename, R_error)
#             np.save('%sTerror'%filename, T_error)
#             save_plot(R_error, T_error, filename)
