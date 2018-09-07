'''
Created on Jul 25, 2018

@author: sovann
'''
import numpy as np
import os
from pyquaternion.quaternion import Quaternion
import math


def quat2matrix(q):
    w, x, y, z = q
    sqw = w*w;
    sqx = x*x;
    sqy = y*y;
    sqz = z*z;

    # invs (inverse square length) is only required if quaternion is not already normalised
    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw)*invs ; # since sqw + sqx + sqy + sqz =1/invs*invs
    m11 = (-sqx + sqy - sqz + sqw)*invs ;
    m22 = (-sqx - sqy + sqz + sqw)*invs ;

    tmp1 = x*y;
    tmp2 = z*w;
    m10 = 2.0 * (tmp1 + tmp2)*invs ;
    m01 = 2.0 * (tmp1 - tmp2)*invs ;

    tmp1 = x*z;
    tmp2 = y*w;
    m20 = 2.0 * (tmp1 - tmp2)*invs ;
    m02 = 2.0 * (tmp1 + tmp2)*invs ;
    tmp1 = y*z;
    tmp2 = x*w;
    m21 = 2.0 * (tmp1 + tmp2)*invs ;
    m12 = 2.0 * (tmp1 - tmp2)*invs ;      

    return np.array([
        [m00, m01, m02],
        [m10, m11, m12],
        [m20, m21, m22]
    ], 'f4')


def get_relative_pose_np(r1, r2, t1, t2):
    
    r1, r2, t1, t2 = r1.astype('f4'), r2.astype('f4'), t1.astype('f4'), t2.astype('f4')
    pose = []
    for i in range(t1.shape[0]):
        
        rr = (Quaternion(r2[i]) * Quaternion(r1[i]).inverse).elements
        R2 = Quaternion(r2[i]).rotation_matrix
        new_c2 = - np.dot(R2, t2[i])
        rt = np.dot(R2, t1[i]) + new_c2   
        
        pose.append(np.concatenate((rt, rr)))
    return np.array(pose)


if __name__ == '__main__':

    prefix = '/data/chercheurs/en'
    if not os.path.exists(prefix):
        prefix = '/home/2017025/sen01'

    for dataset in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch', 'Street']:
        base = '%s/DRP/absolute_cambridge_network/%s/PoseNetD/cambridge/'%(prefix, dataset)

        for each in os.listdir(base):
            print('%s/%s/0/all_pose.npy'%(base, each))
            if not os.path.exists('%s/%s/0/all_pose.npy'%(base, each)): continue
            
            allpose = np.load('%s/%s/0/all_pose.npy'%(base, each))
            
            if np.isnan(allpose.sum()):
                np.save('%s/%s/0/error'%(base, each), np.array(['nan', 'nan', 'nan']))
            
            abspose = open('%s/DRP/absolute_cambridge/%s/test_set.txt'%(prefix, dataset)).readlines()[3:]
            abspose = np.array([e.split() for e in abspose])
            
            
            assert abspose.shape[0] == allpose.shape[0]
            
            abspose = {'/'.join(abspose[i][0].split('/')[-2:]):allpose[i, :7] for i in range(abspose.shape[0]) \
                                    if (abspose[i, 1:].astype('f4') - allpose[i, 7:]).sum() == 0}
            
            annot = open('%s/DRP/relative_cambridge/%s/test_set.txt'%(prefix, dataset)).readlines()[3:]
            annot = np.array([e.split() for e in annot])
            gt_pose = get_relative_pose_np(annot[:, 4:8], annot[:, 12:16], annot[:, 1:4], annot[:, 9:12])

            est_pose = []
            for each_annot in annot:
                name1, name2 = '/'.join(each_annot[0].split('/')[-2:]), '/'.join(each_annot[8].split('/')[-2:])

                pose1 = abspose[name1.replace('relative_cambridge', 'absolute_cambridge')]
                pose2 = abspose[name2.replace('relative_cambridge', 'absolute_cambridge')]
                
                q1, q2 = pose1[3:], pose2[3:]
                t1, t2 = pose1[:3], pose2[:3]
                
                rr = (Quaternion(q2) * Quaternion(q1).inverse).normalised

                R2 = Quaternion(q2).rotation_matrix
                new_c2 = - np.dot(R2, t2)
                rt = np.dot(R2, t1) + new_c2
                est_pose.append(np.concatenate((rt, rr.elements)))
            
            est_pose = np.array(est_pose)

            tm = np.linalg.norm(est_pose[:, :3] - gt_pose[:, :3], axis=1)
            d = np.abs(np.sum(est_pose[:, 3:] * gt_pose[:, 3:], axis=1))
            np.putmask(d , d > 1, 1)
            np.putmask(d , d < -1, -1)
            rd = 2 * np.arccos(d) * 180/math.pi
            
            est_pose[:, :3] = est_pose[:, :3] / np.linalg.norm(est_pose[:, :3], axis=1, keepdims=True)
            gt_pose[:, :3] = gt_pose[:, :3] / np.linalg.norm(gt_pose[:, :3], axis=1, keepdims=True)
            
            d = np.sum(est_pose[:, :3] * gt_pose[:, :3], axis=1)
            np.putmask(d , d > 1, 1)
            np.putmask(d , d < -1, -1)
            td = np.arccos(d) * 180/math.pi
            
            error = np.array([tm, rd, td]).reshape((3, -1))
            
            print(np.median(error, axis=1))
            np.save('%s/%s/0/error'%(base, each), error)
            
            