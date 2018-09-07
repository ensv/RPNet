'''
Created on Jun 19, 2018

@author: en
'''

import numpy as np
import math
from math import sin, cos
from pyquaternion.quaternion import Quaternion
import sys
from numpy import arctan
class Camera:
    '''
    classdocs
    '''


    def __init__(self, center_3d, pos = None, name = 'cam1'):
        '''
        Constructor
        '''
        
        self.focal = 1
        self.pp = (0,0)

        self.camera_matrix = np.zeros((3, 3), 'f4')
        self.camera_matrix[0, 0] = self.camera_matrix[1, 1] = self.focal
        self.camera_matrix[:2, 2] = self.pp
        self.camera_matrix[2, 2] = 1

        self.name = name
        if pos is None:
            self.pos = (np.random.randint(0, 100, 3)  - 10) / 10
#             self.pos = (np.random.randint(0, 100, 3)  - 50) / 10
        else:
            self.pos = pos

        self.center_vector = (center_3d - self.pos) / np.linalg.norm((center_3d - self.pos))
        self.euler, self.orientation_vector = self.get_orientation_vector(self.center_vector)
        
        self.Proj = self.get_projection_matrix(self.euler, self.pos)

    def get_orientation_vector(self, ref_vec):
        while True:
            euler = np.random.randint(0, 180, 3000) * math.pi/180
            euler = euler.reshape((1000, 3))
            vec = np.array([self.euler_2_vector(each) for each in euler])
             
            angle = np.arccos(np.sum(vec * ref_vec.reshape((1, 3)), axis=1)) * 180/math.pi
             
            idx = np.where(angle <25)[0]
            if idx.shape[0] > 0:
                return euler[idx[0]], vec[idx[0]]

    def euler_2_vector(self, param):
        yaw, pitch, roll = param
        return np.array([
                         sin(yaw)*sin(roll) + cos(yaw) * sin(pitch) * cos(roll),
                         -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll),
                         cos(pitch) * cos(roll)
                         ])
        
        
    # code from wikipedia (default convention)
    def angle2quat(self, arr):
        yaw, pitch, roll = arr
        cy = math.cos(yaw * 0.5);
        sy = math.sin(yaw * 0.5);
        cr = math.cos(roll * 0.5);
        sr = math.sin(roll * 0.5);
        cp = math.cos(pitch * 0.5);
        sp = math.sin(pitch * 0.5);
        w = cy * cr * cp + sy * sr * sp;
        x = cy * sr * cp - sy * cr * sp;
        y = cy * cr * sp + sy * sr * cp;
        z = sy * cr * cp - cy * sr * sp;
        return np.array([w, x, y, z])
    
    def angle2mat(self, arr):
        t1, t2, t3 = arr
        return np.array([
                math.cos(t1)*math.cos(t2), 
                math.cos(t1)*math.sin(t2)*math.sin(t3)-math.sin(t1)*math.cos(t3),
                math.cos(t1)*math.sin(t2)*math.cos(t3)+math.sin(t1)*math.sin(t3),
                math.sin(t1)*math.cos(t2),
                math.sin(t1)*math.sin(t2)*math.sin(t3)+math.cos(t1)*math.cos(t3),
                -math.sin(t2),
                math.cos(t2)*math.sin(t3),
                math.cos(t2)*math.cos(t3),
            ]).reshape((3, 3)).astype('f4')
    
    def get_projection_matrix(self, R, T):

        q = self.angle2quat(R)
        mat = self.quat2matrix(q)
    
        R = np.zeros((3, 4), 'f4')
        R[:3, :3] = mat.T
        R[:, 3] = - np.dot(mat.T, T)
        
        return R
    

    # code: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    def quat2matrix(self, q):
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
            
    def plot_projected_point(self, mlab):
        
        tmp = self.point3d[self.mask]
        tmp1 = self.image_in_world_reference[self.mask]

        for t_indx in range(min(40, tmp.shape[0])):
            t_2_im1 = np.vstack((tmp[t_indx], tmp1[t_indx]))              
            mlab.plot3d(t_2_im1[:, 0], t_2_im1[:, 1], t_2_im1[:, 2], color=(1, 0, 1), line_width=0.01)
    
    def plot_pos_orientation(self, mlab):
        mlab.text3d(self.pos[0], self.pos[1], self.pos[2], self.name, scale = 0.1)
        mlab.quiver3d(self.pos[0], self.pos[1], self.pos[2], self.orientation_vector[0], self.orientation_vector[1], self.orientation_vector[2])#, extent=np.concatenate((T, 2*T)))

    def project_3dpoint_onto_2d(self, point3d):
        self.point3d = point3d
        
        self.image_in_camera_reference_with_depth = (np.dot(self.Proj[:3, :3], point3d.T) + self.Proj[:3, 3:4]).T
        self.mask = np.ones((self.point3d.shape[0], ), 'bool')
        
#         self.mask = self.image_in_camera_reference_with_depth[:, 2] > self.focal
#         
#         if self.mask.sum() < 10:
#             print('the camera is too close to the point ...')
#             sys.exit(1)
        
        self.image_in_camera_reference = self.image_in_camera_reference_with_depth / self.image_in_camera_reference_with_depth[:, 2:]

        self.image_in_camera_reference = np.dot(self.camera_matrix, self.image_in_camera_reference.T).T

        # kind of cheating to plot the correct value in world coordinate
        tmp = self.image_in_camera_reference + 0
        tmp[:, 2] = self.focal

        self.image_in_world_reference = np.dot(self.Proj[:3, :3].T, (tmp - self.Proj[:3, 3]).T).T

#         idx = np.argmax(self.image_in_camera_reference[:, 0])
#         print(self.point3d[idx], '\n', self.Proj, '\n', self.image_in_camera_reference[idx], '\n', self.image_in_camera_reference_with_depth[idx])
        
#     def get_RT_in_camera_reference(self, R, T):
#         T_cam = np.dot(self.Proj[:3, :3].T, (T - self.Proj[:3, 3]).reshape((3, 1)))
#         R_cam = R#np.dot(Proj[:3, :3].T, R.flatten() - Proj[:3, 3])
#         
#         T_cam = T_cam.flatten()
#         T_cam = T_cam/ np.linalg.norm(T_cam)
#         return R_cam, T_cam
    
    def relative_rotation(self, camera2):
        # inverse of quaternion
        q2 = camera2.angle2quat(camera2.euler)
        q1 = self.angle2quat(self.euler)
        q2 = q2 * np.array([1., -1., -1., -1.])
        tmp = [q2[0]*q1[0] - q2[1]*q1[1] - q2[2]*q1[2] -  q2[3]*q1[3], \
                q2[0]*q1[1] + q2[1]*q1[0] + q2[2]*q1[3] - q2[3]*q1[2], \
                q2[0]*q1[2] - q2[1]*q1[3] + q2[2]*q1[0] + q2[3]*q1[1], \
                q2[0]*q1[3] + q2[1]*q1[2] - q2[2]*q1[1] + q2[3]*q1[0] ]

        return np.array(tmp)

    def get_relative_T_in_world_reference(self, camera1):
        return self.pos - camera1.pos

    def relative_T(self, camera2):
        T = np.dot(camera2.Proj[:3, :3], self.pos) + camera2.Proj[:3, 3]
        return T / np.linalg.norm(T)
        
    def print_relative_error(self, est_R, est_T, camera2):

        est_R = np.array(Quaternion(matrix=est_R).elements).flatten()
        gt_R = self.relative_rotation(camera2)
        diff_R = min(np.abs(np.dot(est_R, gt_R)), 1)
        print('Relative R:', 2 * np.arccos(diff_R) * 180 /math.pi )
        
        gt_T = self.relative_T(camera2)
#         print(gt_T, est_T.flatten())
        diff_T = np.dot(est_T.flatten(), gt_T)
        diff_T = min(1, diff_T)
        diff_T = max(-1, diff_T)
        print('Relative T:', np.arccos(diff_T) * 180 /math.pi )

        return 2 * np.arccos(diff_R) * 180 /math.pi , np.arccos(diff_T) * 180 /math.pi