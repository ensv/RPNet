'''
Created on Jun 19, 2018

@author: en
'''

import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
import numpy as np
from mayavi import mlab
import cv2
from matplotlib.patches import ConnectionPatch
from camera import Camera  # @UnresolvedImport


def get_3d_function():
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    return np.concatenate((x, y, z)).reshape((3, -1)).T, x.shape

def plot_3d(xyz, r_xyz, shape_, cam1, cam2):
    s = mlab.mesh(xyz[:, 0].reshape(shape_), xyz[:, 1].reshape(shape_), xyz[:, 2].reshape(shape_))
    mlab.axes( x_axis_visibility= True, y_axis_visibility=True, z_axis_visibility=True)
    
    # plot camera position
    pos = np.concatenate((cam1.pos, cam2.pos)).reshape((2, 3))
    mlab.plot3d(pos[:, 0], pos[:, 1], pos[:, 2], color=(0, 0, 1), line_width=0.5)

    T = cam2.get_relative_T_in_world_reference(cam1)
    mlab.quiver3d(cam1.pos[0], cam1.pos[1], cam1.pos[2], T[0], T[1], T[2])#, extent=np.concatenate((T, 2*T)))

    # plot random point
    mlab.points3d(r_xyz[:, 0], r_xyz[:, 1], r_xyz[:, 2], scale_factor = 0.1, color = (1, 1, 1))
    for i in range(r_xyz[:, 0].shape[0]):
        mlab.text3d(r_xyz[i, 0], r_xyz[i, 1], r_xyz[i, 2], 'P%d'%i, scale = 0.1)


    # plot the position of the camera and its directional vector
    cam1.plot_pos_orientation(mlab)
    cam2.plot_pos_orientation(mlab)
    
    # plot projected point on each camera filter
    cam1.plot_projected_point(mlab)
    cam2.plot_projected_point(mlab)
    
    mlab.show(stop=True)

def plot_projected_point(im1, im2, title, block_):
    # Plot images 
    fig = plt.figure()
    plt.suptitle(title)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i in range(im1.shape[0]):
        try:
            con = ConnectionPatch(xyA=im2[i, :2], xyB=im1[i, :2], coordsA="data", coordsB="data",
                                      axesA=ax2, axesB=ax1, color="red")
            ax2.add_artist(con)
        except:
            pass
    ax1.grid()
    ax1.plot(im1[:, 0], im1[:, 1],'ko')
    ax2.plot(im2[:, 0], im2[:, 1],'ko')
    ax2.grid()
    plt.show(block=block_)

if __name__ == '__main__':

    show_gui = True
    nb_points = 100
    
    error= []
    
    for _ in range(100):
    
        xyz, xyz_shape = get_3d_function()
        
        cam1 = Camera(center_3d=np.mean(xyz, axis=0), name = 'cam1')
        cam2 = Camera(center_3d=np.mean(xyz, axis=0), pos = cam1.pos + np.random.randint(20, 30, 3) /30, name = 'cam2')
        
        idx = np.random.randint(0, xyz.shape[0], nb_points)
    
        random_xyz = xyz[idx]
        
        cam1.project_3dpoint_onto_2d(random_xyz)
        cam2.project_3dpoint_onto_2d(random_xyz)
        
        if show_gui:
            plot_projected_point(cam1.image_in_camera_reference, cam2.image_in_camera_reference, 'Projected points', True)
            plot_3d(xyz, random_xyz, xyz_shape, cam1, cam2)
    
        src, dst = cam1.image_in_camera_reference[:, :3], cam2.image_in_camera_reference[:, :3]
    
        # mask to eliminate point that has the depth < 1 in the camera frame reference
        mask1, mask2 = cam1.mask, cam2.mask 
        pair_mask = mask1 * mask2
        
        src = src[pair_mask]
        dst = dst[pair_mask]
        
        src = src[:, :2] 
        dst = dst[:, :2]
        camera_mat = np.identity(3, 'f4') 
            
        E, _mask = cv2.findEssentialMat(src, dst, cameraMatrix = np.identity(3, 'f4'), threshold= 0.001)  # @UndefinedVariable
        _, est_R, est_t, _mask, point3d = cv2.recoverPose(E, src, dst, cameraMatrix = np.identity(3, 'f4'), distanceThresh = 50, mask=_mask)  # @UndefinedVariable    
    
        print('---------------')
        error.append(cam1.print_relative_error(est_R, est_t, cam2))
        if show_gui:
            plot_four_solution(E, )
            
    print('Mean:', np.mean(error, axis=0))
    print('Median:', np.median(error, axis=0))