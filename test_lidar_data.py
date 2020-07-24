import numpy as np 
import cv2
from math import pi


from scipy.spatial.transform import Rotation as R

def gen(ld_data, depth_data, point, cl_point):
    blank_image = np.zeros((300, 300, 3), np.uint32)
    for idx, point in enumerate(ld_data):
        blank_image[point[0]][point[1]][1] =  np.clip(depth_data[idx] + blank_image[point[0]][point[1]][1], 0, cl_point)
    return blank_image

project_matrix_x = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
project_matrix_y = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
project_matrix_z = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def f2d_z(x):
    return project_matrix_z @ x


def f2d_y(x):
    return project_matrix_y @ x

def f2d_x(x):
    return project_matrix_x @ x



lidar_data = np.load('lidar_array_100.npy')
#lidar_data = lidar_data.reshape(-1, 4)

#

roll = 2.6335
pitch = 0.4506
yaw = 1.1684

hud_dim = [1000, 1000]

import pudb; pudb.set_trace()

points = np.frombuffer(lidar_data, dtype=np.dtype('f4'))
points = np.reshape(points, (int(points.shape[0] / 3), 3))
#points[:, [2]] *= -1

lidar_data = np.array(points[:, :2])
lidar_data *= min(hud_dim) / 100.0
lidar_data += (0.5 * hud_dim[0], 0.5 * hud_dim[1])
lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
lidar_data = lidar_data.astype(np.int32)
#lidar_data[2] *= -1
lidar_data = np.reshape(lidar_data, (-1, 2))
lidar_img_size = (hud_dim[0], hud_dim[1], 3)
lidar_img = np.zeros((lidar_img_size), dtype = int)

#for idx, point in enumerate(lidar_data):
    #import pudb; pudb.set_trace()
#    lidar_img[point[0]][point[1]][1] += points[idx][3]*10
    #lidar_img[point.T] = (0, points[idx][3] + lidar_img[point.T][1], 0)


lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

#import pudb; pudb.set_trace()

cv2.imshow('Lidar', lidar_img.astype(np.uint8))
cv2.waitKey(0)




"""
rot = [0, 3*pi/2, 0]



#c = [130, 125, 250]

#c = [130, 200, 250]

c = [0.5, 0.5 , 0.5]


s = [300, 300]

r = [30, 30, 1]

#e = [3, 3, 3]
blank_image = np.zeros((300, 300, 3), np.uint32)
cl_point = 10000000

eps = 1e-6

#import pudb; pudb.set_trace()

for idx, point in enumerate(lidar_data):
    
    #point_no_depth = lidar_data[:, [0, 1, 2]]
    #point_depth = lidar_data[:, 3]
    #if(idx == 4826):
    #    import pudb; pudb.set_trace()

    a = point[:3] / 255
    point_depth = point[3]

    m_1 = np.asarray([[1, 0, 0], [0, np.cos(rot[0]), np.sin(rot[0])], [0, -np.sin(rot[0]), np.cos(rot[0])]])
    m_2 = np.asarray([[np.cos(rot[1]), 0, -np.sin(rot[1])], [0, 1, 0], [np.sin(rot[1]), 0, np.cos(rot[1])]])
    m_3 = np.asarray([[np.cos(rot[2]), np.sin(rot[2]), 0], [-np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])
    vec = a - c

    d_vec = m_1 @ m_2 @ m_3 @ vec

    bx = int((d_vec[0] * s[0]) / (d_vec[2]*r[0] + eps)    *   r[2])
    by = int((d_vec[1] * s[1]) / (d_vec[2]*r[1] + eps)    *   r[2])

    if(bx > 0 and bx < 300 and by > 0 and by < 300):
        #blank_image[bx, by][1] =  np.clip(point_depth + blank_image[bx, by][1], 0, cl_point)
        #blank_image[bx, by][1] =  max(point_depth,  blank_image[bx, by][1])
        blank_image[bx, by][1] += point_depth



cv2.imwrite('./depth_test_.png', blank_image / (blank_image.max()/255.0))

cv2.imshow('Proj_x_test', blank_image / (blank_image.max()/255.0))
cv2.waitKey(0)
cv2.waitKey(1)

"""



"""

# Resolution and Field of View of LIDAR sensor
h_res = 0.35         # horizontal resolution, assuming rate of 20Hz is used 
v_res = 0.4          # vertical res
v_fov = (-24.9, 2.0) # Field of view (-ve, +ve) along vertical axis
v_fov_total = -v_fov[0] + v_fov[1] 

# Convert to Radians
v_res_rad = v_res * (np.pi/180)
h_res_rad = h_res * (np.pi/180)

# Project into image coordinates
x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad


# SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
x_min = -360.0/h_res/2    # Theoretical min x value based on specs of sensor
x_img = x_img - x_min     # Shift
x_max = 360.0/h_res       # Theoretical max x value after shifting

y_min = v_fov[0]/v_res    # theoretical min y value based on specs of sensor
y_img = y_img - y_min     # Shift
y_max = v_fov_total/v_res # Theoretical max x value after shifting
y_max = y_max + 5         # UGLY: Fudge factor because the calculations based on
                          # spec sheet do not seem to match the range of angles
                          # collected by sensor in the data.



lidar_data_nodepth = lidar_data[:, [0, 1, 2]]
lidar_just_depth = lidar_data[:, 3]

lidar_in_2d_z = np.asarray(list(map(f2d_z, lidar_data_nodepth)))
lidar_in_2d_z = lidar_in_2d_z[:, [0, 1]]

lidar_in_2d_x = np.asarray(list(map(f2d_x, lidar_data_nodepth)))
lidar_in_2d_x = lidar_in_2d_x[:, [1, 2]]

lidar_in_2d_y = np.asarray(list(map(f2d_y, lidar_data_nodepth)))
lidar_in_2d_y = lidar_in_2d_y[:, [0, 2]]

#2558556
#2722
img_x_test = gen(lidar_in_2d_x, lidar_just_depth, 1000, 2558556/3)
cv2.imshow('Proj_x_test', img_x_test / (img_x_test.max()/255.0))

img_y_test = gen(lidar_in_2d_y, lidar_just_depth, 1000, 2558556/3)
cv2.imshow('Proj_y_test', img_y_test / (img_y_test.max()/255.0))

img_z_test = gen(lidar_in_2d_z, lidar_just_depth, 1000, 2558556/3)
cv2.imshow('Proj_z_test', img_z_test / (img_z_test.max()/255.0))

cv2.waitKey(0)
cv2.waitKey(1)
"""