import numpy as np 
import cv2



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
lidar_data = lidar_data.reshape(-1, 4)

import pudb; pudb.set_trace()


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
"""
import matplotlib.pyplot as plt

def lidar_to_2d_front_view(points,
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
    assert val in {"depth", "height", "reflectance"}, \
        'val must be one of {"depth", "height", "reflectance"}'


    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3] # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min              # Shift
    x_max = 360.0 / h_res       # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
    y_img -= y_min              # Shift
    y_max = v_fov_total / v_res # Theoretical max x value after shifting

    y_max += y_fudge            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.

    # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
    if val == "reflectance":
        pixel_values = r_lidar
    elif val == "height":
        pixel_values = z_lidar
    else:
        pixel_values = -d_lidar

    # PLOT THE IMAGE
    cmap = "jet"            # Color map to use
    dpi = 130               # Image resolution
    fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
    ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
    ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
    ax.axis('scaled')              # {equal, scaled}
    ax.xaxis.set_visible(False)    # Do not draw axis tick marks
    ax.yaxis.set_visible(False)    # Do not draw axis tick marks
    plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

    if saveto is not None:
        fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    else:
        fig.show()


HRES = 1        # horizontal resolution (assuming 20Hz setting)
VRES = 0.1         # vertical res
VFOV = (0, 10.0) # Field of view (-ve, +ve) along vertical axis
Y_FUDGE = 1         # y fudge factor for velodyne HDL 64E

lidar_to_2d_front_view(lidar_data, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",
                       saveto="/tmp/lidar_depth.png", y_fudge=Y_FUDGE)



"""
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