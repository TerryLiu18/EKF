import time
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler



def load_data(file_name, load_features=False):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        if load_features:
            features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam


def visualize_trajectory_2d(pose, path_name="Unknown", show_ori=False, save=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    n_pose = pose.shape[2]
    ax.plot(pose[0, 3, :], pose[1, 3, :],  'r-', label=path_name)
    ax.scatter(pose[0, 3, 0], pose[1, 3, 0], marker='s', label="start")
    ax.scatter(pose[0, 3, -1], pose[1, 3, -1], marker='o', label="end")
  
    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.title(path_name)
    if save:
        plt.savefig(path_name+".png")
    plt.show(block=True)
    return fig, ax


def show_map(traj, landmarks,save=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(traj[0, 3, :], traj[1, 3, :], 'r-', label="pose_trajectory", linewidth=1)
    if landmarks is not None:
        ax.plot(landmarks[0, :], landmarks[1, :], 'bo', markersize=1, label="landmark", linewidth=1)
    ax.set_ylim([-1200, 600])
    if save:
        plt.savefig("map.png")
    plt.show()


if __name__ == '__main__':
    t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data("./data/10.npz")
    visualize_trajectory_2d(imu_T_cam,show_ori=True)



