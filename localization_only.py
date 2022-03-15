"""dead reckoning localization only"""

import os
import pdb
import time
import numpy as np
import argparse
from scipy import linalg
from pr3_utils import *
from matrix_operation import Matrix, Camera


def predict_vehicle(u_t, tau):
    """
    IMU localization via EKF
    :param u_t: IMU measurements 6 * 1, 3 * 1 velocity, 3 * 1 angular velocity
    :param tau: time interval
    :param cov_diag: diagonal covariance matrix
    :return sigma: covariance matrix
    :return imu_T_world: 4x4 transformation matrix {W} -> {IMU}
    """
    u_t_hat = Matrix.pose2hat(u_t)   # 4x4
    u_t_spike = Matrix.pose2spike(u_t)   # 6x6
    world_T_imu = MU[-16:].reshape(4, -1) @ linalg.expm(tau * u_t_hat)
    MU[-16:] = world_T_imu.reshape(-1, 1)


def get_u_cov(vt, wt):
    """
    Calculate the std of linear and angular velocity
    :param vt: 3x3026
    :type:numpy array
    :return:
    """
    v_x, v_y, v_z = vt[0, :], vt[1, :], vt[2, :]
    w_x, w_y, w_z = wt[0, :], wt[1, :], wt[2, :]
    v_x_sigma = np.std(v_x)
    v_y_sigma = np.std(v_y)
    v_z_sigma = np.std(v_z)
    w_x_sigma = np.std(w_x)
    w_y_sigma = np.std(w_y)
    w_z_sigma = np.std(w_z)
    cov_vec = np.square(np.array([v_x_sigma, v_y_sigma, v_z_sigma, w_x_sigma, w_y_sigma, w_z_sigma],
                                 dtype=np.float64))
    cov_diag = np.diag(cov_vec)
    return cov_diag


if __name__ == '__main__':
    start_time = time.perf_counter()
    print('Start running...')

    DATA_SUFFIX = '10'
    skip = 5

    filename = "./data/{}.npz".format(DATA_SUFFIX)
    t, features, linear_velocity, angular_velocity, K, b, imu_T_opt = load_data(filename, load_features=True)
    imu_T_opt = imu_T_opt.astype(np.float64)
    opt_T_imu = np.linalg.inv(imu_T_opt)
    t = t.reshape(-1)

    indices = [idx for idx in range(0, features.shape[1], skip)]
    chosen_feature = features[:, indices, :]
    n_landmarks = chosen_feature.shape[1]
    n_timestamps = chosen_feature.shape[2]
    print("chosen_feature.shape: {}".format(chosen_feature.shape))

    cov_diag = get_u_cov(linear_velocity, angular_velocity)
    fs_u, fs_v, cu, cv, b = Camera.get_para_from_K(K, b)
    Ks = Camera.build_Ks(fs_u, fs_v, cu, cv, b)

    unvisited = np.array([-1, -1, -1, -1], dtype=np.int8)
    mu_I = np.eye(4, dtype=np.float64)  # 4x4
    sigma_I = 1e-2 * np.eye(6, dtype=np.float64)  # 6×6

    # landmarks
    mu_L = np.zeros((3 * n_landmarks, 1), dtype=np.float64)    # 3M
    sigma_L = 0.5 * np.eye(3 * n_landmarks, dtype=np.float64)  # 3M×3M
    obs_mu_t = -1 * np.ones((4, n_landmarks), dtype=np.int8)

    # stack landmark and vehicle mu and sigma
    MU = np.vstack([mu_L, mu_I.reshape(-1, 1)])
    # del mu_I, mu_L
    SIGMA = np.block([[sigma_L, np.zeros((3 * n_landmarks, 6))],
                      [np.zeros((6, 3 * n_landmarks)), sigma_I]])

    pose_traj = np.empty((4, 4, n_timestamps), dtype=np.float64)
    pose_traj[:, :, 0] = mu_I

    for i in range(1, n_timestamps):
        # IMU Localization via EKF Prediction (predict the vehicle)
        tau = t[i] - t[i - 1]
        u_t = np.vstack((linear_velocity[:, i].reshape(3, 1), angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
        predict_vehicle(u_t, tau)   # predict vehicle
        pose_traj[:, :, i] = MU[-16:].reshape(4, -1)
    print("pose_traj.shape: {}".format(pose_traj.shape))
    visualize_trajectory_2d(pose_traj, path_name='dead reckon only localization', show_ori=True, save=True)
    # show_map(pose_traj, landmarks=None)
    end_time = time.perf_counter()
    print('Running time: %.2f s' % (end_time - start_time))

