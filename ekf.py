import os
import pdb
import time
import numpy as np
import argparse
from scipy import linalg
from tqdm import tqdm
from pr3_utils import *
from matrix_operation import Matrix, Camera, Derivative


def get_kalman_gain(sigma, H, Nt, v=100):
    """
    Calculate Kalman Gain
    :param sigma:
    :param H: Jacobian
    :param Nt: number of update observations
    :param lsq:use lsq (faster) least-squares solution or solve (slower) exact solution
    :param v: noise constant
    """
    I = np.eye(4 * Nt, dtype=np.float64)
    V = np.kron(I, v)
    H_sigma = H @ sigma
    temp = H_sigma @ H.T + V
    K_T, _, _, _ = np.linalg.lstsq(temp, H_sigma, rcond=None)
    return K_T.T, H_sigma


def predict_vehicle(u_t, tau, cov_diag):
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
    imu_T_world = np.linalg.inv(world_T_imu)

    # add noise
    W = np.random.multivariate_normal(mean=[0, 0, 0, 0, 0, 0], cov=cov_diag).reshape(-1, 1)
    noise_spike = Matrix.pose2spike(W)
    noise_p = linalg.expm(tau ** 2 * noise_spike)
    perturbation = linalg.expm(-tau * u_t_spike)
    SIGMA[-6:, -6:] = np.linalg.multi_dot(
        [noise_p, perturbation, SIGMA[-6:, -6:], perturbation.T, noise_p.T]
    )
    return imu_T_world


def joint_update():
    mu_L = MU[0:-16].reshape(3, -1)
    mu_t_j = np.concatenate((mu_L[:, update_feature_index], np.ones((1, len(update_feature_index)))))
    z = features_t[:, update_feature_index]
    z_tilt = Ks @ Camera.projection(opt_T_world @ mu_t_j)
    error = (z - z_tilt).reshape(-1, 1)
    # Update landmarks_mu, landmarks_sigma and T_imu_mu and T_imu_sigma simultaneously
    H_joint = Derivative.projection_Jacobian(Ks, opt_T_imu, imu_T_world, n_landmarks,
                                             np.array(update_feature_index), mu_t_j)
    # update vehicle
    K_joint, H_sigma_joint = get_kalman_gain(SIGMA, H_joint, len(update_feature_index), v=100)
    T_spike = Matrix.pose2hat(K_joint[-6:, :] @ error)
    MU[-16:] = (world_T_imu @ linalg.expm(T_spike)).reshape(-1, 1)
    SIGMA[-6:, -6:] = SIGMA[-6:, -6:] - K_joint[-6:, :] @ H_sigma_joint[:, -6:]
    # update landmarks
    MU[:-16] = mu_L.reshape(-1, 1) + K_joint[:-6, :] @ error
    SIGMA[:-6, :-6] = SIGMA[:-6, :-6] - K_joint[:-6, :] @ H_sigma_joint[:, :-6]
    # delete local variable to save memory


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

    task_flag = 0
    if os.path.exists('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip))):
        print('traj_{}_{}.npy already exists.'.format(DATA_SUFFIX, str(skip)))
        pose_traj = np.load('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip)))
        visualize_trajectory_2d(pose_traj, path_name='dead reckon only localization', show_ori=True, save=True)

    if os.path.exists('map_{}_{}.npy'.format(DATA_SUFFIX, str(skip))) and\
        os.path.exists('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip))):
        print('map_{}_{}.npy and traj_{}_{}.npy already exists.'.format(DATA_SUFFIX, str(skip), DATA_SUFFIX, str(skip)))
        landmarks_mu = np.load('map_{}_{}.npy'.format(DATA_SUFFIX, str(skip)))
        pose_traj = np.load('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip)))
        show_map(pose_traj, landmarks_mu)
        task_flag = 1
    if task_flag:
        print('Task completed.')
        exit()


    t, features, linear_velocity, angular_velocity, K, b, imu_T_opt = load_data(filename, load_features=True)
    imu_T_opt = imu_T_opt.astype(np.float64)
    opt_T_imu = np.linalg.inv(imu_T_opt)
    t = t.reshape(-1)

    indices = [idx for idx in range(0, features.shape[1], skip)]
    chosen_feature = features[:, indices, :]
    n_landmarks = chosen_feature.shape[1]
    n_timestamps = chosen_feature.shape[2]
    del features
    print("chosen_feature.shape: {}".format(chosen_feature.shape))

    cov_diag = get_u_cov(linear_velocity, angular_velocity)
    fs_u, fs_v, cu, cv, b = Camera.get_para_from_K(K, b)
    Ks = Camera.build_Ks(fs_u, fs_v, cu, cv, b)

    unvisited = np.array([-1, -1, -1, -1], dtype=np.int8)
    mu_I = np.eye(4, dtype=np.float64)  # 4x4
    sigma_I = 1e-1 * np.eye(6, dtype=np.float64)  # 6×6

    # landmarks
    mu_L = np.zeros((3 * n_landmarks, 1), dtype=np.float64)    # 3M
    sigma_L = 0.01 * np.eye(3 * n_landmarks, dtype=np.float64)  # 3M×3M
    obs_mu_t = -1 * np.ones((4, n_landmarks), dtype=np.int8)

    # stack landmark and vehicle mu and sigma
    MU = np.vstack([mu_L, mu_I.reshape(-1, 1)])
    del mu_L
    SIGMA = np.block([[sigma_L, np.zeros((3 * n_landmarks, 6))],
                      [np.zeros((6, 3 * n_landmarks)), sigma_I]])
    del sigma_I, sigma_L

    pose_traj = np.empty((4, 4, n_timestamps), dtype=np.float64)
    pose_traj[:, :, 0] = mu_I

    for i in tqdm(range(1, n_timestamps)):
        # IMU Localization via EKF Prediction (predict the vehicle)
        tau = t[i] - t[i - 1]
        u_t = np.vstack((linear_velocity[:, i].reshape(3, 1), angular_velocity[:, i].reshape(3, 1)))  # u(t) \in R^{6}
        imu_T_world = predict_vehicle(u_t, tau, cov_diag)   # predict vehicle MU and SIGMA
        world_T_imu = np.linalg.inv(imu_T_world)

        # Landmark Mapping via EKF Update
        opt_T_world = opt_T_imu @ imu_T_world
        world_T_opt = world_T_imu @ imu_T_opt
        features_t = chosen_feature[:, :, i]
        index_t = tuple(np.where(np.sum(features_t, axis=0) > -4)[0])
        update_feature_index = []

        # if new landmarks are available
        if index_t:
            visited_features_pixels = features_t[:, index_t]
            m_world_ = Camera.pixel2world(visited_features_pixels, K, b, world_T_opt)  # get the world coord of feature

            # update vehicle pose
            for j in range(len(index_t)):
                cur_index = index_t[j]
                if np.allclose(obs_mu_t[:, cur_index], unvisited):
                    obs_mu_t[:, cur_index] = np.full_like(visited_features_pixels[:, j], -2, dtype=np.int8)
                    mu_L = MU[:-16].reshape(3, -1)
                    mu_L[:, cur_index] = np.delete(m_world_[:, j], 3, axis=0)
                    MU[: -16] = mu_L.reshape(-1, 1)
                else:
                    update_feature_index.append(cur_index)

            # EKF joint update
            if update_feature_index:
                joint_update()
        pose_traj[:, :, i] = MU[-16:].reshape(4, -1)

    landmark_positions = MU[:-16].reshape(3, -1)
    visualize_trajectory_2d(pose_traj, path_name='dead reckon only localization', show_ori=True, save=True)
    show_map(pose_traj, landmarks=landmark_positions)
    end_time = time.perf_counter()
    print('Running time: %.2f s' % (end_time - start_time))
    '''Saving Result'''

    if os.path.exists('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip))):
        print('traj_{}_{}.npy exists...'.format(DATA_SUFFIX, str(skip)))
    else:
        print('Saving traj_{}_{}.npy...'.format(DATA_SUFFIX, str(skip)))
        np.save('traj_{}_{}.npy'.format(DATA_SUFFIX, str(skip)), pose_traj)

    if os.path.exists('map_{}_{}.npy'.format(DATA_SUFFIX, str(skip))):
        print('map_{}_{}.npy exists...'.format(DATA_SUFFIX, str(skip)))
    else:
        print('Saving map_{}_{}.npy...'.format(DATA_SUFFIX, str(skip)))
        np.save('map_{}_{}.npy'.format(DATA_SUFFIX, str(skip)), landmark_positions)
