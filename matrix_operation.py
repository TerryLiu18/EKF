import numpy as np


class Matrix:
    def __init__(self) -> None:
        pass

    @classmethod
    def show_matrix(cls, matrix):
        print(matrix)
    
    @classmethod
    def show_matrix_shape(cls, matrix):
        print(matrix.shape)

    
    @classmethod
    def vec2skew(cls, vec):
        """
        Convert a vector to a skew symmetric matrix
        :param vec: vector
        :return: skew symmetric matrix (3x3)
        """
        vec = vec.reshape(-1)
        return np.array([[0, -vec[2], vec[1]], 
                         [vec[2], 0, -vec[0]], 
                         [-vec[1], vec[0], 0]])
    
    @classmethod
    def skew2vec(cls, skew):
        """
        Convert skew symmetric matrix to vector
        :param skew: skew symmetric matrix (3x3)
        :return: vector (3,)
        """
        assert skew.shape == (3,3), 'skew matrix should be 3*3'
        return np.array([skew[2,1], skew[0,2], skew[1,0]], dtype=np.float64)

    @classmethod
    def vec2mat(cls, vec):
        return np.array([[0, -vec[2], vec[1]], 
                         [vec[2], 0, -vec[0]], 
                         [-vec[1], vec[0], 0]], dtype=np.float64)

    @classmethod
    def isso3(cls, matrix):
        assert matrix.shape == (3,3), 'matrix should be 3*3'
        return np.allclose(np.linalg.det(matrix), 1)
    
    @classmethod
    def isSE3(cls, matrix):
        assert matrix.shape == (4,4), 'matrix should be 4*4'

    @classmethod
    def isSO3(cls, matrix):
        assert matrix.shape == (3,3), 'matrix should be 3*3'
        return np.allclose(np.linalg.det(matrix), 1)

    @classmethod
    def ispose(cls, matrix):
        assert matrix.shape == (6,1), 'matrix should be 6*1'
    
    """
    u_t = [v_t, w_t]^T \in R6
    u_hat_t = [[w_hat_t, v_t^T], [0, 0]] \in R4x4
    u_spike_t = [[w_hat_t, v_hat_t][0, w_hat_t]] \in R6x6
    """

    @classmethod
    def pose2hat(cls, pose):
        """
        u_t = [v_t, w_t]^T \in R6
        u_hat_t = [[w_hat_t, v_t^T], [0, 0]] \in R4x4
        u_spike_t = [[w_hat_t, v_hat_t][0, w_hat_t]] \in R6x6
        """
        # assert pose.shape == (6,1), 'pose should be 6*1'
        v_t, w_t = pose[:3].reshape(3, 1), pose[3:].reshape(3, 1)
        # print(v_t, w_t)
        # print(v_t.shape, w_t.shape)
        w_hat_t = cls.vec2skew(w_t)
        # v_hat_t = cls.vec2skew(v_t)
        hat = np.block([[w_hat_t, v_t],
                        [np.zeros((1,3)), 0]])
        hat = hat.astype(np.float64)
        assert hat.shape == (4, 4), 'result should be 4*4'
        return hat
    
    @classmethod
    def hat2pose(cls, pose):
        """
        u_t = [v_t, w_t]^T \in R6
        u_hat_t = [[w_hat_t, v_t^T], [0, 0]] \in R4x4
        u_spike_t = [[w_hat_t, v_hat_t][0, w_hat_t]] \in R6x6
        """
        assert pose.shape == (4,4), 'pose should be 4*4'
        w_hat_t, v_t = pose[:3,:3], pose[:3,3]
        w_t = cls.skew2vec(w_hat_t)
        pose = np.vstack((w_t, v_t), dtype=np.float64)
        return pose

    @classmethod
    def pose2spike(cls, pose):
        """
        u_t = [v_t, w_t]^T \in R6
        u_hat_t = [[w_hat_t, v_t^T], [0, 0]] \in R4x4
        u_spike_t = [[w_hat_t, v_hat_t][0, w_hat_t]] \in R6x6
        """
        assert pose.shape == (6, 1), 'pose should be 6*1'
        v_t, w_t = pose[:3].reshape(3, 1), pose[3:].reshape(3, 1)
        w_hat_t = cls.vec2skew(w_t)
        v_hat_t = cls.vec2skew(v_t)
        spike = np.block([[w_hat_t, v_hat_t], 
                          [np.zeros((3, 3)), w_hat_t]])
        spike = spike.astype(np.float64)
        assert spike.shape == (6, 6), 'result should be 6*6'
        return spike

    @classmethod
    def circle_dot(cls, s):
        """
        circle_dot
        :param s: 4x1 in homogenous coordinate
        :return: 4x6
        """
        s = s.reshape(-1)
        assert s.size == 4
        assert abs(s[-1] - 1) < 1e-8
        res = np.hstack([np.eye(3, dtype=np.float64), -cls.vec2skew(s[:3])])
        res = np.vstack([res, np.zeros((1, 6), dtype=np.float64)])
        return res


class Camera:
    def __init__(self) -> None:
        pass
    
    @classmethod
    def build_Ks(cls, fsu, fsv, cu, cv, b):
        """
        Stereo Camera Calibration Matrix
        :param: fs_u: focal length [m],  pixel scaling [pixels/m]
        :param: fs_v: focal length [m],  pixel scaling [pixels/m]
        :param: cu: principal point [pixels]
        :param: cv: principal point [pixels]
        :param: b: stereo baseline [m]
        :return 4x4 stereo camera calibration matrix
        """
        Ks = np.array([[fsu, 0, cu, 0],
                      [0, fsv, cv, 0],
                      [fsu, 0, cu, -fsu * b],
                      [0, fsv, cv, 0]],
                      dtype=np.float64)
        return Ks
    
    @classmethod
    def get_para_from_K(cls, K, b):
        fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
        fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
        cu = K[0, 2]  # principal point [pixels]
        cv = K[1, 2]  # principal point [pixels]
        b = float(b)  # stereo baseline [m]
        return fs_u, fs_v, cu, cv, b

    @classmethod
    def pixel2world(cls, pixels, K, b, world_T_cam):
        """
        Convert from pixels to world coordinates
        :param pixels: pixel coordinates with number of observations
        :param K: (left)camera intrinsic matrix with shape 3*3
        :param b: stereo camera baseline with shape 1
        :param world_T_cam: pose from camera to world
        :return: world frame landmark positions in homogenous coordinates
        """
        fs_u = K[0, 0]  # focal length [m],  pixel scaling [pixels/m]
        fs_v = K[1, 1]  # focal length [m],  pixel scaling [pixels/m]
        cu = K[0, 2]  # principal point [pixels]
        cv = K[1, 2]  # principal point [pixels]
        m = np.ones((4, pixels.shape[1]), dtype=np.float64)
        m[2, :] = fs_u * b / (pixels[0, :] - pixels[2, :])
        m[1, :] = (pixels[1, :] - cv) / fs_v * m[2, :]
        m[0, :] = (pixels[0, :] - cu) / fs_u * m[2, :]
        m_world = world_T_cam.astype(np.float64) @ m
        return m_world
    
    @classmethod
    def projection(cls, q):
        """
        Projection Function
        π(q) := 1/q3 @ q  ∈ R^{4}
        :param q: numpy.array
        """
        pi_q = q / q[2, :]
        return pi_q

    @classmethod
    def projection_derivative(cls, q):
        """
        Projection Function Derivative
        calculate dπ(q)/dq
        :param q: numpy.array
        return: dπ(q)/dq 4x4
        """

        q1, q2, q3, q4 = q[0], q[1], q[2], q[3]
        dpi_dq = (1/q3) * np.array([[1, 0, -q1/q3, 0],
                                    [0, 1, -q2/q3, 0],
                                    [0, 0,    0,   0],
                                    [0, 0, -q4/q3, 0]], dtype=np.float64)
        return dpi_dq



class Derivative:
    def __init__(self):
        pass

    @classmethod
    def vec2skew(cls, vec):
        """
        Convert a vector to a skew symmetric matrix
        :param vec: vector
        :return: skew symmetric matrix (3x3)
        """
        vec = vec.reshape(-1)
        return np.array([[0, -vec[2], vec[1]],
                         [vec[2], 0, -vec[0]],
                         [-vec[1], vec[0], 0]])

    @classmethod
    def hadamard(cls, s):
        """
        :param s: 4x1 in homogenous coordinate
        :return: 4x6
        """
        s = s.reshape(-1)
        assert s.size == 4
        assert abs(s[-1] - 1) < 1e-8
        res = np.hstack([np.eye(3, dtype=np.float64), -cls.vec2skew(s[:3])])
        res = np.vstack([res, np.zeros((1, 6), dtype=np.float64)])
        return res

    @classmethod
    def projection_derivative(cls, q):
        """
        Projection Function Derivative
        calculate dπ(q)/dq
        :param q: numpy.array
        return: dπ(q)/dq 4x4
        """
        dpi_dq = np.eye(4, dtype=np.float64)
        dpi_dq[2, 2] = 0.0
        dpi_dq[0, 2] = -q[0] / q[2]
        dpi_dq[1, 2] = -q[1] / q[2]
        dpi_dq[3, 2] = -q[3] / q[2]
        dpi_dq = dpi_dq / q[2]
        return dpi_dq

    @classmethod
    def projection_Jacobian(cls, Ks, opt_T_imu, T_imu_inv, Mt, update_feature_index, m):
        """
        Combined Jacobian
        :param Ks: 4x4 stereo camera calibration matrix
        :param opt_T_imu: 4x4 transformation matrix {IMU} -> {CAM}
        :param T_imu_inv: 4x4 transformation matrix {W} -> {IMU}
        :param Mt: number_of_landmarks
        :param update_feature_index: index of update features
        :param m: landmarks position in world frame
        :return:
        """
        P = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0]],
                       dtype=np.float64)
        Nt = update_feature_index.size
        opt_T_world = opt_T_imu @ T_imu_inv

        H = np.zeros((4 * Nt, 3 * Mt + 6), dtype=np.float64)  # Ht+1 ∈{4Ntx3M+6}

        for j in range(Nt):
            idx = update_feature_index[j]
            dpi_dq = cls.projection_derivative(opt_T_world @ m[:, j])
            H_ij_obs = np.linalg.multi_dot([Ks, dpi_dq, opt_T_world, P])            # 4×3
            H[4*j: 4*j+4, 3*idx: 3*idx+3] = H_ij_obs

            prod = T_imu_inv @ m[:, j]
            dpi_dq_loc = cls.projection_derivative(opt_T_imu @ prod)
            s_circle_dot = cls.hadamard(prod)
            H_ij_act = np.linalg.multi_dot([-Ks, dpi_dq_loc, opt_T_imu, s_circle_dot])  # 4×6
            H[j * 4:(j + 1) * 4, -6:] = H_ij_act
        return H


if __name__ == '__main__':
    matrix = Matrix()
    print(Matrix.pose2hat(np.array([1, 2, 3, 4, 5, 6])))
