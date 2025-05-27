import numpy as np
import scipy as sc


def ConvertToSkewSymmetric(x: np.ndarray):
    return np.array([[   0, -x[2],  x[1]],
                    [ x[2],     0, -x[0]],
                    [-x[1],  x[0],    0]], dtype=float)


class SRBD():

    def __init__(self, dt, num_legs):

        self.dt = dt
        self.num_legs = num_legs


    def calculateStateTransitionMatrix(self, torso_rpy, frame='body'):

        A = np.zeros((13, 13))

        sin_roll = np.sin(torso_rpy[0])
        cos_roll = np.cos(torso_rpy[0])
        sin_pitch = np.sin(torso_rpy[1])
        cos_pitch = np.cos(torso_rpy[1])
        tan_pitch = np.tan(torso_rpy[1])
        sin_yaw = np.sin(torso_rpy[2])
        cos_yaw = np.cos(torso_rpy[2])
        
        if frame == 'body':

            option = 1
            
            if option == 1:
                temp_matrix = np.array([
                    [1, 0, -sin_pitch],
                    [0, cos_roll, cos_pitch*sin_roll],
                    [0, -sin_roll, cos_pitch*cos_roll]])

                angular_velocity_to_rpy_rate = np.linalg.pinv(temp_matrix)
            
            else:
                angular_velocity_to_rpy_rate = np.array([
                    [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
                    [-sin_yaw, cos_yaw, 0],
                    [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]])
        
            A[0:3, 6:6 + 3] = angular_velocity_to_rpy_rate
            A[3, 9] = 1
            A[4, 10] = 1
            A[5, 11] = 1

            A[9, 12] = -sin_pitch * cos_roll
            A[10, 12] = sin_roll
            A[11, 12] = cos_pitch * cos_roll

        elif frame == 'world':        
            angular_velocity_to_rpy_rate = np.array([
                [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
                [-sin_yaw, cos_yaw, 0],
                [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]])
        
            A[0:3, 6:6 + 3] = angular_velocity_to_rpy_rate
            A[3, 9] = 1
            A[4, 10] = 1
            A[5, 11] = 1

            A[9, 12] = 0
            A[10, 12] = 0
            A[11, 12] = 1

        return A


    def calculateInputMatrix(self, foot_positions_i, foot_contact_masks_i, inv_mass, inv_inertia):

        # ic(inv_inertia)
        # ic(foot_positions_i)

        B = np.zeros((13, 3*self.num_legs))
        for j in range(self.num_legs):
            B[6:6 + 3, j * 3:j * 3 + 3] = foot_contact_masks_i[j] * \
                                inv_inertia @ ConvertToSkewSymmetric(foot_positions_i[j])
            B[9, j * 3] = foot_contact_masks_i[j] * inv_mass
            B[10, j * 3 + 1] = foot_contact_masks_i[j] * inv_mass
            B[11, j * 3 + 2] = foot_contact_masks_i[j] * inv_mass

            # x = inv_inertia @ ConvertToSkewSymmetric(foot_positions_i[j])
            # ic(x)

        # According to Euler's integration scheme
        # B_discrete = B.T
        # where T is the sampling interval (1/sampling_frequency)

        # B_discrete = B*self.dt
        # return B_discrete

        return B


    def getDiscreteMatrices(self, A_continuous, B_continuous, matrix_exp = False):

        if matrix_exp:
            n = A_continuous.shape[0] # State dim
            m = B_continuous.shape[1] # Action dim

            AB_concatenated = np.zeros((n+m, n+m))
            AB_concatenated[0:n, 0:n] = A_continuous*self.dt
            AB_concatenated[0:n, n:n+m] = B_continuous*self.dt

            AB_concatenated_exp = sc.linalg.expm(AB_concatenated)
            A_discrete = AB_concatenated_exp[0:n, 0:n]
            B_discrete = AB_concatenated_exp[0:n, n:n+m]

        else:
            A_discrete = np.eye(A_continuous.shape[0]) + A_continuous * self.dt
            B_discrete = B_continuous*self.dt

        # print("A_discrete: \n", A_discrete)
        # print("B_discrete: \n", B_discrete)
   
            
        return A_discrete, B_discrete