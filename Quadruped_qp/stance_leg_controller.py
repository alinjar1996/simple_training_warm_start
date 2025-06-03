

"""A torque based stance controller framework."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

from typing import Any, Sequence, Tuple
from scipy.spatial.transform import Rotation as R

import qpsolvers

from srbd import SRBD
from qp_solver import QPCostMatrices



class ForceStanceLegController():
  """A torque based stance leg controller framework.

  Takes in high level parameters like walking speed and turning speed, and
  generates necessary the torques for stance legs.
  """
  def __init__(
      self,
      desired_speed: Tuple[float, float],
      desired_twisting_speed: float,
      desired_body_height: float,
      body_mass: float,
      body_inertia: Tuple[float, float, float, float, float, float, float,
                          float, float],
      num_legs: int,
      friction_coeff: float,
      timestep: float,
      horizon: int,

  ):
    """Initializes the class.

    Tracks the desired position/velocity of the robot by computing proper joint
    torques using MPC module.

    Args:
      robot: A robot instance.
      desired_speed: desired CoM speed in x-y plane.
      desired_twisting_speed: desired CoM rotating speed in z direction.
      desired_body_height: The standing height of the robot.
      body_mass: The total mass of the robot.
      body_inertia: The inertia matrix in the body principle frame. We assume
        the body principle coordinate frame has x-forward and z-up.
      num_legs: The number of legs used for force planning.
      friction_coeff: The friction coeffs on the contact surfaces.
      timestep: The timestep of the simulation.
      horizon: The MPC planning horizon in terms of numberof steps.
    """

    
    self.dt = timestep
    self.inv_mass = 1/body_mass
    self.inv_inertia_body = np.linalg.pinv(np.reshape(body_inertia, (3,3)))
    self.num_legs = num_legs

    self.friction_coeffs = np.array([friction_coeff]*num_legs)
    self.max_z_force = body_mass*9.81/2.0
    self.horizon = horizon

    mpc_force_weight_scale= 5e-6
    mpc_weight_rpy = np.array([0.2, 0.2, 0.0])
    mpc_weight_rpy_dot = np.array([0.2, 0.2, 0.35])
    mpc_weight_xyz = np.array([0.7, 0.7, 1.0])
    mpc_weight_xyz_dot = np.array([0.3, 0.3, 1.0])
   
   
    # Flatten into a single 1D array
    MPC_WEIGHTS = np.concatenate([
        mpc_weight_rpy,
        mpc_weight_xyz,
        mpc_weight_rpy_dot,
        mpc_weight_xyz_dot,
        np.array([0.0])
    ])

    self.L_i = np.diag(MPC_WEIGHTS)
    self.K_i = np.eye(3*num_legs)*mpc_force_weight_scale 

    self.srbd = SRBD(self.dt, num_legs)
    self._n_count = 0

    self.desired_speed = desired_speed
    self.desired_twisting_speed = desired_twisting_speed
    self.desired_body_height = desired_body_height
    
    self.Ut = np.zeros((num_legs*3, 1), dtype=np.float64)
    self.Ut[[2,5,8,11], :] = np.array([body_mass*9.8/4 for _ in range(num_legs)]).reshape(-1, 1)
    self.Ud = np.zeros((12, horizon), dtype=np.float64)


  def _calculateFrictionCone(self, foot_contact_state, add_contact_mask=False):

    if self.friction_coeffs is None:
        raise("Friction coeff not set")

    if self.num_legs is None:
        self.num_legs = len(self.friction_coeffs)
    else:
        assert self.num_legs == len(self.friction_coeffs)

    # right_mu, left_mu = self.friction_coeffs
    c = np.zeros(6*self.num_legs*self.horizon)
    C = np.zeros((6*self.num_legs*self.horizon, 3*self.num_legs*self.horizon))

    for i in range(self.horizon):

        for j in range(self.num_legs):
            if add_contact_mask:
              c[i*12 + j*6] = self.max_z_force*foot_contact_state[j] + 15
            else:
              c[i*12 + j*6] = self.max_z_force
            c[i*12 + j*6 + 1] = 0

            #   z force
            C[i*12 + j*6, i*6 + j*3:i*6 + j*3 + 3] = [0, 0, 1]
            C[i*12 + j*6 + 1, i*6 + j*3:i*6 + j*3 + 3] = [0, 0, -1]
            #   friction cone in x direction
            C[i*12 + j*6 + 2, i*6 + j*3:i*6 + j*3 + 3] = [1, 0, -1*self.friction_coeffs[j]]
            C[i*12 + j*6 + 3, i*6 + j*3:i*6 + j*3 + 3] = [-1, 0, -1*self.friction_coeffs[j]]
            #   friction cone in y direction
            C[i*12 + j*6 + 4, i*6 + j*3:i*6 + j*3 + 3] = [0, 1, -1*self.friction_coeffs[j]]
            C[i*12 + j*6 + 5, i*6 + j*3:i*6 + j*3 + 3] = [0, -1, -1*self.friction_coeffs[j]]

    return c, C

  def _calculateQPMatrices(self, A_mats, B_mats, change_A_with_time=True, change_B_with_time=True):

    A_i = np.array(A_mats[0]).copy()
    A_qp = np.array(A_mats[0]).copy()
    B_i = np.array(B_mats[0]).copy()
    B_qp = np.array(B_mats[0]).copy()

    for i in range(1, self.horizon):

        if change_B_with_time:
            B_i = np.hstack((A_i@B_mats[i], B_i))
        else:
            B_i = np.hstack((A_i@B_mats[0], B_i))

        if change_A_with_time:
            A_i = A_i@A_mats[i]
        else:
            A_i = A_i@A_mats[0]

        A_qp = np.vstack((A_qp, A_i))
        B_qp = np.hstack((B_qp, np.zeros((B_qp.shape[0], B_mats[0].shape[1])))) # To match dimensions
        B_qp = np.vstack((B_qp, B_i))

    return A_qp, B_qp
  

  def _calculateReferenceTrajectory(self, desired_com_position, desired_com_velocity, 
                                    desired_com_roll_pitch_yaw, desired_com_rpy_rate, slope_estimate):
    x_ref = []
    x_i = np.array([*desired_com_roll_pitch_yaw, *desired_com_position, 
                    *desired_com_rpy_rate, *desired_com_velocity, -9.81])
    
    for i in range(self.horizon):
      x_i[0:3] += desired_com_rpy_rate*self.dt
      x_i[3:6] += desired_com_velocity*self.dt
      x_ref.append(x_i.copy())

    return np.array(x_ref).ravel()


  def reset(self, current_time):
    del current_time

  def update(self, current_time):
    del current_time


  def get_com_height(self, foot_pos_body, contact_state):
    
    h = 0
    n = 0
    for i in range(len(contact_state)):
      if contact_state[i] is True:
        h += foot_pos_body[i][2]
        n += 1

    if n == 0:
      return self.desired_body_height
    else:
      return -h/n


  def getMatrices(self, 
                     BaseRollPitchYaw, 
                     AngularVelocityBodyFrame, 
                     ComVelocityBodyFrame,
                     FootPositionsInBodyFrame,
                     FootContacts,
                     slope_estimate,
                     RotationBodyWrtWorld,
                     Training: bool):
    """Computes the torque for stance legs."""
    # desired_speed is in the body frame
    desired_com_velocity = np.array((self.desired_speed[0], self.desired_speed[1], 0.), dtype=np.float64)
    desired_com_rpy_rate = np.array((0., 0., self.desired_twisting_speed), dtype=np.float64)
    foot_contact_state = FootContacts

    desired_com_roll_pitch_yaw = np.array((slope_estimate[0], slope_estimate[1], BaseRollPitchYaw[2]), dtype=np.float64)
    #desired_com_roll_pitch_yaw = np.array((0, 0, BaseRollPitchYaw[2]), dtype=np.float64)
    BaseRollPitchYaw = np.array(BaseRollPitchYaw, dtype=np.float64)
    

    com_height_from_ground = self.get_com_height(FootPositionsInBodyFrame, FootContacts)
    desired_com_height_body_frame = self.desired_body_height - com_height_from_ground
    FootPositionsInBodyFrame = FootPositionsInBodyFrame.flatten()

    foot_pos_base_frame = [np.array(FootPositionsInBodyFrame[i*3: i*3+3]) for i in range(self.num_legs)]

    com_pos = [0.0, 0.0, 0.0]
    desired_com_position = [0.0, 0.0, desired_com_height_body_frame]

    # BaseRollPitchYaw is in the original world frame (start of simulation), rest all values are in body frame
    x_init = np.array([*BaseRollPitchYaw, *com_pos, *AngularVelocityBodyFrame, *ComVelocityBodyFrame, -9.81])
    x_ref = self._calculateReferenceTrajectory(desired_com_position, desired_com_velocity, 
                                    desired_com_roll_pitch_yaw, desired_com_rpy_rate, slope_estimate)

    # Since we don't have the foot placement trajectory, we're using the same A and B matrices throughout the horizon
    # Changed the inv_inerta and foot pos to world frame
    A_cont = self.srbd.calculateStateTransitionMatrix(BaseRollPitchYaw, frame='body')
    B_cont = self.srbd.calculateInputMatrix(foot_pos_base_frame, foot_contact_state, self.inv_mass, self.inv_inertia_body)


    A, B = self.srbd.getDiscreteMatrices(A_cont, B_cont, matrix_exp=False)

    # We're keeping the A and B constant
    A_mats = [A]*self.horizon
    B_mats = [B]*self.horizon

    A_qp, B_qp = self._calculateQPMatrices(A_mats, B_mats)
    c, C = self._calculateFrictionCone(foot_contact_state)

    qp_costmatrices = QPCostMatrices(self.L_i, self.K_i, self.horizon)
    
    qp_costmatrices._createFullCostMatrices()

    self.L = qp_costmatrices.L
    self.K = qp_costmatrices.K
    
    H = 2*(B_qp.T@self.L@B_qp + self.K)
    g = 2*B_qp.T@self.L@(A_qp@x_init - x_ref)

    if Training:
      U = np.zeros((3*self.num_legs*self.horizon, ))
    else:    
      U = np.array(qpsolvers.solve_qp(H, g, C, c, solver="clarabel", verbose=False))
    # U = np.array(qpsolvers.solve_qp(A_qp, B_qp, x_init, x_ref, (c, C)))

    # try:
    #   x_res = A_qp@x_init - B_qp@U
    # except Exception as e:
    #    print(e)
    # self._n_count += 1
    return H, g, C, c, U