import os
current_working_directory = os.getcwd()
print(current_working_directory)



import matplotlib.pyplot as plt


import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import scipy.io as sio

from tqdm import trange, tqdm

# Import the single DOF finite difference model MLP model
from mlp_quadruped_rnn import MLP, MLPQuadrupedProjectionFilter, CustomGRULayer, GRU_Hidden_State, CustomLSTMLayer, LSTM_Hidden_State

from scipy.spatial.transform import Rotation as R

import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

parser = argparse.ArgumentParser(description="Choose RNN module: LSTM or GRU")
parser.add_argument("--rnn_module", type=str, default="LSTM", help="Choose RNN module: LSTM or GRU")
args = parser.parse_args()

def sample_uniform_variables(key, var_min, var_max, dataset_size, nvar):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(
        low=var_min,
        high=var_max,
        size=(dataset_size, nvar)
    )
    return xi_samples, rng

# Parameters for Quadruped Model
num_batch = 1
timestep = 0.05  # 50 Hz control frequency
horizon = 10     # prediction horizon steps
num_legs = 4
friction_coeff = 0.2
body_mass = 50.0  # kg

body_inertia=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# Desired motion parameters
# desired_speed_x_tensor, rng = sample_uniform_variable(42, -0.5, 0.5, 1, 1)
# desired_speed_x = desired_speed_x_tensor.squeeze().item()
# print("desired_speed_x", desired_speed_x)
desired_body_height = 0.5     # m


# Problem dimensions for quadruped force control


##Parameters for MLP model
# Default states
BaseRollPitchYaw = (0.0, 0.0, 0.0)
AngularVelocityBodyFrame = (0.0, 0.0, 0.0)
ComVelocityBodyFrame = (0.0, 0.0, 0.0)
FootContacts = (True, True, True, True)
slope_estimate = (0.0, 0.0, 0.0)
RotationBodyWrtWorld = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

roll, pitch, yaw = BaseRollPitchYaw

# Create rotation object from Euler angles (in radians)
rot = R.from_euler('xyz', [roll, pitch, yaw])

# Convert to rotation matrix
rotation_matrix = rot.as_matrix()  # Shape (3, 3)

# Flatten into a 9-element tuple (row-major order)
RotationBodyWrtWorld = tuple(rotation_matrix.flatten())
#self.RotationBodyWrtWorld = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

"""Setup foot positions and default states for quadruped"""
# Foot positions in body frame
foot_x=0.2
foot_y=0.2
foot_z=-desired_body_height

FootPositionsInBodyFrame = torch.tensor([
    [ foot_x,  foot_y, foot_z],
    [-foot_x,  foot_y, foot_z],
    [ foot_x, -foot_y, foot_z],
    [-foot_x, -foot_y, foot_z]])

# Maximum Iterations
maxiter_projection = 20

nvar = 3*num_legs*horizon

num_total_constraints = 2*nvar



desired_speed_batched, rng_desired_speed_batched = sample_uniform_variables(41, var_min= -0.2, var_max = 0.2, dataset_size=num_batch, nvar=2)
desired_twisting_speed_batched,  rng_desired_twisting_speed_batched = sample_uniform_variables(
                                                                      42, var_min= -0.2, var_max = 0.2, dataset_size=num_batch, 
                                                                      nvar=1)

desired_speed_batched_val, rng_desired_speed_batched = sample_uniform_variables(40, var_min= -0.1, var_max = 0.1, dataset_size=num_batch, nvar=2)
desired_twisting_speed_batched_val,  rng_desired_twisting_speed_batched = sample_uniform_variables(
                                                                      39, var_min= -0.1, var_max = 0.1, dataset_size=num_batch,
                                                                      nvar=1)

print("desired_speed_batched.shape", desired_speed_batched.shape)
print("desired_twisting_speed_batched.shape", desired_twisting_speed_batched.shape)

inp = np.hstack((desired_speed_batched, desired_twisting_speed_batched))

if args.rnn_module == "GRU":
    print("Inferencing with GRU")
    #GRU handling
    rnn = "GRU"
    gru_input_size = 3*num_total_constraints+3*nvar
    # print(gru_input_size)
    gru_hidden_size = 512
    # gru_output_size = (2*nvar)**2+2*nvar
    gru_output_size = num_total_constraints+nvar
    # gru_context_size = mlp_planner_inp_dim

    gru_context = CustomGRULayer(gru_input_size, gru_hidden_size, gru_output_size)

    rnn_context = gru_context


    input_hidden_state_init = np.shape(inp)[1]
    mid_hidden_state_init = 512
    out_hidden_state_init = gru_hidden_size

    gru_init  =  GRU_Hidden_State(input_hidden_state_init, mid_hidden_state_init, out_hidden_state_init)
    
    rnn_init = gru_init
    ##
elif args.rnn_module == "LSTM":
    print("Inferencing with LSTM")
    #LSTM handling
    rnn = "LSTM"
    lstm_input_size = 3*num_total_constraints+3*nvar
    # print(lstm_input_size)
    lstm_hidden_size = 512
    # lstm_output_size = (2*nvar)**2+2*nvar
    lstm_output_size = num_total_constraints+nvar
    # lstm_context_size = mlp_planner_inp_dim

    lstm_context = CustomLSTMLayer(lstm_input_size, lstm_hidden_size, lstm_output_size)

    rnn_context = lstm_context

    input_hidden_state_init = np.shape(inp)[1]
    mid_hidden_state_init = 512
    out_hidden_state_init = lstm_hidden_size

    lstm_init = LSTM_Hidden_State(input_hidden_state_init, mid_hidden_state_init, out_hidden_state_init)

    rnn_init = lstm_init

    ##

##


enc_inp_dim = np.shape(inp)[1] 
mlp_inp_dim = enc_inp_dim
hidden_dim = 1024
mlp_out_dim = 2*nvar + num_total_constraints #( xi_samples- 0:nvar, lamda_smples- nvar:2*nvar)

mlp =  MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)


# Create the quadruped model
model = MLPQuadrupedProjectionFilter(
    mlp=mlp,
    rnn_context=rnn_context, 
    rnn_init=rnn_init, 
    num_batch=num_batch,
    maxiter_projection=maxiter_projection,
    BaseRollPitchYaw=BaseRollPitchYaw,
    AngularVelocityBodyFrame=AngularVelocityBodyFrame,
    ComVelocityBodyFrame=ComVelocityBodyFrame,
    FootPositionsInBodyFrame=FootPositionsInBodyFrame,
    FootContacts=FootContacts,
    slope_estimate=slope_estimate,
    RotationBodyWrtWorld=RotationBodyWrtWorld, 
    desired_body_height=desired_body_height,
    body_mass=body_mass,
    body_inertia=body_inertia,
    num_legs=num_legs,
    friction_coeff=friction_coeff,
    timestep=timestep,
    horizon=horizon,
    rnn=rnn).to(device)

print(type(model))

model.load_state_dict(torch.load(f'./training_weights/mlp_learned_quadruped_{rnn}.pth', weights_only=True))
model.eval()

######
#Generate Test Data


inp_test = inp
inp_test = torch.tensor(inp_test).float()
inp_test = inp_test.to(device)
inp_mean = inp_test.mean()
inp_std = inp_test.std()
# inp_test = torch.vstack([inp_test] * num_batch)
inp_norm_test = (inp_test - inp_mean) / inp_std
inp_norm_test = inp_norm_test.to(device)



desired_speed_batched = np.array([[0.0, 0.0]])               # shape (1, 2)
desired_twisting_speed_batched = np.array([[0.0]])           # shape (1, 1)


desired_speed_test = torch.from_numpy(desired_speed_batched).float().to(device)
desired_twisting_speed_test = torch.from_numpy(desired_twisting_speed_batched).float().to(device)



with torch.no_grad():
    (xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost,
     res_primal_history, res_fixed_point_history, res_qp_cost_history) = model.decoder_function(inp_norm_test, desired_speed_test,
                                                                                                desired_twisting_speed_test, rnn)
# Convert to numpy for analysis

xi_filtered_np = np.array(xi_projected.cpu().detach().numpy())
prime_residuals_np = np.array(res_primal_history.cpu().detach().numpy())
fixed_residuals_np = np.array(res_fixed_point_history.cpu().detach().numpy())
# Print convergence statistics
print(f"\nConvergence Statistics:")
# print(f"Final primal residual - Mean: {np.mean(prime_residuals_np[-1]):.6f}, Max: {np.max(prime_residuals_np[-1]):.6f}")
# print(f"Final fixed point residual - Mean: {np.mean(fixed_residuals_np[-1]):.6f}, Max: {np.max(fixed_residuals_np[-1]):.6f}")

print(f"Prime residuals shape: {prime_residuals_np.shape}")
print(f"Fixed point residuals shape: {fixed_residuals_np.shape}")

#Save
os.makedirs('results_quadruped_{rnn}_inference', exist_ok=True)

#Start
print("Start")
print(f"Max Prime residuals start: {max(prime_residuals_np[0])}")
print(f"Max fixed residuals start: {max(fixed_residuals_np[0])}")
print(f"Min Prime residuals start: {min(prime_residuals_np[0])}")
print(f"Min fixed residuals start: {min(fixed_residuals_np[0])}")

#End
print("End")
print(f"Max Prime residuals end: {max(prime_residuals_np[-1])}")
print(f"Max fixed residuals end: {max(fixed_residuals_np[-1])}")
print(f"Min Prime residuals end: {min(prime_residuals_np[-1])}")
print(f"Min fixed residuals end: {min(fixed_residuals_np[-1])}")



np.savetxt('results_quadruped_{rnn}_inference/projected_trajectory.csv', xi_filtered_np, delimiter=',')
np.savetxt('results_quadruped_{rnn}_inference/prime_residuals.csv', prime_residuals_np, delimiter=',')
np.savetxt('results_quadruped_{rnn}_inference/fixed_residuals.csv', fixed_residuals_np, delimiter=',')