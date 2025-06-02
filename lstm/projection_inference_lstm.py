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
from mlp_singledof_lstm import MLP, MLPProjectionFilter, CustomLSTMLayer, LSTM_Hidden_State

def sample_uniform_trajectories(key, var_min, var_max, dataset_size, nvar):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(
        low=var_min,
        high=var_max,
        size=(dataset_size, nvar)
    )
    return xi_samples, rng

#Parameters for MLP model

num_batch = 1000
num_dof=1
num_steps=50
timestep=0.05
v_max=1.0
a_max=2.0
j_max=5.0
p_max=180.0*np.pi/180.0 
theta_init=0.0

# vmax = 1.0
# num_batch = 1000
# nvar = 1
nvar_single = num_steps
nvar = num_dof * nvar_single
theta_init_min=0.0
theta_init_max=2*np.pi


#calculating number of constraints
num_acc = num_steps - 1
num_jerk = num_acc - 1
num_pos = num_steps
num_vel_constraints = 2 * num_steps * num_dof
num_acc_constraints = 2 * num_acc * num_dof
num_jerk_constraints = 2 * num_jerk * num_dof
num_pos_constraints = 2 * num_pos * num_dof
num_total_constraints = (num_vel_constraints + num_acc_constraints + 
                            num_jerk_constraints + num_pos_constraints)
dataset_size = num_batch#200000

#Maximum Iterations
maxiter_projection = 20


# Cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

#For training
theta_init, rng_theta_init = sample_uniform_trajectories(41, var_min= theta_init_min, var_max = theta_init_max, dataset_size=dataset_size, nvar=1)
#print("theta_init", theta_init.shape)
v_start, rng_v_start = sample_uniform_trajectories(40, var_min =-0.8*v_max, var_max = 0.8*v_max, dataset_size=dataset_size, nvar=1)
#print("v_start", v_start.shape)
v_goal, rng_v_goal = sample_uniform_trajectories(39, var_min =-0.8*v_max, var_max = 0.8*v_max, dataset_size=dataset_size, nvar=1)

#For Testing with Scalar value
theta_init_scalar = 0.0*np.pi
v_start_scalar = 0.0
v_goal_scalar = 0.0
theta_init = np.tile(theta_init_scalar, (dataset_size,1))
v_start = np.tile(v_start_scalar, (dataset_size,1))
v_goal = np.tile(v_goal_scalar, (dataset_size,1))


#For training
xi_samples, rng = sample_uniform_trajectories(42, var_min=-v_max, var_max=v_max ,dataset_size=dataset_size, nvar=nvar)


inp = np.hstack(( xi_samples, theta_init, v_start, v_goal))




#LSTM handling

lstm_input_size = 3*num_total_constraints+3*nvar
# print(gru_input_size)
lstm_hidden_size = 512
# lstm_output_size = (2*nvar)**2+2*nvar
lstm_output_size = num_total_constraints+nvar
# lstm_context_size = mlp_planner_inp_dim

lstm_context = CustomLSTMLayer(lstm_input_size, lstm_hidden_size, lstm_output_size)


input_hidden_state_init = np.shape(inp)[1]
mid_hidden_state_init = 512
out_hidden_state_init = lstm_hidden_size

lstm_init  =  LSTM_Hidden_State(input_hidden_state_init, mid_hidden_state_init, out_hidden_state_init)

##

enc_inp_dim = np.shape(inp)[1] 
mlp_inp_dim = enc_inp_dim
hidden_dim = 1024
mlp_out_dim = 2*nvar + num_total_constraints #( xi_samples- 0:nvar, lamda_smples- nvar:2*nvar)

mlp =  MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)



model = MLPProjectionFilter(mlp=mlp,lstm_context=lstm_context, lstm_init=lstm_init, num_batch = num_batch,num_dof=num_dof,num_steps=num_steps,
							timestep=timestep,v_max=v_max,a_max=a_max,j_max=j_max,p_max=p_max, 
                            maxiter_projection=maxiter_projection).to(device)

print(type(model))

model.load_state_dict(torch.load('./training_weights/mlp_learned_single_dof_lstm.pth', weights_only=True))
model.eval()


######
#Generate Test Data


inp_test = inp
inp_test = torch.tensor(inp_test).float()
inp_test = inp_test.to(device)
inp_mean = inp_test.mean()
inp_std = inp_test.std()

theta_init_test = torch.from_numpy(theta_init).float().to(device)
v_start_test = torch.from_numpy(v_start).float().to(device)
v_goal_test = torch.from_numpy(v_goal).float().to(device)

# inp_test = torch.vstack([inp_test] * num_batch)
inp_norm_test = (inp_test - inp_mean) / inp_std

xi_samples_input_nn_test = inp_test

with torch.no_grad():
    xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = model.decoder_function(inp_norm_test, xi_samples_input_nn_test, 
                                                                                                                            theta_init_test, v_start_test, 
                                                                                                                            v_goal_test)

# Convert to numpy for analysis
xi_np = np.array(xi_samples)
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
os.makedirs('results_LSTM_inference', exist_ok=True)

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


np.savetxt('results_LSTM_inference/original_trajectory.csv', xi_np, delimiter=',')  # Save first sample
np.savetxt('results_LSTM_inference/projected_trajectory.csv', xi_filtered_np, delimiter=',')
np.savetxt('results_LSTM_inference/prime_residuals.csv', prime_residuals_np, delimiter=',')
np.savetxt('results_LSTM_inference/fixed_residuals.csv', fixed_residuals_np, delimiter=',')