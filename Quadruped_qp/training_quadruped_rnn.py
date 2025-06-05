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
import argparse

# Import the quadruped model components from your second file
from mlp_quadruped_rnn import MLP, MLPQuadrupedProjectionFilter, CustomGRULayer, GRU_Hidden_State, CustomLSTMLayer, LSTM_Hidden_State


from scipy.spatial.transform import Rotation as R

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ForceDataset(Dataset):
    """Expert Trajectory Dataset."""
    def __init__(self, inp, desired_speed, desired_twisting_speed):
        # input
        self.inp = inp
        self.desired_speed = desired_speed
        self.desired_twisting_speed = desired_twisting_speed

    def __len__(self):
        return len(self.inp)    
        
    def __getitem__(self, idx):
        # Input
        inp = self.inp[idx]
        desired_speed = self.desired_speed[idx]
        desired_twisting_speed = self.desired_twisting_speed
        return torch.tensor(inp).float(), torch.tensor(desired_speed).float(), torch.tensor(desired_twisting_speed).float()    


def sample_uniform_variables(key, var_min, var_max, dataset_size, nvar):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(
        low=var_min,
        high=var_max,
        size=(dataset_size, nvar)
    )
    return xi_samples, rng

parser = argparse.ArgumentParser(description="Choose RNN module: LSTM or GRU")
parser.add_argument("--rnn_module", type=str, default="LSTM", help="Choose RNN module: LSTM or GRU")
args = parser.parse_args()

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

# Define QP matrices for quadruped control (simplified example)
# In practice, these would come from your quadruped dynamics model




# Maximum Iterations
maxiter_projection = 5

nvar = 3*num_legs*horizon

num_total_constraints = 2*nvar

# Generate training and validation data
# inp = xi_samples
# inp_val = xi_val


#desired_twisting_speed_batched = np.full((num_batch, 1), desired_twisting_speed)


dataset_size = 1*num_batch
desired_speed_batched, rng_desired_speed_batched = sample_uniform_variables(None, var_min= -0.0, var_max = 0.0, dataset_size=dataset_size, nvar=2)
desired_twisting_speed_batched,  rng_desired_twisting_speed_batched = sample_uniform_variables(
                                                                      42, var_min= -0.0, var_max = 0.0, dataset_size=dataset_size, 
                                                                      nvar=1)

desired_speed_batched_val, rng_desired_speed_batched = sample_uniform_variables(None, var_min= -0.1, var_max = 0.1, dataset_size=dataset_size, nvar=2)
desired_twisting_speed_batched_val,  rng_desired_twisting_speed_batched = sample_uniform_variables(
                                                                      39, var_min= -0.1, var_max = 0.1, dataset_size=dataset_size,
                                                                      nvar=1)

print("desired_speed_batched.shape", desired_speed_batched.shape)
print("desired_twisting_speed_batched.shape", desired_twisting_speed_batched.shape)

inp = np.hstack((desired_speed_batched, desired_twisting_speed_batched))

inp_val = np.hstack(( desired_speed_batched_val, desired_twisting_speed_batched_val))

# Using PyTorch Dataloader
train_dataset = ForceDataset(inp, desired_speed_batched, desired_twisting_speed_batched)
val_dataset = ForceDataset(inp_val, desired_speed_batched_val, desired_twisting_speed_batched_val)

train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=num_batch, shuffle=True, num_workers=0, drop_last=True)

if args.rnn_module == "GRU":
    print("Training with GRU")
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
    print("Training with LSTM")
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

# MLP setup
enc_inp_dim = np.shape(inp)[1] 
mlp_inp_dim = enc_inp_dim
hidden_dim = 1024
mlp_out_dim = 2*nvar + num_total_constraints  # xi_samples, lambda_samples, slack_variables

mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)

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

print(f"Model type: {type(model)}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Training
epochs = 3000
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=6e-5)

losses = []
last_loss = torch.inf
avg_train_loss, avg_primal_loss, avg_fixed_point_loss, avg_qp_cost_loss = [], [], [], []
avg_val_loss = []

for epoch in range(epochs):
    
    # Train Loop
    model.train()
    losses_train, primal_losses, fixed_point_losses, qp_cost_losses = [], [], [], []
    
    for (inp, desired_speed_batched, desired_twisting_speed_batched) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        
        desired_speed_batched, rng_desired_speed_batched = sample_uniform_variables(None, var_min= -2, var_max = 2, dataset_size=dataset_size, nvar=2)
        #desired_twisting_speed_batched, rng_desired_twisting_speed_batched = sample_uniform_variables(None, var_min= -0.5, var_max = 0.5, 
                                                                                                     # dataset_size=dataset_size, nvar=1)
        
        
        print("desired_speed_batched", desired_speed_batched)

        desired_speed_batched = torch.tensor(desired_speed_batched).float()
        #desired_twisting_speed_batched = torch.tensor(desired_twisting_speed_batched).float()
        # Input and Output 
        inp = inp.to(device)
        desired_speed_batched = desired_speed_batched.to(device)
        desired_twisting_speed_batched = desired_twisting_speed_batched.to(device)
        
        # Forward pass through quadruped model
        xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost, res_primal_history, res_fixed_point_history, _ = model(inp, desired_speed_batched, 
                                                                                                               desired_twisting_speed_batched, rnn)
        
        
        # Compute loss
        primal_loss, fixed_point_loss, qp_cost_loss ,loss = model.mlp_loss(
            avg_res_primal, avg_res_fixed_point, avg_res_qp_cost)

        optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses_train.append(loss.detach().cpu().numpy()) 
        primal_losses.append(primal_loss.detach().cpu().numpy())
        fixed_point_losses.append(fixed_point_loss.detach().cpu().numpy())
        qp_cost_losses.append(qp_cost_loss.detach().cpu().numpy())
        #projection_losses.append(projection_loss.detach().cpu().numpy())
        
    # Validation every 2 epochs
    if epoch % 2 == 0:
        model.eval()
        val_losses = []

        with torch.no_grad():
            for (inp_val, desired_speed_batched_val, desired_twisting_speed_batched_val) in tqdm(val_loader, desc="Validation"):
                inp_val = inp_val.to(device)
                desired_speed_batched_val = desired_speed_batched_val.to(device)
                desired_twisting_speed_batched_val = desired_twisting_speed_batched_val.to(device)

                xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost, res_primal_history, res_fixed_point_history, _ = model(inp_val, desired_speed_batched_val, 
                                                                                                                       desired_twisting_speed_batched_val,
                                                                                                                       rnn)
                
                _, _, _, val_loss = model.mlp_loss(
                    avg_res_primal, avg_res_fixed_point, avg_res_qp_cost
                )

                val_losses.append(val_loss.detach().cpu().numpy())

    # Print progress every 2 epochs
    if epoch % 2 == 0:    
        print(
            f"Epoch: {epoch + 1}, Train Loss: {np.average(losses_train):.4f}, "
            f"Primal Loss: {np.average(primal_losses):.4f}, "
            f"Fixed-Point Loss: {np.average(fixed_point_losses):.4f}, "
            f"QP Cost Loss: {np.average(qp_cost_losses):.4f}"
        )

              #f"Projection Loss: {np.average(projection_losses):.4f}")
        
        if len(val_losses) > 0:
            print(f"Validation Loss: {np.average(val_losses):.4f}")

    # Save best model
    os.makedirs("./training_weights", exist_ok=True)
    if loss <= last_loss:
        torch.save(model.state_dict(), f"./training_weights/mlp_learned_quadruped_{rnn}.pth")
        last_loss = loss

    # Store metrics
    avg_train_loss.append(np.average(losses_train))
    avg_primal_loss.append(np.average(primal_losses))
    avg_qp_cost_loss.append(np.average(qp_cost_losses))
    avg_fixed_point_loss.append(np.average(fixed_point_losses))
    
    if len(val_losses) > 0:
        avg_val_loss.append(np.average(val_losses))
    else:
        avg_val_loss.append(avg_val_loss[-1] if avg_val_loss else 0.0)

print("Training completed!")

