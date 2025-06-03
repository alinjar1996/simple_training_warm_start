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

# Import the quadruped model components from your second file
from Quadruped_qp.mlp_quadruped_rnn import MLP, MLPQuadrupedProjectionFilter, CustomGRULayer, GRU_Hidden_State
from stance_leg_controller import ForceStanceLegController

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ForceDataset(Dataset):
    """Expert Trajectory Dataset."""
    def __init__(self, inp):
        # input
        self.inp = inp

    def __len__(self):
        return len(self.inp)    
        
    def __getitem__(self, idx):
        # Input
        inp = self.inp[idx]
        return torch.tensor(inp).float()    

def sample_uniform_forces(key, F_max, num_batch, nvar):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(
        low=-F_max,
        high=F_max,
        size=(num_batch, nvar)
    )
    return xi_samples, rng

def sample_uniform_variable(key, min, max, row_size, col_size):
    rng = np.random.default_rng(seed=key)
    xi_samples = rng.uniform(
        low=min,
        high=max,
        size=(row_size, col_size)
    )
    return xi_samples, rng

# Parameters for Quadruped Model
num_batch = 1000
timestep = 0.05  # 50 Hz control frequency
horizon = 10     # prediction horizon steps
num_legs = 4
friction_coeff = 0.2
body_mass = 50.0  # kg

body_inertia=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

# Desired motion parameters
desired_speed_x_tensor, rng = sample_uniform_variable(42, -0.5, 0.5, 1, 1)
desired_speed_x = desired_speed_x_tensor.squeeze().item()
print("desired_speed_x", desired_speed_x)
#desired_speed_x = 0.0  # m/s
desired_speed = (desired_speed_x, 0.0)        # m/s
desired_twisting_speed = 0.5  # rad/s
desired_body_height = 0.5     # m

# Force limits
F_max = 100.0  # Maximum force magnitude for sampling

# Problem dimensions for quadruped force control


##Parameters for MLP model
# Default states
BaseRollPitchYaw = (0.0, 0.0, 0.0)
AngularVelocityBodyFrame = (0.0, 0.0, 0.0)
ComVelocityBodyFrame = (0.0, 0.0, 0.0)
FootContacts = (True, True, True, True)
slope_estimate = (0.0, 0.0, 0.0)
RotationBodyWrtWorld = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

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
controller = ForceStanceLegController(
    desired_speed=desired_speed,
    desired_twisting_speed=desired_twisting_speed,
    desired_body_height=desired_body_height,
    body_mass=body_mass,
    body_inertia=body_inertia,
    num_legs=num_legs,
    friction_coeff=friction_coeff,
    timestep=timestep,
    horizon=horizon
)

# Get QP matrices
H, g, C, c = controller.getMatrices(
    BaseRollPitchYaw=BaseRollPitchYaw,
    AngularVelocityBodyFrame=AngularVelocityBodyFrame,
    ComVelocityBodyFrame=ComVelocityBodyFrame,
    FootPositionsInBodyFrame=FootPositionsInBodyFrame,
    FootContacts=FootContacts,
    slope_estimate=slope_estimate,
    RotationBodyWrtWorld=RotationBodyWrtWorld
)


# Store matrices
H = H                       # QP Hessian (3nk x 3nk)
g = g                       # Linear term (3nk)
C = C 
c = c                      # Constraint matrix (num_total_constraints x 3nk)

# Maximum Iterations
maxiter_projection = 20

nvar = H.shape[0]
C_torch = torch.from_numpy(C).float().to(device)
A_control = C_torch #torch.vstack((C_torch, -C_torch))


num_total_constraints = A_control.shape[0]

# Generate training and validation data
xi_samples, rng = sample_uniform_forces(42, F_max, num_batch, nvar)
xi_val, rng_val = sample_uniform_forces(43, F_max, num_batch, nvar)

# xi_samples = torch.zeros(num_batch, nvar)
# xi_val = torch.zeros(num_batch, nvar)
inp = xi_samples
inp_val = xi_val

# Using PyTorch Dataloader
train_dataset = ForceDataset(inp)
val_dataset = ForceDataset(inp_val)

train_loader = DataLoader(train_dataset, batch_size=num_batch, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=num_batch, shuffle=True, num_workers=0, drop_last=True)

# GRU handling
gru_input_size = 3 * num_total_constraints + 3 * nvar
gru_hidden_size = 512
gru_output_size = num_total_constraints + nvar

gru_context = CustomGRULayer(gru_input_size, gru_hidden_size, gru_output_size)

input_hidden_state_init = np.shape(inp)[1]
mid_hidden_state_init = 512
out_hidden_state_init = gru_hidden_size

gru_init = GRU_Hidden_State(input_hidden_state_init, mid_hidden_state_init, out_hidden_state_init)

# MLP setup
enc_inp_dim = np.shape(inp)[1] 
mlp_inp_dim = enc_inp_dim
hidden_dim = 1024
mlp_out_dim = 2*nvar + num_total_constraints  # xi_samples, lambda_samples, slack_variables

mlp = MLP(mlp_inp_dim, hidden_dim, mlp_out_dim)

# Create the quadruped model
model = MLPQuadrupedProjectionFilter(
    mlp=mlp,
    gru_context=gru_context, 
    gru_init=gru_init, 
    num_batch=num_batch,
    H=H, 
    g=g, 
    C=C, 
    c =c, 
    maxiter_projection=maxiter_projection,
    desired_speed=desired_speed,
    desired_twisting_speed=desired_twisting_speed,
    desired_body_height=desired_body_height,
    body_mass=body_mass,
    body_inertia=body_inertia,
    num_legs=num_legs,
    friction_coeff=friction_coeff,
    timestep=timestep,
    horizon=horizon).to(device)

print(f"Model type: {type(model)}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# Training
epochs = 500
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=6e-5)

losses = []
last_loss = torch.inf
avg_train_loss, avg_primal_loss, avg_fixed_point_loss, avg_projection_loss = [], [], [], []
avg_val_loss = []

for epoch in range(epochs):
    
    # Train Loop
    model.train()
    losses_train, primal_losses, fixed_point_losses, projection_losses = [], [], [], []
    
    for (inp) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        
        # Input and Output 
        inp = inp.to(device)
        
        # Forward pass through quadruped model
        xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = model(inp)
        
        # Compute loss
        primal_loss, fixed_point_loss, projection_loss, loss = model.mlp_loss(
            avg_res_primal, avg_res_fixed_point, inp, xi_projected)

        optimizer.zero_grad()
        loss.backward()
        
        # Optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses_train.append(loss.detach().cpu().numpy()) 
        primal_losses.append(primal_loss.detach().cpu().numpy())
        fixed_point_losses.append(fixed_point_loss.detach().cpu().numpy())
        projection_losses.append(projection_loss.detach().cpu().numpy())
        
    # Validation every 2 epochs
    if epoch % 2 == 0:
        model.eval()
        val_losses = []

        with torch.no_grad():
            for (inp_val) in tqdm(val_loader, desc="Validation"):
                inp_val = inp_val.to(device)

                xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = model(inp_val)

                _, _, _, val_loss = model.mlp_loss(
                    avg_res_primal, avg_res_fixed_point, inp_val, xi_projected
                )

                val_losses.append(val_loss.detach().cpu().numpy())

    # Print progress every 2 epochs
    if epoch % 2 == 0:    
        print(f"Epoch: {epoch + 1}, Train Loss: {np.average(losses_train):.4f}, "
              f"Primal Loss: {np.average(primal_losses):.4f}, "
              f"Fixed-Point Loss: {np.average(fixed_point_losses):.4f}, "
              f"Projection Loss: {np.average(projection_losses):.4f}")
        
        if len(val_losses) > 0:
            print(f"Validation Loss: {np.average(val_losses):.4f}")

    # Save best model
    os.makedirs("./training_weights", exist_ok=True)
    if loss <= last_loss:
        torch.save(model.state_dict(), f"./training_weights/mlp_learned_quadruped_gru.pth")
        last_loss = loss

    # Store metrics
    avg_train_loss.append(np.average(losses_train))
    avg_primal_loss.append(np.average(primal_losses))
    avg_projection_loss.append(np.average(projection_losses))
    avg_fixed_point_loss.append(np.average(fixed_point_losses))
    
    if len(val_losses) > 0:
        avg_val_loss.append(np.average(val_losses))
    else:
        avg_val_loss.append(avg_val_loss[-1] if avg_val_loss else 0.0)

print("Training completed!")

