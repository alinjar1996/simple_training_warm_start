import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GRU layer class
class CustomGRULayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Custom GRU layer with output transformation for single sequence element
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state in GRU
            output_size (int): Size of the output after transformation
        """
        super(CustomGRULayer, self).__init__()
        
        # GRU cell for processing input
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        
        # Transformation layer to generate output from hidden state
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.hidden_size = hidden_size
        
    def forward(self, x, h_t):
        """
        Forward pass through the GRU layer
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size]
            h_t (torch.Tensor): Hidden state tensor of shape [batch_size, hidden_size]
        
        Returns:
            tuple: (output, hidden_state)
                - output: tensor of shape [batch_size, output_size]
                - hidden_state: tensor of shape [batch_size, hidden_size]
        """
        # Update hidden state with GRU cell
        h_t = self.gru_cell(x, h_t)
        
        # Transform hidden state to generate output
        g_t = self.output_transform(h_t)
        
        return g_t, h_t


class GRU_Hidden_State(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(GRU_Hidden_State, self).__init__()
        
        # MC Dropout Architecture
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        out = self.mlp(x)
        return out


# MLP class
class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability
            
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout with 20% probability
            
            nn.Linear(256, out_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)


class MLPQuadrupedProjectionFilter(nn.Module):
    
    def __init__(self, mlp, gru_context, gru_init, num_batch, 
                 H, g, C, c, maxiter_projection,
                 desired_speed, desired_twisting_speed,
                 desired_body_height, body_mass,
                 body_inertia,
                 num_legs, friction_coeff, timestep, horizon):
        super(MLPQuadrupedProjectionFilter, self).__init__()
        
        # MLP Model
        self.mlp = mlp

        # GRU Model
        self.gru_context = gru_context
        self.gru_init = gru_init
        
        # Problem dimensions
        self.num_batch = num_batch
        self.maxiter_projection = maxiter_projection
        
        # Quadruped parameters
        self.desired_speed = desired_speed
        self.desired_twisting_speed = desired_twisting_speed
        self.desired_body_height = desired_body_height
        self.body_mass = body_mass
        self.body_inertia = body_inertia
        self.num_legs = num_legs
        self.friction_coeff = friction_coeff
        self.timestep = timestep
        self.horizon = horizon
        
        # QP matrices (converted from JAX to PyTorch)
        self.H = torch.tensor(H, dtype=torch.float32, device=device)
        self.g = torch.tensor(g, dtype=torch.float32, device=device)
        self.C = torch.tensor(C, dtype=torch.float32, device=device)
        self.c = torch.tensor(c, dtype=torch.float32, device=device)
 
        
        # Problem dimensions
        self.nvar = self.H.shape[0]
        
        # Projection parameters
        self.A_projection = torch.eye(self.nvar, device=device)
        self.rho_ineq = 1.0
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
        
        # Loss function
        self.rcl_loss = nn.MSELoss()

    def setup_optimization_matrices(self):
        """Setup matrices following JAX approach"""
        
        # Combined control matrix (stack C and -C)
        self.A_control = self.C #torch.vstack((self.C, -self.C))
        
        # Number of constraints
        self.num_constraints = self.A_control.shape[0]
        
        print(f"Problem dimensions:")
        print(f"H matrix shape: {self.H.shape}")
        print(f"g vector shape: {self.g.shape}")
        print(f"C matrix shape: {self.C.shape}")
        print(f"Number of variables: {self.nvar}")
        print(f"Number of constraints: {self.num_constraints}")

        # A_eq_single_horizon: block-diagonal of identity matrices
        self.A_eq_single_horizon = torch.tile(torch.eye(3),(1,self.num_legs)).to(device) # shape: (3 * num_legs, 3)
        
        self.I_horizon = torch.eye(self.horizon).to(device)

        # A_eq: kron product with identity across horizon
        self.A_eq = torch.kron(self.I_horizon, self.A_eq_single_horizon).to(device) # shape: (3 * num_legs * horizon, 3 * horizon)


        print("self.A_eq.shape", self.A_eq.shape)

        # b_eq_single_horizon: gravity force applied per batch
        self.b_eq_single_horizon = torch.tile(
            torch.tensor([[0.0, 0.0, self.body_mass * 9.81]]), (self.num_batch, 1)
        )  # shape: (num_batch, 3)

        self.b_eq_single_horizon = self.b_eq_single_horizon.to(device)

        # b_eq: repeat across horizon
        self.b_eq = self.b_eq_single_horizon.repeat(1, self.horizon)  # shape: (num_batch, 3 * horizon)

        self.b_eq = self.b_eq.to(device)

        print("self.b_eq.shape", self.b_eq.shape)

                
        self.cost = (self.H + self.rho_ineq * torch.matmul(self.A_control.T, self.A_control))

        self.cost = self.cost.to(device)

        print("self.cost.shape", self.cost.shape)
        
        self.cost_matrix =torch.vstack((
            torch.hstack((self.cost, self.A_eq.T)),
            torch.hstack((self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0])).to(device))),
        ))

        self.cost_matrix = self.cost_matrix.to(device)



    def compute_feasible_control(self, s, xi_projected, lamda):
        """Compute feasible control following JAX approach exactly"""
        
        b_eq = self.b_eq

        b_control = self.c
        
        # Augmented bounds with slack variables
        b_control_aug = b_control - s

        # Cost matrix 
        
        cost_mat = self.cost_matrix
        # Linear cost term (following JAX)
        lincost = (-lamda + self.g - 
                  self.rho_ineq * torch.matmul(self.A_control.T, b_control_aug.T).T)
        
        self.Q_inv = torch.linalg.inv(cost_mat)
        
        # Solve KKT system
        rhs = torch.hstack((-lincost, b_eq))
        sol = torch.matmul(self.Q_inv, rhs.T).T
        # # Solve KKT system
        # rhs = 
        # sol = torch.linalg.solve(cost_mat, (-lincost, b_eq).T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following JAX)
        s = torch.maximum(
            torch.zeros((self.num_batch, self.num_constraints), device=device),
            -torch.matmul(self.A_control, xi_projected.T).T + b_control
        )
        
        # Compute residual (following JAX)
        res_vec = torch.matmul(self.A_control, xi_projected.T).T - b_control + s
        res_norm = torch.linalg.norm(res_vec, dim=1)
        
        # Update Lagrange multipliers (following JAX)
        lamda = lamda - self.rho_ineq * torch.matmul(self.A_control.T, res_vec.T).T
        
        return xi_projected, s, res_norm, lamda

    def compute_projection(self, xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, h_0):
        """Project sampled trajectories following JAX approach"""
        
        # Initialize variables
        xi_projected_init = xi_projected_output_nn
        lamda_init = lamda_init_nn_output
        
        # Initialize slack variables
        s_init = s_init_nn_output 
        
        # Initialize tracking variables
        xi_projected = xi_projected_init
        lamda = lamda_init
        s = s_init
        h = h_0
        
        primal_residuals = []
        fixed_point_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter_projection):
            xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda = self.compute_feasible_control(
                s, xi_projected, lamda)
            
            # Perform GRU acceleration after fixed-point iteration
            r_1 = torch.hstack((s_prev, lamda_prev))
            r_2 = torch.hstack((s, lamda))
            r = torch.hstack((r_1, r_2, r_2 - r_1))

            gru_output, h = self.gru_context(r, h)

            s_delta = gru_output[:, 0: self.num_constraints]
            lamda_delta = gru_output[:, self.num_constraints: self.num_constraints + self.nvar]

            lamda = lamda + lamda_delta 
            s = s + s_delta
            s = torch.maximum(torch.zeros((self.num_batch, self.num_constraints), device=device), s)

            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (torch.linalg.norm(lamda_prev - lamda, dim=1) + 
                                  torch.linalg.norm(s_prev - s, dim=1))
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)

        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        
        return xi_projected, primal_residuals, fixed_point_residuals

    def decoder_function(self, inp_norm):
        """Decoder function to compute initials from normalized input"""
        # Get neural network output
        neural_output_batch = self.mlp(inp_norm)
        
        # Structure neural output for quadruped force control
        xi_projected_output_nn = neural_output_batch[:, :self.nvar]
        lamda_init_nn_output = neural_output_batch[:, self.nvar: 2*self.nvar]
        s_init_nn_output = neural_output_batch[:, 2*self.nvar: 2*self.nvar + self.num_constraints]

        s_init_nn_output = torch.maximum(torch.zeros((self.num_batch, self.num_constraints), device=device), s_init_nn_output)

        h_0 = self.gru_init(inp_norm)

        # Run projection
        xi_projected, primal_residuals, fixed_point_residuals = self.compute_projection(
            xi_projected_output_nn, lamda_init_nn_output, s_init_nn_output, h_0)
        
        # Compute average residuals
        avg_res_primal = torch.mean(primal_residuals, dim=0)
        avg_res_fixed_point = torch.mean(fixed_point_residuals, dim=0)
        
        return xi_projected, avg_res_fixed_point, avg_res_primal, primal_residuals, fixed_point_residuals

    def mlp_loss(self, avg_res_primal, avg_res_fixed_point, xi_samples_input_nn, xi_projected_output_nn):
        # Normalize input
        inp_mean = xi_samples_input_nn.mean()
        inp_std = xi_samples_input_nn.std()
        inp_norm = (xi_samples_input_nn - inp_mean) / (inp_std + 1e-8)

        """Compute loss for optimization"""
        # Component losses
        primal_loss = 0.5 * torch.mean(avg_res_primal)
        fixed_point_loss = 0.5 * torch.mean(avg_res_fixed_point)
        projection_loss = 0.5 * self.rcl_loss(xi_projected_output_nn, inp_norm)

        # Total loss
        loss = primal_loss + fixed_point_loss + 0.1 * projection_loss

        return primal_loss, fixed_point_loss, projection_loss, loss

    def forward(self, xi_samples_input_nn):
        """Forward pass through the model"""
        # Normalize input
        inp_mean = xi_samples_input_nn.mean()
        inp_std = xi_samples_input_nn.std()
        inp_norm = (xi_samples_input_nn - inp_mean) / (inp_std + 1e-8)

        # Decode input to get control
        xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = self.decoder_function(
            inp_norm)
        
            
        return xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history
