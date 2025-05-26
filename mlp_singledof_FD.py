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


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, out_dim),
        )
    
    def forward(self, x):
        return self.mlp(x)


class MLPProjectionFilter(nn.Module):
    
    def __init__(self, mlp = MLP, num_batch = 1000, 
                 num_dof=1, num_steps=50, timestep=0.05, 
                 v_max=1.0, a_max=2.0, j_max=5.0, 
                 p_max=180.0*np.pi/180.0, theta_init=0.0):
        super(MLPProjectionFilter, self).__init__()
        
        # MLP Model
        self.mlp = mlp
        
        # Problem dimensions
        self.num_dof = num_dof
        self.num_steps = num_steps
        self.nvar_single = num_steps  # For velocity trajectory
        self.nvar = num_dof * num_steps
        self.num_batch = num_batch
        self.t = timestep
        
        # Constraint parameters (matching JAX version)
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.p_max = p_max
        self.theta_init = theta_init
        
        # Boundary conditions
        self.v_start = torch.zeros(num_dof, device=device)
        self.v_goal = torch.zeros(num_dof, device=device)

        # Projection parameters
        self.A_projection = torch.eye(self.nvar, device=device)
        self.rho_ineq = 1.0
        self.rho_projection = 1.0
        
        # Setup finite difference matrices
        self.setup_finite_difference_matrices()
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
        

        
        # Solver parameters
        self.maxiter = 15
        
        
        # Loss function
        self.rcl_loss = nn.MSELoss()

    def setup_finite_difference_matrices(self):
        """Create finite difference matrices matching JAX version"""
        # Create identity matrix for velocity (like P_vel)
        self.P_vel = torch.eye(self.num_steps, device=device)
        
        # Create acceleration matrix using diff
        vel_diff = torch.diff(self.P_vel, dim=0)
        self.P_acc = vel_diff / self.t
        
        # Create jerk matrix using diff of acceleration
        acc_diff = torch.diff(self.P_acc, dim=0)
        self.P_jerk = acc_diff / self.t
        
        # Create position matrix using cumsum
        self.P_pos = torch.cumsum(self.P_vel, dim=0) * self.t
        
        # Number of constraints
        self.num_acc = self.num_steps - 1
        self.num_jerk = self.num_acc - 1
        self.num_pos = self.num_steps

    def setup_optimization_matrices(self):
        """Setup matrices following JAX approach"""
        
        # For multi-DOF, we need to expand the single-DOF matrices
        # Velocity constraints: [P_vel; -P_vel] for each DOF
        A_vel_single = torch.vstack((self.P_vel, -self.P_vel))
        
        # Acceleration constraints: [P_acc; -P_acc] for each DOF
        A_acc_single = torch.vstack((self.P_acc, -self.P_acc))
        
        # Jerk constraints: [P_jerk; -P_jerk] for each DOF
        A_jerk_single = torch.vstack((self.P_jerk, -self.P_jerk))

        # Position constraints: [P_pos; -P_pos] for each DOF
        A_pos_single = torch.vstack((self.P_pos, -self.P_pos))
        
        # Expand for multiple DOFs using Kronecker product
        self.A_vel = torch.kron(torch.eye(self.num_dof, device=device), A_vel_single)
        self.A_acc = torch.kron(torch.eye(self.num_dof, device=device), A_acc_single)
        self.A_jerk = torch.kron(torch.eye(self.num_dof, device=device), A_jerk_single)
        self.A_pos = torch.kron(torch.eye(self.num_dof, device=device), A_pos_single)
        
        # Combined control matrix (like A_control in JAX)
        self.A_control = torch.vstack((self.A_vel, self.A_acc, self.A_jerk, self.A_pos))
        
        # Equality constraint matrix (boundary conditions)
        # Constrain first and last velocity for each DOF
        boundary_matrix = torch.tensor([
            [1.0] + [0.0] * (self.num_steps - 1),    # first timestep
            [0.0] * (self.num_steps - 1) + [1.0]     # last timestep
        ], device=device)
        
        self.A_eq = torch.kron(torch.eye(self.num_dof, device=device), boundary_matrix)
        
        # Constraint dimensions
        self.num_vel_constraints = 2 * self.num_steps * self.num_dof
        self.num_acc_constraints = 2 * self.num_acc * self.num_dof
        self.num_jerk_constraints = 2 * self.num_jerk * self.num_dof
        self.num_pos_constraints = 2 * self.num_pos * self.num_dof
        self.num_total_constraints = (self.num_vel_constraints + self.num_acc_constraints + 
                                    self.num_jerk_constraints + self.num_pos_constraints)
        
        # Compute inverse of Q matrix for KKT system
        self.Q_inv = self._get_Q_inv()

    def _get_Q_inv(self):
        """Compute inverse of Q matrix for the KKT system"""
        cost = (self.A_projection.T @ self.A_projection + 
                self.rho_ineq * self.A_control.T @ self.A_control)
        
        # KKT system matrix
        Q_top = torch.hstack((cost, self.A_eq.T))
        Q_bottom = torch.hstack((self.A_eq, torch.zeros((self.A_eq.shape[0], self.A_eq.shape[0]), device=device)))
        Q = torch.vstack((Q_top, Q_bottom))
        
        return torch.inverse(Q)

    def compute_boundary_vec(self):
        """Compute boundary condition vector"""
        # Stack start and goal velocities for all DOFs
        v_start_batch = self.v_start.unsqueeze(0).repeat(self.num_batch, 1)
        v_goal_batch = self.v_goal.unsqueeze(0).repeat(self.num_batch, 1)
        b_eq = torch.hstack([v_start_batch, v_goal_batch])
        return b_eq

    def compute_s_init(self, xi_projected):
        """Initialize slack variables following JAX approach"""
        # Create bounds vector
        b_vel = torch.hstack((
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device),
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device)
        ))
        
        b_acc = torch.hstack((
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device),
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device)
        ))
        
        b_jerk = torch.hstack((
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device),
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device)
        ))
        
        b_pos = torch.hstack((
            (self.p_max - self.theta_init) * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device),
            (self.p_max + self.theta_init) * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device)
        ))

        b_control = torch.hstack((b_vel, b_acc, b_jerk, b_pos))

        # Initialize slack variables
        s = torch.maximum(
            torch.zeros((self.num_batch, self.num_total_constraints), device=device),
            -torch.matmul(self.A_control, xi_projected.T).T + b_control
        )

        return s

    def compute_feasible_control(self, xi_samples, s, xi_projected, lamda):
        """Compute feasible control following JAX approach exactly"""
        b_vel = torch.hstack((
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device),
            self.v_max * torch.ones((self.num_batch, self.num_vel_constraints // 2), device=device)
        ))
        
        b_acc = torch.hstack((
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device),
            self.a_max * torch.ones((self.num_batch, self.num_acc_constraints // 2), device=device)
        ))
        
        b_jerk = torch.hstack((
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device),
            self.j_max * torch.ones((self.num_batch, self.num_jerk_constraints // 2), device=device)
        ))
        
        b_pos = torch.hstack((
            self.p_max * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device),
            self.p_max * torch.ones((self.num_batch, self.num_pos_constraints // 2), device=device)
        ))

        b_control = torch.hstack((b_vel, b_acc, b_jerk, b_pos))
        
        # Augmented bounds with slack variables
        b_control_aug = b_control - s
        
        # Boundary conditions
        b_eq = self.compute_boundary_vec()

        # print("b_control_aug", b_control_aug.shape)
        # print("self.A_control", self.A_control.shape)

        # Linear cost term (following JAX)
        lincost = (-lamda - 
                  torch.matmul(self.A_projection.T, xi_samples.T).T - 
                  self.rho_ineq * torch.matmul(self.A_control.T, b_control_aug.T).T)
        
        # Solve KKT system
        rhs = torch.hstack((-lincost, b_eq))
        sol = torch.matmul(self.Q_inv, rhs.T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following JAX)
        s = torch.maximum(
            torch.zeros((self.num_batch, self.num_total_constraints), device=device),
            -torch.matmul(self.A_control, xi_projected.T).T + b_control
        )
        
        # Compute residual (following JAX)
        res_vec = torch.matmul(self.A_control, xi_projected.T).T - b_control + s
        res_norm = torch.linalg.norm(res_vec, dim=1)
        
        # Update Lagrange multipliers (following JAX)
        lamda = lamda - self.rho_ineq * torch.matmul(self.A_control.T, res_vec.T).T
        
        return xi_projected, s, res_norm, lamda

    def compute_projection(self, xi_samples, xi_projected_output_nn, lamda_init_nn):
        """Project sampled trajectories following JAX approach"""
        
        # Initialize variables
        xi_projected_init = xi_projected_output_nn
        lamda_init = lamda_init_nn
        #lamda_init = torch.zeros((self.num_batch, self.nvar), device=device)
        
        # Initialize slack variables
        s_init = self.compute_s_init(xi_projected_init)
        
        # Initialize tracking variables
        xi_projected = xi_projected_init
        lamda = lamda_init
        s = s_init
        
        primal_residuals = []
        fixed_point_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter):
            xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda = self.compute_feasible_control(
                xi_samples, s, xi_projected, lamda)
            
            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (torch.linalg.norm(xi_projected_prev - xi_projected, dim=1) + 
                                  torch.linalg.norm(lamda_prev - lamda, dim=1) + 
                                  torch.linalg.norm(s_prev - s, dim=1))
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)

        
        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        
        return xi_projected, primal_residuals, fixed_point_residuals

    def decoder_function(self, inp_norm, xi_samples_input_nn):
        """Decoder function to compute control from normalized input"""
        # Get neural network output
        neural_output_batch = self.mlp(inp_norm)
        
        # For simplicity, use neural output as initial guess
        # In practice, you might want to structure this differently
        xi_projected_output_nn = neural_output_batch[:, :self.nvar]
        lambda_init_nn = neural_output_batch[:, self.nvar: 2*self.nvar]

        
        # Run projection
        xi_projected, primal_residuals, fixed_point_residuals = self.compute_projection(
            xi_samples_input_nn, xi_projected_output_nn, lambda_init_nn)
        
        # Compute average residuals
        avg_res_primal = torch.mean(primal_residuals, dim=0)
        avg_res_fixed_point = torch.mean(fixed_point_residuals, dim=0)
        
        return xi_projected, avg_res_fixed_point, avg_res_primal, primal_residuals, fixed_point_residuals

    def mlp_loss(self, avg_res_primal, avg_res_fixed_point, xi_samples_input_nn, xi_projected_output_nn):
        """Compute loss for optimization"""
        # Component losses
        primal_loss = 0.5 * torch.mean(avg_res_primal)
        fixed_point_loss = 0.5 * torch.mean(avg_res_fixed_point)
        projection_loss = self.rcl_loss(xi_projected_output_nn, xi_samples_input_nn)

        # Total loss
        loss = primal_loss + fixed_point_loss + 0.1 * projection_loss

        return primal_loss, fixed_point_loss, projection_loss, loss

    def forward(self, xi_samples_input_nn):
        """Forward pass through the model"""
        # Normalize input
        inp_mean = xi_samples_input_nn.mean()
        inp_std = xi_samples_input_nn.std()
        inp_norm = (xi_samples_input_nn - inp_mean) / inp_std

        # Decode input to get control
        xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history = self.decoder_function(
            inp_norm, xi_samples_input_nn)
            
        return xi_projected, avg_res_fixed_point, avg_res_primal, res_primal_history, res_fixed_point_history