import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stance_leg_controller import ForceStanceLegController


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

#LSTM layer class
class CustomLSTMLayer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		"""
		Custom LSTM layer with output transformation for single sequence element
		
		Args:
			input_size (int): Size of input features
			hidden_size (int): Size of hidden state in LSTM
			output_size (int): Size of the output after transformation
		"""
		super(CustomLSTMLayer, self).__init__()
		
        #In LSTM, hidden state = long term memory, cell state = short term memory
		# LSTM cell for processing input
		self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
		
		# Transformation layer to generate output from hidden state
		self.output_transform = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Linear(hidden_size, output_size)
		)
		
		self.hidden_size = hidden_size
		
	def forward(self, x, h_t, c_t):
		"""
		Forward pass through the LSTM layer
		
		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, input_size]
			h_t (torch.Tensor): Hidden state tensor of shape [batch_size, hidden_size]
			c_t (torch.Tensor): Cell state tensor of shape [batch_size, hidden_size]
		
		Returns:
			tuple: (output, hidden_state, cell_state)
				- output: tensor of shape [batch_size, output_size]
				- hidden_state: tensor of shape [batch_size, hidden_size]
				- cell_state: tensor of shape [batch_size, hidden_size]
		"""
		# Update hidden state and cell state with LSTM cell
		h_t, c_t = self.lstm_cell(x, (h_t, c_t))
		
		# Transform hidden state to generate output
		g_t = self.output_transform(h_t)
		
		return g_t, h_t, c_t


class LSTM_Hidden_State(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(LSTM_Hidden_State, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),

			nn.Linear(hidden_dim, hidden_dim),
			#nn.BatchNorm1d(hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, out_dim * 2),  # Output both h_0 and c_0
		)
		
		self.hidden_dim = out_dim
	
	def forward(self, x):
		out = self.mlp(x)
		# Split output into hidden state and cell state
		h_0 = out[:, :self.hidden_dim]
		c_0 = out[:, self.hidden_dim:]
		return h_0, c_0


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
    
    def __init__(self, mlp, rnn_context, rnn_init, num_batch,
                 maxiter_projection,
                 BaseRollPitchYaw,AngularVelocityBodyFrame,
                 ComVelocityBodyFrame,FootPositionsInBodyFrame,
                 FootContacts,slope_estimate,
                 RotationBodyWrtWorld, 
                 desired_body_height, body_mass,
                 body_inertia,
                 num_legs, friction_coeff, timestep, horizon, rnn):
        super(MLPQuadrupedProjectionFilter, self).__init__()
        
        # MLP Model
        self.mlp = mlp

        if rnn == 'GRU':
            self.gru_context = rnn_context
            self.gru_init = rnn_init
        elif rnn == 'LSTM':
            self.lstm_context = rnn_context
            self.lstm_init = rnn_init
        
        # Problem dimensions
        self.num_batch = num_batch
        self.maxiter_projection = maxiter_projection


        
        # Quadruped parameters
        # self.desired_speed = desired_speed
        # self.desired_twisting_speed = desired_twisting_speed
        self.desired_body_height = desired_body_height
        self.body_mass = body_mass
        self.body_inertia = body_inertia
        self.num_legs = num_legs
        self.friction_coeff = friction_coeff
        self.timestep = timestep
        self.horizon = horizon
        
        self.BaseRollPitchYaw = BaseRollPitchYaw
        self.AngularVelocityBodyFrame = AngularVelocityBodyFrame
        self.ComVelocityBodyFrame = ComVelocityBodyFrame
        self.FootPositionsInBodyFrame = FootPositionsInBodyFrame
        self.FootContacts = FootContacts
        self.slope_estimate = slope_estimate
        self.RotationBodyWrtWorld = RotationBodyWrtWorld

        # # QP matrices (converted from JAX to PyTorch)
        # self.H = torch.tensor(self.H, dtype=torch.float32, device=device)
        # self.g = torch.tensor(self.g, dtype=torch.float32, device=device)
        # self.C = torch.tensor(self.C, dtype=torch.float32, device=device)
        # self.c = torch.tensor(self.c, dtype=torch.float32, device=device)


 
        
        # Problem dimensions
        #Specific for Quadruped
        self.nvar = 3*self.num_legs*self.horizon
        self.num_constraints = 2*self.nvar  
        
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
        #A_control = self.C #torch.vstack((self.C, -self.C))
        
        # Number of constraints
        #self.num_constraints = A_control.shape[0]
        
        # print(f"Problem dimensions:")
        # print(f"H matrix shape: {self.H.shape}")
        # print(f"g vector shape: {self.g.shape}")
        # print(f"C matrix shape: {self.C.shape}")
        # print(f"Number of variables: {self.nvar}")
        # print(f"Number of constraints: {self.num_constraints}")

        # # A_eq_single_horizon: block-diagonal of identity matrices
        # self.A_eq_single_horizon = torch.tile(torch.asarray([0.0,0.0,1.0]),(1,self.num_legs)).to(device) # shape: (3 * num_legs, 3)
        
        # self.I_horizon = torch.eye(self.horizon).to(device)

        # # A_eq: kron product with identity across horizon
        # self.A_eq = torch.kron(self.I_horizon, self.A_eq_single_horizon).to(device) # shape: (3 * num_legs * horizon, 3 * horizon)


        # print("self.A_eq.shape", self.A_eq.shape)

        # # b_eq_single_horizon: gravity force applied per batch
        # self.b_eq_single_horizon = torch.tile(
        #     torch.tensor([self.body_mass * 9.81]), (self.num_batch, 1)
        # )  # shape: (num_batch, 3)

        # self.b_eq_single_horizon = self.b_eq_single_horizon.to(device)

        # # b_eq: repeat across horizon
        # self.b_eq = self.b_eq_single_horizon.repeat(1, self.horizon)  # shape: (num_batch, 3 * horizon)

        # self.b_eq = self.b_eq.to(device)

        # print("self.b_eq.shape", self.b_eq.shape)

                
    def compute_feasible_control(self, s, lamda, desired_speed, desired_twisting_speed):
        """Compute feasible control following JAX approach exactly"""
        
        desired_speed_tuple = tuple(desired_speed[0].tolist())  # → (0.5, -0.2)
        # desired_twisting_speed_tuple = tuple(desired_twisting_speed[0].tolist())  # → (0.0, 0.0)
        # print("desired_speed_tuple", desired_speed_tuple)
        # print("desired_twisting_speed_tuple", desired_twisting_speed_tuple)

        desired_twisting_speed_scalar = desired_twisting_speed[0].item()

        # print("desired_twisting_speed_scalar", desired_twisting_speed_scalar)
        
        controller = ForceStanceLegController(
            desired_speed=desired_speed_tuple,
            desired_twisting_speed=desired_twisting_speed_scalar,
            desired_body_height=self.desired_body_height,
            body_mass=self.body_mass,
            body_inertia=self.body_inertia,
            num_legs=self.num_legs,
            friction_coeff=self.friction_coeff,
            timestep=self.timestep,
            horizon=self.horizon
        )

        # Get QP matrices
        H, g, C, c, _ = controller.getMatrices(
            BaseRollPitchYaw=self.BaseRollPitchYaw,
            AngularVelocityBodyFrame=self.AngularVelocityBodyFrame,
            ComVelocityBodyFrame=self.ComVelocityBodyFrame,
            FootPositionsInBodyFrame=self.FootPositionsInBodyFrame,
            FootContacts=self.FootContacts,
            slope_estimate=self.slope_estimate,
            RotationBodyWrtWorld=self.RotationBodyWrtWorld,
            Training = True
        )
        H = torch.tensor(H, dtype=torch.float32, device=device)
        g = torch.tensor(g, dtype=torch.float32, device=device)
        c = torch.tensor(c, dtype=torch.float32, device=device)
        C = torch.tensor(C, dtype=torch.float32, device=device)

        A_control = C

        reg_param = 1e-4

        # Cost matrix 

        cost = (H + self.rho_ineq * torch.matmul(A_control.T, A_control) + reg_param * torch.eye(self.nvar, device=device)).to(device)
        
        # print(f"Regularized condition number: {torch.linalg.cond(cost)}")

        cost_matrix = cost
        
        #b_eq = self.b_eq

        b_control = c
        
        # Augmented bounds with slack variables
        b_control_aug = b_control - s

        # Linear cost term 
        lincost = (-lamda + g - 
                  self.rho_ineq * torch.matmul(A_control.T, b_control_aug.T).T)
        
        Q_inv = torch.linalg.pinv(cost_matrix)
        
        # Solve KKT system
        #rhs = torch.hstack((-lincost, b_eq))
        rhs = -lincost
        sol = torch.matmul(Q_inv, rhs.T).T
        # # Solve KKT system
        # rhs = 
        # sol = torch.linalg.solve(cost_mat, (-lincost, b_eq).T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following JAX)
        s = torch.maximum(
            torch.zeros((self.num_batch, self.num_constraints), device=device),
            -torch.matmul(A_control, xi_projected.T).T + b_control
        )
        
        # Compute residual (following JAX)
        res_vec = torch.matmul(A_control, xi_projected.T).T - b_control + s
        res_norm = torch.linalg.norm(res_vec, dim=1)
        
        # Update Lagrange multipliers (following JAX)
        lamda = lamda - self.rho_ineq * torch.matmul(A_control.T, res_vec.T).T

        qp_cost = 0.5 * xi_projected @ H @ xi_projected.T + xi_projected @ g

        qp_cost_norm = torch.linalg.norm(qp_cost, dim=1)

        # # Debug prints
        # print(f"H matrix condition number: {torch.linalg.cond(H)}")
        # print(f"xi_projected range: [{xi_projected.min():.4f}, {xi_projected.max():.4f}]")
        # print(f"H diagonal range: [{H.diag().min():.4f}, {H.diag().max():.4f}]")
        # print(f"g range: [{g.min():.4f}, {g.max():.4f}]")
        
        return xi_projected, s, res_norm, lamda, qp_cost_norm, H, g

    def compute_projection_gru(self, lamda_init_nn_output, s_init_nn_output, desired_speed, desired_twisting_speed, h_0):
        """Project sampled trajectories following JAX approach"""
        desired_speed_tuple = tuple(desired_speed[0].tolist())  # → (0.5, -0.2)
        # desired_twisting_speed_tuple = tuple(desired_twisting_speed[0].tolist())  # → (0.0, 0.0)
        # print("desired_speed_tuple", desired_speed_tuple)
        # print("desired_twisting_speed_tuple", desired_twisting_speed_tuple)

        desired_twisting_speed_scalar = desired_twisting_speed[0].item()

        # print("desired_twisting_speed_scalar", desired_twisting_speed_scalar)
        
        controller = ForceStanceLegController(
            desired_speed=desired_speed_tuple,
            desired_twisting_speed=desired_twisting_speed_scalar,
            desired_body_height=self.desired_body_height,
            body_mass=self.body_mass,
            body_inertia=self.body_inertia,
            num_legs=self.num_legs,
            friction_coeff=self.friction_coeff,
            timestep=self.timestep,
            horizon=self.horizon
        )

        # Get QP matrices
        H, g, C, c, _ = controller.getMatrices(
            BaseRollPitchYaw=self.BaseRollPitchYaw,
            AngularVelocityBodyFrame=self.AngularVelocityBodyFrame,
            ComVelocityBodyFrame=self.ComVelocityBodyFrame,
            FootPositionsInBodyFrame=self.FootPositionsInBodyFrame,
            FootContacts=self.FootContacts,
            slope_estimate=self.slope_estimate,
            RotationBodyWrtWorld=self.RotationBodyWrtWorld,
            Training = True
        )
        H = torch.tensor(H, dtype=torch.float32, device=device)
        g = torch.tensor(g, dtype=torch.float32, device=device)
        c = torch.tensor(c, dtype=torch.float32, device=device)
        C = torch.tensor(C, dtype=torch.float32, device=device)

        A_control = C

        reg_param = 1e-4

        # Cost matrix 

        #cost = (H + self.rho_ineq * torch.matmul(A_control.T, A_control) + reg_param * torch.eye(self.nvar, device=device)).to(device)
        
        cost = H
        # print(f"Regularized condition number: {torch.linalg.cond(cost)}")

        cost_matrix = cost + 1e-4*torch.eye(self.nvar, device=device)
        
        # #b_eq = self.b_eq

        # b_control = c
        
        # # Augmented bounds with slack variables
        # b_control_aug = b_control - s

        # Linear cost term 
        # lincost = (-lamda + g - self.rho_ineq * torch.matmul(A_control.T, b_control_aug.T).T)
        
        lincost = ( -0.0*lamda_init_nn_output + g )
        
        Q_inv = torch.linalg.pinv(cost_matrix)
        
        # Solve KKT system
        #rhs = torch.hstack((-lincost, b_eq))
        rhs = -lincost
        sol = torch.matmul(Q_inv, rhs.T).T
        # # Solve KKT system
        # rhs = 
        # sol = torch.linalg.solve(cost_mat, (-lincost, b_eq).T).T
        
        # Extract primal solution
        xi_projected_ = sol[:, 0:self.nvar]
        
        # Update slack variables (following JAX)
        s_ = torch.maximum(
            torch.zeros((self.num_batch, self.num_constraints), device=device),
            -torch.matmul(A_control, xi_projected_.T).T + c
        )
        
        
        # Initialize variables
        
        lamda_init = lamda_init_nn_output
        
        # Initialize slack variables
        s_init = 0.00*s_init_nn_output+1.0*s_ 



        
        # Initialize tracking variables
        lamda = lamda_init
        s = s_init
        h = h_0
        
        primal_residuals = []
        fixed_point_residuals = []
        qp_cost_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter_projection):
            #xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda, qp_cost_norm, H, g = self.compute_feasible_control(
                s, lamda, desired_speed, desired_twisting_speed)
            
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
            
            qp_cost_residual = qp_cost_norm
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)
            qp_cost_residuals.append(qp_cost_residual)

        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        qp_cost_residuals = torch.stack(qp_cost_residuals) #+ (s_ - s_init)**2
        
        return xi_projected, primal_residuals, fixed_point_residuals, qp_cost_residuals
    
    def compute_projection_lstm(self, lamda_init_nn_output, s_init_nn_output, desired_speed, desired_twisting_speed, h_0, c_0):
        """Project sampled trajectories following JAX approach"""
        
        # Initialize variables
        
        lamda_init = lamda_init_nn_output
        
        # Initialize slack variables
        s_init = s_init_nn_output 
        
        # Initialize tracking variables
        
        lamda = lamda_init
        s = s_init
        h = h_0
        c= c_0
        
        primal_residuals = []
        fixed_point_residuals = []
        
        # Projection iterations
        for idx in range(self.maxiter_projection):
            #xi_projected_prev = xi_projected.clone()
            lamda_prev = lamda.clone()
            s_prev = s.clone()
            
            # Perform projection step
            xi_projected, s, res_norm, lamda, qp_cost_norm = self.compute_feasible_control(
                s, lamda, desired_speed, desired_twisting_speed)
            
            # Perform GRU acceleration after fixed-point iteration
            r_1 = torch.hstack((s_prev, lamda_prev))
            r_2 = torch.hstack((s, lamda))
            r = torch.hstack((r_1, r_2, r_2 - r_1))

            lstm_output, h, c = self.lstm_context(r, h, c)

            s_delta = lstm_output[:, 0: self.num_constraints]
            lamda_delta = lstm_output[:, self.num_constraints: self.num_constraints+self.nvar]

            lamda = lamda + lamda_delta 
            s = s + s_delta
            s = torch.maximum(torch.zeros((self.num_batch, self.num_constraints), device=device), s)

            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (torch.linalg.norm(lamda_prev - lamda, dim=1) + 
                                  torch.linalg.norm(s_prev - s, dim=1))
            
            qp_cost_residual = qp_cost_norm
            
            primal_residuals.append(primal_residual)
            fixed_point_residuals.append(fixed_point_residual)
            qp_cost_residuals.append(qp_cost_residual)

        # Stack residuals
        primal_residuals = torch.stack(primal_residuals)
        fixed_point_residuals = torch.stack(fixed_point_residuals)
        qp_cost_residuals = torch.stack(qp_cost_residuals)
        
        return xi_projected, primal_residuals, fixed_point_residuals, qp_cost_residuals

    def decoder_function(self, inp_norm, desired_speed, desired_twisting_speed, rnn):
        """Decoder function to compute initials from normalized input"""
        # Get neural network output
        neural_output_batch = self.mlp(inp_norm)
        
        # Structure neural output for quadruped force control
        xi_projected_output_nn = neural_output_batch[:, :self.nvar]
        lamda_init_nn_output = neural_output_batch[:, self.nvar: 2*self.nvar]
        s_init_nn_output = neural_output_batch[:, 2*self.nvar: 2*self.nvar + self.num_constraints]

        s_init_nn_output = torch.maximum(torch.zeros((self.num_batch, self.num_constraints), device=device), s_init_nn_output)

        if rnn == "GRU":
            h_0 = self.gru_init(inp_norm)
            xi_projected, primal_residuals, fixed_point_residuals, qp_cost_residuals = self.compute_projection_gru(
            lamda_init_nn_output, s_init_nn_output, desired_speed, desired_twisting_speed, h_0)
        
        elif rnn == "LSTM":
            h_0, c_0 = self.lstm_init(inp_norm)
            xi_projected, primal_residuals, fixed_point_residuals, qp_cost_residuals = self.compute_projection_lstm(
            lamda_init_nn_output, s_init_nn_output, desired_speed, desired_twisting_speed, h_0, c_0)

        
        # Compute average residuals
        avg_res_primal = torch.mean(primal_residuals, dim=0)
        avg_res_fixed_point = torch.mean(fixed_point_residuals, dim=0)
        avg_res_qp_cost = torch.mean(qp_cost_residuals, dim=0)
        
        return xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost ,primal_residuals, fixed_point_residuals, qp_cost_residuals

    def mlp_loss(self, avg_res_primal, avg_res_fixed_point, avg_res_qp_cost_loss):


        """Compute loss for optimization"""
        # Component losses
        primal_loss = 0.5 * torch.mean(avg_res_primal)
        fixed_point_loss = 0.5 * torch.mean(avg_res_fixed_point)
        qp_cost_loss = 0.5 * torch.mean(avg_res_qp_cost_loss)
        # projection_loss = 0.5 * self.rcl_loss(xi_projected_output_nn, inp_norm)
        #projection_loss = 0.0 * torch.mean(avg_res_fixed_point)
        # Total loss
        loss = primal_loss + 1.0*fixed_point_loss + 0.8*qp_cost_loss

        return primal_loss, fixed_point_loss, qp_cost_loss, loss

    def forward(self, input_nn, desired_speed, desired_twisting_speed, rnn):
        """Forward pass through the model"""
        # # Normalize input
        # inp_mean = input_nn.mean()
        # inp_std = input_nn.std()
        # inp_norm = (input_nn - inp_mean) / (inp_std + 1e-8)

        inp_median_ = torch.median(input_nn, dim=0).values
        inp_q1 = torch.quantile(input_nn, 0.25, axis=0)
        inp_q3 = torch.quantile(input_nn, 0.75, axis=0)
        inp_iqr_ = inp_q3 - inp_q1
        # Handle constant features
        inp_iqr_ = torch.where(inp_iqr_ == 0, torch.tensor(1.0), inp_iqr_)
        inp_norm = (input_nn - inp_median_) / inp_iqr_

        # Decode input to get control
        # xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost, res_primal_history, res_fixed_point_history, res_qp_cost_history = self.decoder_function(
        #     inp_norm, desired_speed, desired_twisting_speed, rnn)
        
        (xi_projected,
        avg_res_fixed_point,
        avg_res_primal,
        avg_res_qp_cost,
        res_primal_history,
        res_fixed_point_history,
        res_qp_cost_history) = self.decoder_function(
        inp_norm,
        desired_speed,
        desired_twisting_speed,
        rnn)
        
        
            
        return xi_projected, avg_res_fixed_point, avg_res_primal, avg_res_qp_cost, res_primal_history, res_fixed_point_history, res_qp_cost_history
