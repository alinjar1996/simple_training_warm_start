import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.lax as lax

class TrajectoryProjector:
    def __init__(self, 
                 num_dof=1,            # Number of degrees of freedom (joints)
                 num_steps=50,         # Number of time steps in trajectory
                 num_batch=500,        # Batch size for parallel processing
                 timestep=0.05,        # Time step size in seconds
                 maxiter_projection=10,# Maximum iterations for projection
                 rho_projection=1.0,   # ADMM penalty parameter
                 rho_ineq=1.0,         # Penalty for inequality constraints
                 v_max=1.0,            # Maximum joint velocity
                 a_max=2.0,            # Maximum joint acceleration
                 j_max=5.0,              # Maximum joint jerk
                 p_max=180.0*np.pi/180.0, # Maximum joint displacement from theta_init    
                 theta_init = 0.0):        
        
        self.num_dof = num_dof
        self.num_steps = num_steps
        self.num_batch = num_batch
        self.nvar = num_dof * num_steps
        self.t = timestep
        self.maxiter_projection = maxiter_projection
        self.rho_projection = rho_projection
        self.rho_ineq = rho_ineq
        
        # Limits
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.p_max = p_max
        
        # Boundaries
        self.theta_init = theta_init
        self.v_start = jnp.zeros(num_dof)
        self.v_goal = jnp.zeros(num_dof)
        
        # Setup finite difference matrices ach
        self.setup_finite_difference_matrices()
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
        
        # Setup random key for JAX
        self.key = jax.random.PRNGKey(0)
    
    def setup_finite_difference_matrices(self):
        """Create finite difference matrices using jnp.diff """
        # Create identity matrix for velocity (like P_vel )
        self.P_vel = jnp.identity(self.num_steps)
        
        # Create acceleration matrix using diff ()
        self.P_acc = jnp.diff(self.P_vel, axis=0) / self.t
        
        # Create jerk matrix using diff of acceleration ()
        self.P_jerk = jnp.diff(self.P_acc, axis=0) / self.t
        
        self.P_pos = jnp.cumsum(self.P_vel, axis=0) * self.t
        #Create Position matrix using finite difference ()
        # Number of acceleration and jerk constraints
        
        self.num_acc = self.num_steps - 1
        self.num_jerk = self.num_acc - 1
        self.num_pos = self.num_steps

    def setup_optimization_matrices(self):
        """Setup matrices following  approach"""
        
        # For multi-DOF, we need to expand the single-DOF matrices
        # Velocity constraints: [P_vel; -P_vel] for each DOF
        A_vel_single = jnp.vstack((self.P_vel, -self.P_vel))
        
        # Acceleration constraints: [P_acc; -P_acc] for each DOF
        A_acc_single = jnp.vstack((self.P_acc, -self.P_acc))
        
        # Jerk constraints: [P_jerk; -P_jerk] for each DOF
        A_jerk_single = jnp.vstack((self.P_jerk, -self.P_jerk))

        #Position constraints: [P_pos; -P_pos] for each DOF
        A_pos_single = jnp.vstack((self.P_pos, -self.P_pos))
        
        # Expand for multiple DOFs using block diagonal structure
        self.A_vel = jnp.kron(jnp.identity(self.num_dof), A_vel_single)
        self.A_acc = jnp.kron(jnp.identity(self.num_dof), A_acc_single)
        self.A_jerk = jnp.kron(jnp.identity(self.num_dof), A_jerk_single)
        self.A_pos = jnp.kron(jnp.identity(self.num_dof), A_pos_single)
        
        # Combined control matrix (like A_control in )
        self.A_control = jnp.vstack((self.A_vel, self.A_acc, self.A_jerk, self.A_pos))
        
        # Equality constraint matrix (boundary conditions)
        # Constrain first and last velocity for each DOF
        self.A_eq = jnp.kron(
            jnp.eye(self.num_dof),
            jnp.array([[1.0] + [0.0] * (self.num_steps - 1),    # first timestep
                      [0.0] * (self.num_steps - 1) + [1.0]])   # last timestep
        )
        
        # Projection matrix
        self.A_projection = jnp.identity(self.nvar)
        
        # Constraint dimensions
        self.num_vel_constraints = 2 * self.num_steps * self.num_dof
        self.num_acc_constraints = 2 * self.num_acc * self.num_dof
        self.num_jerk_constraints = 2 * self.num_jerk * self.num_dof
        self.num_pos_constraints = 2 * self.num_pos * self.num_dof
        self.num_total_constraints = self.num_vel_constraints + self.num_acc_constraints + self.num_jerk_constraints + self.num_pos_constraints

    @partial(jax.jit, static_argnums=(0,))
    def compute_boundary_vec(self):
        """Compute boundary condition vector"""
        # Stack start and goal velocities for all DOFs
        v_start_batch = jnp.tile(self.v_start, (self.num_batch, 1))
        v_goal_batch = jnp.tile(self.v_goal, (self.num_batch, 1))
        b_eq = jnp.hstack([v_start_batch, v_goal_batch])
        return b_eq
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_s_init(self, xi_projected):
        """Initialize slack variables following  approach"""
        # Create bounds vector
        b_vel = jnp.hstack((
            self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // 2)),
            self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // 2))
        ))
        
        b_acc = jnp.hstack((
            self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // 2)),
            self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // 2))
        ))
        
        b_jerk = jnp.hstack((
            self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // 2)),
            self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // 2))
        ))
        
        b_pos = jnp.hstack((
            (self.p_max - self.theta_init) * jnp.ones((self.num_batch, self.num_pos_constraints // 2)),
            (self.p_max + self.theta_init) * jnp.ones((self.num_batch, self.num_pos_constraints // 2))
        ))

        b_control = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))

        # Initialize slack variables ()
        s = jnp.maximum(
            jnp.zeros((self.num_batch, self.num_total_constraints)),
            -jnp.dot(self.A_control, xi_projected.T).T + b_control
        )

        return s
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_feasible_control(self, xi_samples, s, xi_projected, lamda):
        """
        Compute feasible control following  approach exactly
        """
        b_vel = jnp.hstack((
            self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // 2)),
            self.v_max * jnp.ones((self.num_batch, self.num_vel_constraints // 2))
        ))
        
        b_acc = jnp.hstack((
            self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // 2)),
            self.a_max * jnp.ones((self.num_batch, self.num_acc_constraints // 2))
        ))
        
        b_jerk = jnp.hstack((
            self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // 2)),
            self.j_max * jnp.ones((self.num_batch, self.num_jerk_constraints // 2))
        ))
        
        b_pos = jnp.hstack((
            self.p_max * jnp.ones((self.num_batch, self.num_pos_constraints // 2)),
            self.p_max * jnp.ones((self.num_batch, self.num_pos_constraints // 2))
        ))

        b_control = jnp.hstack((b_vel, b_acc, b_jerk, b_pos))
        
        # Augmented bounds with slack variables
        b_control_aug = b_control - s
        
        # Boundary conditions
        b_eq = self.compute_boundary_vec()
        
        # Cost matrix 
        cost = (jnp.dot(self.A_projection.T, self.A_projection) + 
                self.rho_ineq * jnp.dot(self.A_control.T, self.A_control))
        
        # KKT system matrix ()
        cost_mat = jnp.vstack((
            jnp.hstack((cost, self.A_eq.T)),
            jnp.hstack((self.A_eq, jnp.zeros((jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0]))))
        ))
        
        # Linear cost term (following )
        lincost = (-lamda - 
                  jnp.dot(self.A_projection.T, xi_samples.T).T - 
                  self.rho_ineq * jnp.dot(self.A_control.T, b_control_aug.T).T)
        
        # Solve KKT system ()
        sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq)).T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following )
        s = jnp.maximum(
            jnp.zeros((self.num_batch, self.num_total_constraints)),
            -jnp.dot(self.A_control, xi_projected.T).T + b_control
        )
        
        # Compute residual (following )
        res_vec = jnp.dot(self.A_control, xi_projected.T).T - b_control + s
        res_norm = jnp.linalg.norm(res_vec, axis=1)
        
        # Update Lagrange multipliers (following )
        lamda = lamda - self.rho_ineq * jnp.dot(self.A_control.T, res_vec.T).T
        
        return xi_projected, s, res_norm, lamda
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_projection(self, xi_samples, xi_filtered):
        """Project sampled trajectories following  approach"""
        
        # Initialize variables
        xi_projected_init = xi_filtered
        lamda_init = jnp.zeros((self.num_batch, self.nvar))
        
        # Initialize slack variables
        s_init = self.compute_s_init(xi_projected_init)
        
        # Define scan function (following  structure)
        def lax_custom_projection(carry, idx):
            xi_projected, lamda, s = carry
            xi_projected_prev = xi_projected
            lamda_prev = lamda
            
            # Perform projection step
            xi_projected, s, res_norm, lamda = self.compute_feasible_control(
                xi_samples, s, xi_projected, lamda)
            
            # Compute residuals
            primal_residual = res_norm
            fixed_point_residual = (jnp.linalg.norm(xi_projected_prev - xi_projected, axis=1) + 
                                  jnp.linalg.norm(lamda_prev - lamda, axis=1))
            
            return (xi_projected, lamda, s), (primal_residual, fixed_point_residual)
        
        # Initialize carry
        carry_init = (xi_projected_init, lamda_init, s_init)
        
        # Run scan
        carry_final, res_tot = lax.scan(
            lax_custom_projection, 
            carry_init, 
            jnp.arange(self.maxiter_projection)
        )
        
        xi_projected, lamda, s = carry_final
        primal_residual, fixed_point_residual = res_tot
        
        return xi_projected, primal_residual, fixed_point_residual
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_uniform_trajectories(self, key):
        """Sample velocity trajectories uniformly between -v_max and v_max"""
        key, subkey = jax.random.split(key)
        xi_samples = jax.random.uniform(
            key, 
            shape=(self.num_batch, self.nvar), 
            minval=-self.v_max, 
            maxval=self.v_max
        )
        return xi_samples, key

def main():
    # Initialize the projector
    projector = TrajectoryProjector(
        num_dof=1,              # Multi-DOF example
        num_steps=4,
        num_batch=100,
        timestep=0.05,
        maxiter_projection=50,  # More iterations to see convergence
        v_max=1.0,
        a_max=2.0,
        j_max=3.0,
        p_max=180.0*np.pi/180.0,
        rho_ineq=1.0,
        rho_projection=1.0,
    )
    
    # Sample trajectories
    key = jax.random.PRNGKey(42)
    xi_samples, _ = projector.sample_uniform_trajectories(key)
    
    print(f"Sampled trajectory shape: {xi_samples.shape}")
    print(f"Number of variables: {projector.nvar}")
    print(f"Total constraints: {projector.num_total_constraints}")
    
    xi_filtered_init = xi_samples
    
    # Project the trajectories
    start_time = time.time()
    xi_filtered, prime_residuals, fixed_point_residuals = projector.compute_projection(
        xi_samples, xi_filtered_init)
    print(f"Projection time: {time.time() - start_time:.3f} seconds")
    
    # Convert to numpy for analysis
    xi_np = np.array(xi_samples)
    xi_filtered_np = np.array(xi_filtered)
    prime_residuals_np = np.array(prime_residuals)
    fixed_residuals_np = np.array(fixed_point_residuals)
    
    # Print convergence statistics
    print(f"\nConvergence Statistics:")
    # print(f"Final primal residual - Mean: {np.mean(prime_residuals_np[-1]):.6f}, Max: {np.max(prime_residuals_np[-1]):.6f}")
    # print(f"Final fixed point residual - Mean: {np.mean(fixed_residuals_np[-1]):.6f}, Max: {np.max(fixed_residuals_np[-1]):.6f}")
    
    print(f"Prime residuals shape: {prime_residuals_np.shape}")
    print(f"Fixed point residuals shape: {fixed_residuals_np.shape}")
    
    # Save results
    os.makedirs('results_FD', exist_ok=True)
    np.savetxt('results_FD/original_trajectory.csv', xi_np, delimiter=',')  # Save first sample
    np.savetxt('results_FD/projected_trajectory.csv', xi_filtered_np, delimiter=',')
    np.savetxt('results_FD/prime_residuals.csv', prime_residuals_np, delimiter=',')
    np.savetxt('results_FD/fixed_residuals.csv', fixed_residuals_np, delimiter=',')
    
    # # Visualize results for first DOF
    # visualize_trajectory(xi_np[0], xi_filtered_np[0], dof_idx=0, dof=projector.num_dof, dt=projector.t)
    
    # # Visualize residuals convergence
    # visualize_residuals(prime_residuals_np, fixed_residuals_np, batch_idx=0)
    
    print("Analysis complete. Check the generated plots and saved CSV files.")

if __name__ == "__main__":
    main()