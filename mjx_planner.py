import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time
import matplotlib.pyplot as plt

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
                 j_max=5.0):           # Maximum joint jerk
        # Total number of variables (velocities for all DoFs and timesteps)
        
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
        
        #Boundaries
        self.v_start = jnp.zeros(num_dof)
        self.v_goal = jnp.zeros(num_dof)

        
        # Setup finite difference matrices for velocity, acceleration, and jerk
        self.setup_finite_difference_matrices()
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
        
        # Setup random key for JAX
        self.key = jax.random.PRNGKey(0)

        self.compute_boundary_vec_batch = (jax.vmap(self.compute_boundary_vec_single, in_axes = (0)  ))
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_boundary_vec_single(self, boundary_state):
        b_eq_term = boundary_state.reshape(2, self.num_dof).T
        b_eq_term = b_eq_term.reshape(2 * self.num_dof )
        return b_eq_term
    
    def setup_finite_difference_matrices(self):
        """Create finite difference matrices for computing acceleration and jerk"""
        n = self.num_steps

        self.t = self.t + 0.001  # Add a small value to avoid division by zero
        
        # First-order finite difference for acceleration (from velocity)
        D1 = np.zeros((n-1, n))
        for i in range(n-1):
            D1[i, i] = -1
            D1[i, i+1] = 1
        D1 = D1 / (self.t)
        
        # Second-order finite difference for jerk (from velocity)
        D2 = np.zeros((n-2, n))
        for i in range(n-2):
            D2[i, i] = 1
            D2[i, i+1] = -2
            D2[i, i+2] = 1
        D2 = D2 / (self.t**2)
        
        self.D1 = jnp.array(D1)  # Acceleration operator
        self.D2 = jnp.array(D2)  # Jerk operator
    
    def setup_optimization_matrices(self):
        """Setup matrices needed for the quadratic programming problem"""
        n = self.num_steps
        d = self.num_dof
        
        
        
        # Identity matrix for the objective function
        #self.Q = jnp.eye(self.nvar)
        #self.Q_inv = jnp.linalg.inv(self.Q)

        
        # Construct constraint matrices
        
        # Velocity constraints (box constraints)
        self.A_v_ineq = jnp.eye(self.nvar)
        
        # Acceleration constraints
        # For each DOF, we need a matrix to compute accelerations
        A_a_list = []
        for i in range(d):
            # Create acceleration constraint matrix for this DOF
            A_dof = jnp.zeros((n-1, n*d))
            # Fill in the finite difference coefficients
            A_dof = A_dof.at[:, i*n:(i+1)*n].set(self.D1)
            A_a_list.append(A_dof)
        
        # Combine all DOF constraints
        self.A_a_ineq = jnp.vstack(A_a_list)

        
        
        # Jerk constraints
        # For each DOF, we need a matrix to compute jerks
        A_j_list = []
        for i in range(d):
            # Create jerk constraint matrix for this DOF
            A_dof = jnp.zeros((n-2, n*d))
            # Fill in the finite difference coefficients
            A_dof = A_dof.at[:, i*n:(i+1)*n].set(self.D2)
            A_j_list.append(A_dof)
        
        # Combine all DOF constraints
        self.A_j_ineq = jnp.vstack(A_j_list)

        self.A_eq = jnp.kron(
                    jnp.eye(self.num_dof),                    # shape: (d, d)
                    jnp.array([[1.0] + [0.0] * (self.num_steps - 1),  # shape: (2, T)
                            [0.0] * (self.num_steps - 1) + [1.0]]) # picks first and last timestep
                )
        
        
        
        
        # Matrix to go from optimization variables to DOF velocities
        self.A_thetadot = jnp.eye(self.nvar)
        
        # Projection matrices
        self.A_projection = jnp.eye(self.nvar)

        self.Q_ = jnp.dot(self.A_projection.T, self.A_projection) + \
                  self.rho_ineq * jnp.dot(self.A_v_ineq.T, self.A_v_ineq) + \
                  self.rho_ineq * jnp.dot(self.A_a_ineq.T, self.A_a_ineq) + \
                  self.rho_ineq * jnp.dot(self.A_j_ineq.T, self.A_j_ineq)
        
        
        self.Q_inv = jnp.linalg.inv(
            jnp.vstack((
                jnp.hstack((
                    self.Q_,
                    self.A_eq.T
                )),
                jnp.hstack((
                    self.A_eq,
                    np.zeros((np.shape(self.A_eq)[0], np.shape(self.A_eq)[0]))
                ))
            ))
        )

        
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_projection(self, lamda_v, lamda_a, lamda_j, s_v, s_a, s_j, xi_samples):
        """
        Project the sampled velocities onto the feasible set defined by constraints
        
        Args:
            lamda_v, lamda_a, lamda_j: Lagrange multipliers for constraints
            s_v, s_a, s_j: Slack variables for inequality constraints
            xi_samples: Sampled velocity trajectories
        
        Returns:
            Projected velocities, updated slack variables, updated multipliers, residual
        """
        # Setup constraint bounds
        # Velocity constraints
        v_max_vec = self.v_max * jnp.ones((self.num_batch, self.nvar))
        b_v = v_max_vec
        
        # Acceleration constraints
        a_max_vec = self.a_max * jnp.ones((self.num_batch, (self.num_steps-1)*self.num_dof))
        b_a = a_max_vec
        
        # Jerk constraints
        j_max_vec = self.j_max * jnp.ones((self.num_batch, (self.num_steps-2)*self.num_dof))
        b_j = j_max_vec
        
        # Augmented bounds with slack variables
        b_v_aug = b_v - s_v
        b_a_aug = b_a - s_a
        b_j_aug = b_j - s_j
        

        v_start = jnp.tile(self.v_start, (self.num_batch, 1))
        v_goal = jnp.tile(self.v_goal, (self.num_batch, 1))

        boundary_state = jnp.hstack([v_start, v_goal])  # shape (2 * num_dof,)
        boundary_state = jnp.asarray(boundary_state)  
        
        b_eq_term = self.compute_boundary_vec_batch(boundary_state)


        # Linear cost term for the QP
        lincost = -lamda_v - lamda_a - lamda_j - self.rho_projection * jnp.dot(self.A_projection.T, xi_samples.T).T - \
                  self.rho_ineq * jnp.dot(self.A_v_ineq.T, b_v_aug.T).T - \
                  self.rho_ineq * jnp.dot(self.A_a_ineq.T, b_a_aug.T).T - \
                  self.rho_ineq * jnp.dot(self.A_j_ineq.T, b_j_aug.T).T
        
        
        # Solve the QP
        sol = jnp.dot(self.Q_inv,  jnp.hstack(( -lincost, b_eq_term )).T).T
        primal_sol = sol[:, 0:self.nvar]
        
        # jax.debug.print("shape of primal_sol: {}", primal_sol.shape)

        # Update slack variables
        s_v = jnp.maximum(jnp.zeros((self.num_batch, self.nvar)), 
                         -jnp.dot(self.A_v_ineq, primal_sol.T).T + b_v)
        res_v = jnp.dot(self.A_v_ineq, primal_sol.T).T - b_v + s_v
        
        s_a = jnp.maximum(jnp.zeros((self.num_batch, (self.num_steps-1)*self.num_dof)), 
                         -jnp.dot(self.A_a_ineq, primal_sol.T).T + b_a)
        res_a = jnp.dot(self.A_a_ineq, primal_sol.T).T - b_a + s_a
        
        s_j = jnp.maximum(jnp.zeros((self.num_batch, (self.num_steps-2)*self.num_dof)), 
                         -jnp.dot(self.A_j_ineq, primal_sol.T).T + b_j)
        res_j = jnp.dot(self.A_j_ineq, primal_sol.T).T - b_j + s_j
        
        # Update Lagrange multipliers
        lamda_v = lamda_v - self.rho_ineq * jnp.dot(self.A_v_ineq.T, res_v.T).T
        lamda_a = lamda_a - self.rho_ineq * jnp.dot(self.A_a_ineq.T, res_a.T).T
        lamda_j = lamda_j - self.rho_ineq * jnp.dot(self.A_j_ineq.T, res_j.T).T
        
        # Calculate residuals
        res_v_vec = jnp.linalg.norm(res_v, axis=1)
        res_a_vec = jnp.linalg.norm(res_a, axis=1)
        res_j_vec = jnp.linalg.norm(res_j, axis=1)
        
        res_projection = res_v_vec + res_a_vec + res_j_vec
        
        return primal_sol, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, res_projection
    
    @partial(jax.jit, static_argnums=(0,))
    def project_trajectories(self, xi_samples):
        """Project sampled trajectories to satisfy constraints"""
        # Initialize slack variables and Lagrange multipliers
        s_v = jnp.zeros((self.num_batch, self.nvar))
        s_a = jnp.zeros((self.num_batch, (self.num_steps-1)*self.num_dof))
        s_j = jnp.zeros((self.num_batch, (self.num_steps-2)*self.num_dof))
        
        lamda_v = jnp.zeros((self.num_batch, self.nvar))
        lamda_a = jnp.zeros((self.num_batch, self.nvar))
        lamda_j = jnp.zeros((self.num_batch, self.nvar))
        
        # Run ADMM iterations
        def proj_iter(i, state):
            primal_sol, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, _ = state
            primal_sol, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, res = self.compute_projection(
                lamda_v, lamda_a, lamda_j, s_v, s_a, s_j, xi_samples)
            #jax.debug.print("Iteration {}: primal_sol = {}", i, primal_sol)
            return primal_sol, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, res
        
        # Initialize state
        state = (jnp.zeros((self.num_batch, self.nvar)), s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, 
                 jnp.zeros(self.num_batch))
        
        # Run iterations
        for i in range(self.maxiter_projection):
            state = proj_iter(i, state)
        
        primal_sol, _, _, _, _, _, _, _ = state
        return primal_sol
    
    @partial(jax.jit, static_argnums=(0,))
    def sample_uniform_trajectories(self, key):
        """Sample velocity trajectories uniformly between -v_max and v_max"""
        key, subkey = jax.random.split(key)
        # Sample uniform velocities
        xi_samples = jax.random.uniform(
            key, 
            shape=(self.num_batch, self.nvar), 
            minval=-self.v_max, 
            maxval=self.v_max
        )
        return xi_samples, key
    
    def generate_dataset(self, num_samples, output_dir="trajectory_dataset"):
        """Generate dataset of trajectories by sampling and projecting"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate samples in batches
        for i in range(0, num_samples, self.num_batch):
            print(f"Generating samples {i} to {min(i+self.num_batch, num_samples)}")
            
            # Sample random trajectories
            xi_samples, self.key = self.sample_uniform_trajectories(self.key)
            
            # Project to feasible trajectories
            projected_trajectories = self.project_trajectories(xi_samples)
            
            # Compute accelerations and jerks from projected velocities
            accelerations = []
            jerks = []
            
            for j in range(self.num_dof):
                # Extract velocities for this DOF
                vels = projected_trajectories[:, j*self.num_steps:(j+1)*self.num_steps]
                
                # Compute accelerations for this DOF
                accs = jnp.matmul(vels, self.D1.T)
                accelerations.append(accs)
                
                # Compute jerks for this DOF
                jrks = jnp.matmul(vels, self.D2.T)
                jerks.append(jrks)
            
            # Convert to numpy for saving
            xi_np = np.array(xi_samples)
            proj_np = np.array(projected_trajectories)
            acc_np = np.array(jnp.concatenate(accelerations, axis=1))
            jerk_np = np.array(jnp.concatenate(jerks, axis=1))
            
            # Save batch
            batch_filename = f"batch_{i//self.num_batch}"
            np.savez(
                os.path.join(output_dir, batch_filename),
                original=xi_np,
                projected=proj_np,
                accelerations=acc_np,
                jerks=jerk_np
            )
            
            # If we've generated enough samples, break
            if i + self.num_batch >= num_samples:
                break
        
        print(f"Dataset generated with {num_samples} samples in {output_dir}")

def visualize_trajectory(original, projected, dof_idx=0):
    """Visualize original and projected trajectories for a specific DOF"""
    
    
    # Get the number of time steps
    num_steps = original.shape[0] // 6
    
    # Extract the velocities for the specified DOF
    orig_vel = original[dof_idx*num_steps:(dof_idx+1)*num_steps]
    proj_vel = projected[dof_idx*num_steps:(dof_idx+1)*num_steps]
    
    # Create time vector
    time = np.arange(num_steps) * 0.05  # assuming timestep=0.05
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, orig_vel, 'b-', label='Original')
    plt.plot(time, proj_vel, 'r-', label='Projected')
    plt.axhline(y=1.0, color='g', linestyle='--', label='v_max')
    plt.axhline(y=-1.0, color='g', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Joint {dof_idx} Velocity')
    plt.title('Original vs Projected Joint Velocity')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize the projector
    projector = TrajectoryProjector(
        num_dof=1,
        num_steps=50,
        num_batch=100,
        timestep=0.05,
        maxiter_projection=50,
        v_max=2.0,
        a_max=3.0,
        j_max=6.0
    )
    
    # Sample a trajectory
    key = jax.random.PRNGKey(42)
    xi_samples, _ = projector.sample_uniform_trajectories(key)
    
    # Project the trajectory
    start_time = time.time()
    projected = projector.project_trajectories(xi_samples)
    print(f"Projection time: {time.time() - start_time:.3f} seconds")
    
    # Convert to numpy for saving/analysis
    xi_np = np.array(xi_samples[0])
    projected_np = np.array(projected[0])
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/original_trajectory.csv', xi_np, delimiter=',')
    np.savetxt('results/projected_trajectory.csv', projected_np, delimiter=',')
    
    print("Generated sample trajectories")
    print(f"Original shape: {xi_np.shape}")
    print(f"Projected shape: {projected_np.shape}")
    
    # Generate dataset
    # Uncomment to generate a dataset
    # projector.generate_dataset(num_samples=5000, output_dir="trajectory_dataset")

if __name__ == "__main__":
    main()