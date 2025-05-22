import os
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import time
import matplotlib
matplotlib.use("Agg")

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
        self.v_goal = 0.1*jnp.ones(num_dof)

        
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

        # self.t = self.t + 0.001  # Add a small value to avoid division by zero
        
        # First-order finite difference for acceleration (from velocity)
        D1 = np.zeros((n, n))
        for i in range(n):
            D1[i, i] = -1
            if i < n-1:
                D1[i, i+1] = 1
   

        D1 = D1 / (self.t)
        
        # Second-order finite difference for jerk (from velocity)
        D2 = np.zeros((n, n))
        for i in range(n):
            #D2[i, i] = 1
            if i < n-2:
                D2[i, i] = 1
                
                D2[i, i+1] = -2
                D2[i, i+2] = 1


        D2 = D2 / (self.t**2)
        
        self.D1 = jnp.array(D1)  # Acceleration operator
        self.D2 = jnp.array(D2)  # Jerk operator

        # print("self.D1", self.D1.shape)
        # print("self.D2", self.D2.shape)
    
    def setup_optimization_matrices(self):
        """Setup matrices needed for the quadratic programming problem"""
        n = self.num_steps
        d = self.num_dof
        
        
        
        # Identity matrix for the objective function
        #self.Q = jnp.eye(self.nvar)
        #self.Q_inv = jnp.linalg.inv(self.Q)

        
        # Construct constraint matrices
        
        # Velocity constraints (box constraints)

        
        self.A_v = np.vstack((np.identity(self.num_steps), -np.identity(self.num_steps)))

        self.A_v_ineq = np.kron(np.identity(self.num_dof), self.A_v )


        self.A_a = np.vstack((self.D1, -self.D1))

        self.A_a_ineq = np.kron(np.identity(self.num_dof), self.A_a )


        self.A_j = np.vstack((self.D2, -self.D2))

        self.A_j_ineq = np.kron(np.identity(self.num_dof), self.A_j )

        # print("self.A_v_ineq", self.A_v_ineq.shape)
        # print("self.A_a_ineq", self.A_a_ineq.shape)
        # print("self.A_j_ineq", self.A_j_ineq.shape)

        

        # #self.A_v_ineq = jnp.eye(self.nvar)
        
        # # Acceleration constraints
        # # For each DOF, we need a matrix to compute accelerations
        # A_a_list = []
        # for i in range(d):
        #     # Create acceleration constraint matrix for this DOF
        #     A_dof = jnp.zeros((n, n*d))
        #     # Fill in the finite difference coefficients
        #     A_dof = A_dof.at[:, i*n:(i+1)*n].set(self.D1)
        #     A_a_list.append(A_dof)
        
        # # Combine all DOF constraints
        # self.A_a_ineq = jnp.vstack(A_a_list, -A_a_list)

        
        
        # # Jerk constraints
        # # For each DOF, we need a matrix to compute jerks
        # A_j_list = []
        # for i in range(d):
        #     # Create jerk constraint matrix for this DOF
        #     A_dof = jnp.zeros((n, n*d))
        #     # Fill in the finite difference coefficients
        #     A_dof = A_dof.at[:, i*n:(i+1)*n].set(self.D2)
        #     A_j_list.append(A_dof)
        
        # # Combine all DOF constraints
        # self.A_j_ineq = jnp.vstack(A_j_list, -A_j_list)

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
    def compute_feasible_control(self, lamda_v, lamda_a, lamda_j, s_v, s_a, s_j, xi_samples):
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
        v_max_temp = jnp.hstack(( self.v_max*jnp.ones((self.num_batch, self.num_steps)),  self.v_max*jnp.ones((self.num_batch, self.num_steps)) ))
		
        v_max_vec = jnp.tile(v_max_temp, (1, self.num_dof))
        
        b_v = v_max_vec
        
        # Acceleration constraints
        a_max_temp = jnp.hstack(( self.a_max*jnp.ones((self.num_batch, self.num_steps)),  self.a_max*jnp.ones((self.num_batch, self.num_steps)) ))
        
        a_max_vec = jnp.tile(a_max_temp, (1, self.num_dof))


        b_a = a_max_vec
        
        # Jerk constraints
        j_max_temp = jnp.hstack(( self.j_max*jnp.ones((self.num_batch, self.num_steps)),  self.j_max*jnp.ones((self.num_batch, self.num_steps)) ))
        
        j_max_vec = jnp.tile(j_max_temp, (1, self.num_dof))

        b_j = j_max_vec
        
        # Augmented bounds with slack variables
        b_v_aug = b_v - s_v
        b_a_aug = b_a - s_a
        b_j_aug = b_j - s_j

        # jax.debug.print("shape of b_v_aug: {}", b_v_aug.shape)
        

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
        s_v = jnp.maximum(jnp.zeros((self.num_batch, 2*self.nvar)), 
                         -jnp.dot(self.A_v_ineq, primal_sol.T).T + b_v)
        res_v = jnp.dot(self.A_v_ineq, primal_sol.T).T - b_v + s_v
        
        s_a = jnp.maximum(jnp.zeros((self.num_batch, 2*self.nvar)), 
                         -jnp.dot(self.A_a_ineq, primal_sol.T).T + b_a)
        res_a = jnp.dot(self.A_a_ineq, primal_sol.T).T - b_a + s_a
        
        s_j = jnp.maximum(jnp.zeros((self.num_batch, 2*self.nvar)), 
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

        # jax.debug.print("max res_projection: {}", jnp.max(res_projection))
        # jax.debug.print("min res_projection: {}", jnp.min(res_projection))
        
        return primal_sol, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, res_projection
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_projection(self, xi_samples, xi_filtered):
        """Project sampled trajectories to satisfy constraints using jax.lax.scan"""
        # Initialize slack variables and Lagrange multipliers
        s_v = jnp.zeros((self.num_batch, 2*self.nvar))
        s_a = jnp.zeros((self.num_batch, 2*self.nvar))
        s_j = jnp.zeros((self.num_batch, 2*self.nvar))
        
        lamda_v = jnp.zeros((self.num_batch, self.nvar))
        lamda_a = jnp.zeros((self.num_batch, self.nvar))
        lamda_j = jnp.zeros((self.num_batch, self.nvar))

        xi_filtered_init = xi_filtered
        
        # Define the scan function for ADMM iterations
        def lax_custom_projection(carry, x):
            # carry contains the state variables
            xi_filtered, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j = carry
            xi_filtered_prev =  xi_filtered
            lamda_v_prev = lamda_v
            lamda_a_prev = lamda_a
            lamda_j_prev = lamda_j
            
            # Perform one projection iteration
            xi_filtered, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j, primal_residual = self.compute_feasible_control(
                lamda_v, lamda_a, lamda_j, s_v, s_a, s_j, xi_samples)
            
            # Update carry state
            new_carry = (xi_filtered, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j)
			
            fixed_point_residual = jnp.linalg.norm(xi_filtered_prev-xi_filtered, axis = 1)+jnp.linalg.norm(lamda_v_prev-lamda_v, axis = 1)
            fixed_point_residual += jnp.linalg.norm(lamda_a_prev-lamda_a, axis = 1) + jnp.linalg.norm(lamda_j_prev-lamda_j, axis = 1)
            
            # Return residuals as outputs to be accumulated
            #return new_carry, res
        
            return (xi_filtered, s_v, s_a, s_j, lamda_v, lamda_a, lamda_j), (primal_residual, fixed_point_residual)
        
        # Initialize carry state
        carry_init = (
            xi_filtered_init,  # primal_sol
            s_v, s_a, s_j,                          # slack variables
            lamda_v, lamda_a, lamda_j                # Lagrange multipliers
        )
        
        # Run scan over iterations
        # x input is just iteration indices, not used in computation
        
        carry_final, residual_tot = jax.lax.scan(
            lax_custom_projection, 
            carry_init, 
            jnp.arange(self.maxiter_projection)
        )
        
        # Extract final primal solution
        final_primal_sol,_,_,_,_,_,_ = carry_final

        prime_residual, fixed_point_residual = residual_tot
        
        return final_primal_sol, prime_residual, fixed_point_residual
    
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

def visualize_trajectory(original, projected, dof_idx=0, dof=1, dt=0.05):
    """Visualize original and residual trajectories separately for a specific DOF"""
    
    num_steps = original.shape[0] // dof
    
    # Extract velocities for the specified DOF
    orig_vel = original[dof_idx*num_steps : (dof_idx+1)*num_steps]
    proj_vel = projected[dof_idx*num_steps : (dof_idx+1)*num_steps]
    
    # Calculate residual velocity (original - projected)
    residual_vel = orig_vel - proj_vel
    
    # Create time vector
    time = np.arange(num_steps) * dt
    
    # Plot original velocity
    plt.figure(figsize=(10, 4))
    plt.plot(time, orig_vel, 'b-', label='Original')
    plt.axhline(y=1.0, color='g', linestyle='--', label='v_max')
    plt.axhline(y=-1.0, color='g', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Joint {dof_idx} Velocity')
    plt.title(f'Original Joint {dof_idx} Velocity')
    plt.legend()
    plt.grid(True)
    try:
        plt.show()
    except:
        plt.savefig(f"original_trajectory_dof{dof_idx}.png")
        print(f"Original plot saved as original_trajectory_dof{dof_idx}.png")
    
    # Plot residual velocity
    plt.figure(figsize=(10, 4))
    plt.plot(time, residual_vel, 'm-', label='Residual (Original - Projected)')
    plt.axhline(y=0.0, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Joint {dof_idx} Residual Velocity')
    plt.title(f'Residual Velocity for Joint {dof_idx}')
    plt.legend()
    plt.grid(True)
    try:
        plt.show()
    except:
        plt.savefig(f"residual_trajectory_dof{dof_idx}.png")
        print(f"Residual plot saved as residual_trajectory_dof{dof_idx}.png")


def visualize_residuals(prime_residuals, num_steps, batch_idx=0, dt=0.05):
    """Visualize residuals across iterations for a specific batch sample"""
    
    # prime_residuals shape: (maxiter_projection, num_batch)
    # Extract residuals for specific batch sample
    time = np.arange(num_steps) * dt
    residuals_sample = prime_residuals[:, batch_idx]
    
    iterations = np.arange(len(residuals_sample))
    
    plt.figure(figsize=(10, 6))
    plt.plot(residuals_sample, 'b-', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.title(f'ADMM Residuals Convergence (Batch Sample {batch_idx})')
    plt.grid(True)
    
    try:
        plt.show()
    except:
        plt.savefig(f"residuals_batch{batch_idx}.png")
        print(f"Residuals plot saved as residuals_batch{batch_idx}.png")

def main():
    # Initialize the projector
    projector = TrajectoryProjector(
        num_dof=1,
        num_steps=10,
        num_batch=100,
        timestep=0.05,
        maxiter_projection=20,  # Increased to see convergence better
        v_max=1.0,
        a_max=2.0,
        j_max=3.0,
        rho_ineq= 1.0,
        rho_projection=1.0,
    )
    
    # Sample a trajectory
    key = jax.random.PRNGKey(42)
    xi_samples, _ = projector.sample_uniform_trajectories(key)

    print(f"Sampled trajectory shape: {xi_samples.shape}")

    xi_filtered_init = xi_samples
    
    # Project the trajectory
    start_time = time.time()
    xi_filtered, prime_residuals, fixed_point_residuals = projector.compute_projection(xi_samples, xi_filtered_init)

    print(f"prime residuals shape: {prime_residuals.shape}")  # Should be (maxiter_projection, num_batch)
    print(f"fixed point residuals shape: {fixed_point_residuals.shape}")  # Should be (num_batch)

    print(f"Projection time: {time.time() - start_time:.3f} seconds")
    
    # Convert to numpy for saving/analysis
    # xi_np = np.mean(xi_samples, axis=0) 
    # xi_filtered_np = np.mean(xi_filtered, axis=0)
    xi_np = np.array(xi_samples)
    xi_filtered_np = np.array(xi_filtered)
    prime_residuals_np = np.array(prime_residuals)
    fixed_residuals_np = np.array(fixed_point_residuals)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.savetxt('results/original_trajectory.csv', xi_np, delimiter=',')
    np.savetxt('results/projected_trajectory.csv', xi_filtered_np, delimiter=',')
    np.savetxt('results/prime_residuals.csv', prime_residuals_np, delimiter=',')
    np.savetxt('results/fixed_residuals.csv', fixed_residuals_np, delimiter=',')
    
    print("Generated sample trajectories")
    print(f"Original shape: {xi_np.shape}")
    print(f"xi_filtered shape: {xi_filtered_np.shape}")
    print(f"Prime residuals shape: {prime_residuals_np.shape}")
    print(f"Fixed residuals shape: {fixed_residuals_np.shape}")
    
    # Generate dataset
    # Uncomment to generate a dataset
    # projector.generate_dataset(num_samples=5000, output_dir="trajectory_dataset")

    # # Visualize trajectory
    # visualize_trajectory(xi_np, xi_filtered_np, dof_idx=0, dof=1, dt=0.05)
    
    # # Visualize residuals convergence
    # visualize_residuals(prime_residuals_np, batch_idx=0)

if __name__ == "__main__":
    main()