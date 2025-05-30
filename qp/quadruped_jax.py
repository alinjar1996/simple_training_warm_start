import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import time

from stance_leg_controller import ForceStanceLegController

#jax.config.update("jax_enable_x64", True)

class QuadrupedQPProjector:
    def __init__(self, 
                 num_batch=10,           # Batch size for parallel processing
                 maxiter=10,             # Maximum iterations for ADMM
                 rho=1.0,                # ADMM penalty parameter
                 desired_speed=(0.0, 0.0),
                 desired_twisting_speed=0.0,
                 desired_body_height=0.5,
                 body_mass=30.0,
                 body_inertia=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                 num_legs=4,
                 friction_coeff=0.2,
                 timestep=0.05,
                 horizon=10,
                 foot_x=0.2,
                 foot_y=0.2,
                 foot_z=-0.5):
        
        # Store parameters
        self.num_batch = num_batch
        self.maxiter = maxiter
        self.rho = rho
        
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
        
        # Foot positions
        self.foot_x = foot_x
        self.foot_y = foot_y
        self.foot_z = foot_z
        
        # Setup quadruped matrices
        self.setup_quadruped_matrices()
        
        # Setup optimization matrices
        self.setup_optimization_matrices()
        
        # Setup random key for JAX
        self.key = jax.random.PRNGKey(0)

    def setup_quadruped_matrices(self):
        """Setup foot positions and default states for quadruped"""
        # Foot positions in body frame
        FootPositionsInBodyFrame = jnp.array([
            self.foot_x, self.foot_y, self.foot_z,
            -self.foot_x, self.foot_y, self.foot_z,
            self.foot_x, -self.foot_y, self.foot_z,
            -self.foot_x, -self.foot_y, self.foot_z
        ])
        self.FootPositionsInBodyFrame = FootPositionsInBodyFrame.reshape(4, 3)
        
        # Default states
        self.BaseRollPitchYaw = (0.0, 0.0, 0.0)
        self.AngularVelocityBodyFrame = (0.0, 0.0, 0.0)
        self.ComVelocityBodyFrame = (0.0, 0.0, 0.0)
        self.FootContacts = (True, True, True, True)
        self.slope_estimate = (0.0, 0.0, 0.0)
        self.RotationBodyWrtWorld = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def setup_optimization_matrices(self):
        """Setup QP matrices using ForceStanceLegController"""
        # Create controller instance
        controller = ForceStanceLegController(
            desired_speed=self.desired_speed,
            desired_twisting_speed=self.desired_twisting_speed,
            desired_body_height=self.desired_body_height,
            body_mass=self.body_mass,
            body_inertia=self.body_inertia,
            num_legs=self.num_legs,
            friction_coeff=self.friction_coeff,
            timestep=self.timestep,
            horizon=self.horizon
        )
        
        # Get QP matrices
        H, g, C, c = controller.getMatrices(
            BaseRollPitchYaw=self.BaseRollPitchYaw,
            AngularVelocityBodyFrame=self.AngularVelocityBodyFrame,
            ComVelocityBodyFrame=self.ComVelocityBodyFrame,
            FootPositionsInBodyFrame=self.FootPositionsInBodyFrame,
            FootContacts=self.FootContacts,
            slope_estimate=self.slope_estimate,
            RotationBodyWrtWorld=self.RotationBodyWrtWorld
        )
        
        
        # Store matrices
        self.H = H                       # QP Hessian (3nk x 3nk)
        self.g = g                       # Linear term (3nk)
        self.C = C                       # Constraint matrix (num_total_constraints x 3nk)
                              # Lower bound (num_total_constraints)
        self.c = c                     # Upper bound (num_total_constraints)


        # Dimensions
        self.nvar = H.shape[0]         # 3nk
        

        #self.A_control = jnp.vstack((self.C,-self.C)) 
        
        self.A_control =self.C 

        self.num_total_constraints = self.A_control.shape[0] #Since stacking them later
        print("self.num_total_constraints", self.num_total_constraints)

        
        self.A_eq_single_horizon = jnp.tile(jnp.eye(3),self.num_legs)

        self.A_eq = jnp.kron(jnp.eye(self.horizon), self.A_eq_single_horizon)


        print("self.A_eq.shape", self.A_eq.shape)

        self.b_eq_single_horizon = jnp.tile(jnp.array([0.0, 0.0, self.body_mass*9.81]), (self.num_batch, 1))

        self.b_eq = jnp.tile(self.b_eq_single_horizon,(1,self.horizon))

        # self.b_eq = jnp.tile(jnp.array([0.0, 0.0, self.body_mass*9.81*self.horizon]), (self.num_batch, 1))

        print("self.b_eq.shape", self.b_eq.shape)



    
    @partial(jax.jit, static_argnums=(0,))
    def compute_s_init(self, xi_projected):
        """Initialize slack variables following  approach"""


        

        b_control = self.c

        # jax.debug.print("xi_projected: {0}", jnp.shape(xi_projected))
        # jax.debug.print("b_control: {0}", jnp.shape(b_control))
        # jax.debug.print("self.A_control: {0}", jnp.shape(self.A_control))

        # Initialize slack variables ()
        s = jnp.maximum(
            jnp.zeros((self.num_batch, self.num_total_constraints)),
            -jnp.dot(self.A_control, xi_projected.T).T + b_control
        )

        return s
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_feasible_control(self, s, xi_projected, lamda): 
        """
        Compute feasible control following  approach exactly
        """
        
        
        b_eq = self.b_eq

        b_control = self.c

        # Augmented bounds with slack variables
        b_control_aug = b_control - s
        
        # Cost matrix 
        cost = (self.H + self.rho * jnp.dot(self.A_control.T, self.A_control))

        print("cost.shape", cost.shape)
        print("self.A_eq.shape", self.A_eq.shape)
        
        # KKT system matrix ()
        #cost_mat = cost + 0.001*jnp.eye(self.nvar)
                
        cost_mat = jnp.vstack((
            jnp.hstack((cost, self.A_eq.T)),
            jnp.hstack((self.A_eq, jnp.zeros((jnp.shape(self.A_eq)[0], jnp.shape(self.A_eq)[0]))))
        ))
        
        # Linear cost term (following )
        lincost = (-lamda 
                    + self.g - 
                  self.rho * jnp.dot(self.A_control.T, b_control_aug.T).T)
        
        # Solve KKT system ()
        sol = jnp.linalg.solve(cost_mat, jnp.hstack((-lincost, b_eq)).T).T
        
        #sol = jnp.linalg.solve(cost_mat, (-lincost).T).T
        
        # Extract primal solution
        xi_projected = sol[:, 0:self.nvar]
        
        # Update slack variables (following )
        s = jnp.maximum(
            jnp.zeros((self.num_batch, self.num_total_constraints)),
            -jnp.dot(self.A_control, xi_projected.T).T + b_control
        )

        #print("s", (jnp.dot(self.A_control, xi_projected.T).T + b_control).shape)
        
        # Compute residual (following )
        res_vec = jnp.dot(self.A_control, xi_projected.T).T - b_control + s

        #print("res_vec" , res_vec.shape)
        res_norm = jnp.linalg.norm(res_vec, axis=1)
        #print("res_norm", res_norm.shape)
        
        # Update Lagrange multipliers (following )
        lamda = lamda - self.rho * jnp.dot(self.A_control.T, res_vec.T).T

        
        # print("lamda", lamda.shape)
        return xi_projected, s, res_norm, lamda

    @partial(jax.jit, static_argnums=(0,))
    def compute_qp_projection(self, xi_init, lamda_init):
        """Run batched ADMM iterations to project xi_init onto constraints"""

        xi_projected_init = xi_init
        
        s_init = self.compute_s_init(xi_projected_init)

        
        def lax_custom_qp(carry, _):
            
            xi_projected, lamda, s = carry

            xi_prev = xi_projected
            lamda_prev = lamda
            s_prev = s

            xi_projected, s, res_norm, lamda = self.compute_feasible_control(s, xi_projected, lamda)
            # xi_new, lamda_new, primal_residual, fixed_point_residual = self.compute_feasible_control(xi, s_init, xi_projected, lamda)
            primal_residual = res_norm
            fixed_point_residual = (jnp.linalg.norm(xi_projected - xi_prev, axis=1) + 
                                  jnp.linalg.norm(lamda - lamda_prev, axis=1) +
                                  jnp.linalg.norm(s - s_prev, axis=1))
            return (xi_projected, lamda, s), (primal_residual, fixed_point_residual)

        # Initialize
         # Initialize carry
        carry_init = (xi_projected_init, lamda_init, s_init)
    
        carry_final, (primal_residual, fixed_point_residual) = jax.lax.scan(
            lax_custom_qp, carry_init, xs=None, length=self.maxiter
        )

        xi_projected, lamda, s = carry_final

        #xi_final, lamda_final = carry_final
        return xi_projected, primal_residual, fixed_point_residual


    def print_problem_info(self):
        """Print information about the QP problem dimensions"""
        print("=== Quadruped QP Problem Information ===")
        print(f"H matrix shape: {self.H.shape}")
        print(f"g vector shape: {self.g.shape}")
        print(f"C matrix shape: {self.C.shape}")
        print(f"constraint limit vector shape: {self.c.shape}")
        print(f"Number of variables: {self.nvar}")
        print(f"Number of constraints: {self.num_total_constraints}")
        print(f"Batch size: {self.num_batch}")
        print(f"Max iterations: {self.maxiter}")
        print(f"ADMM penalty (rho): {self.rho}")
        print("=" * 40)

def main():
    """Main function demonstrating the batched quadruped QP projector"""
    
    num_batch=1  # Increased batch size to demonstrate batching
    maxiter=200
    rho=1.0
    desired_speed=(0.0, 0.0)
    desired_twisting_speed=0.0
    desired_body_height=0.5
    body_mass=30.0
    body_inertia=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    num_legs=4
    friction_coeff=0.2
    timestep=0.05
    horizon=2
    foot_x=0.2
    foot_y=0.2
    foot_z=-desired_body_height
    
    # Initialize the projector with organized parameters
    projector = QuadrupedQPProjector(
        num_batch=num_batch,
        maxiter=maxiter,
        rho=rho,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=desired_body_height,
        body_mass=body_mass,
        body_inertia=body_inertia,
        num_legs=num_legs,
        friction_coeff=friction_coeff,
        timestep=timestep,
        horizon=horizon,
        foot_x=foot_x,
        foot_y=foot_y,
        foot_z=foot_z
    )
    
    # Print problem information
    projector.print_problem_info()
    
    # Sample batched initial guess
    #key = jax.random.PRNGKey(42)
    

    lamda_init = jnp.zeros((projector.num_batch, projector.nvar))
    
    #print(f"Initial xi batch shape: {xi_init.shape}")
    print(f"Initial lambda batch shape: {lamda_init.shape}")
    
    # Sample random force trajectories for projection
    #xi_samples, key = projector.sample_uniform_forces(key)

    xi_init = jnp.zeros((projector.num_batch, projector.nvar))
    # for i in range(projector.num_legs*projector.horizon):
    #     xi_init = xi_init.at[:, 3*i+2].set(body_mass * 9.81 / 4.0)
    
    force_input = -xi_init[:, :12]
    print(f"Initial Force Input shape: {force_input.shape}")
    print(f"Initial Force Input: {force_input[0]}")


    # Solve batched QP projection
    start_time = time.time()
    xi_proj, primal_residual, fixed_point_residual = projector.compute_qp_projection(xi_init, lamda_init)
    solve_time = time.time() - start_time
    
    print(f"\n=== Solution Results ===")

    ##
    print(f"Primal Residual Shape: {primal_residual.shape}")
    print(f"Primal Residual: {primal_residual[-1]}")
    print(f"Projection time: {solve_time:.6f} seconds")
    print(f"Projected xi batch shape: {xi_proj.shape}")
    print(f"Final primal residual shape: {primal_residual[-1].shape}")
    print(f"Final Fixed_Point Residual shape: {fixed_point_residual[-1].shape}")
    
    # Display convergence statistics
    primal_residual_np = np.array(primal_residual)
    fixed_point_residual_np = np.array(fixed_point_residual)
    
    print(f"\n=== Convergence Statistics ===")
    print(f"Primal residual - Initial (mean): {np.mean(primal_residual_np[0]):.6f}, Final (mean): {np.mean(primal_residual_np[-1]):.6f}")
    print(f"Fixed_Point Residual - Initial (mean): {np.mean(fixed_point_residual_np[0]):.6f}, Final (mean): {np.mean(fixed_point_residual_np[-1]):.6f}")
    print(f"Primal residual - Final (max): {np.max(primal_residual_np[-1]):.6f}")
    print(f"Fixed_Point Residual - Final (max): {np.max(fixed_point_residual_np[-1]):.6f}")
    
    # Extract force outputs (first 12 elements as in original code) for all batches
    force_output = -xi_proj[:, :12]
    print(f"\n=== Force Output ===")
    print(f"Projected forces batch shape: {force_output.shape}")
    print(f"First horizon projected forces: {force_output[0]}")
    force_output = -xi_proj[:, 12:24]
    print(f"\n=== Force Output ===")
    print(f"Projected forces batch shape: {force_output.shape}")
    print(f"Second horizon projected forces: {force_output[0]}")
    

    # Checking Equality Constraints
    
    print("Checking Equality Constraints:")
    # print("projector.A_eq.shape", projector.A_eq.shape)
    # print(f"xi_proj.shape", xi_proj.shape)
    # print(f"projector.b_eq.shape", projector.b_eq.shape)
    eq_res = jnp.matmul(projector.A_eq, xi_proj.T) - projector.b_eq.T
    print("eq_res max:", max(eq_res))
    print("eq_res min:", min(eq_res))
    
    print("\nBatched Quadruped QP projection complete!")

if __name__ == "__main__":
    main()