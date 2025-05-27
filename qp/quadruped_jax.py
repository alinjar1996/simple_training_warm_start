import jax
import jax.numpy as jnp
from functools import partial

from stance_leg_controller import TorqueStanceLegController

class QuadrupedQPProjector:
    def __init__(self, H, g, C, c, cb, num_batch=500, maxiter=20, rho=1.0):
        self.H = H                       # QP Hessian (3nk x 3nk)
        self.g = g                       # Linear term (3nk)
        self.C = C                       # Constraint matrix (num_constraints x 3nk)
        self.c = c                       # Lower bound (num_constraints)
        self.cb = cb                     # Upper bound (num_constraints)

        self.num_batch = num_batch       # Batch size
        self.nvar = H.shape[0]           # 3nk
        self.num_constraints = C.shape[0]

        self.rho = rho                   # ADMM penalty
        self.maxiter = maxiter

        self.H_rho = H + rho * C.T @ C

    @partial(jax.jit, static_argnums=(0,))
    def solve_admm(self, xi_init, lam_init):
        """ Run ADMM iterations to project xi_init onto constraints"""

        def admm_step(carry, _):
            xi, lam = carry

            # U-update: solve (H + rho * C^T C) x = -g + rho C^T(z - lam)
            lincost = -self.g + self.rho * self.C.T @ (jnp.clip(self.C @ xi + lam, self.c, self.cb) - lam)
            xi_new = jnp.linalg.solve(self.H_rho, lincost)

            # Z-update: project to box
            # Slack variable
            s = jnp.clip(self.C @ xi_new + lam, self.c, self.cb)

            # Dual update
            lam_new = lam + self.C @ xi_new - s

            # Residuals
            primal_res = jnp.linalg.norm(self.C @ xi_new - s)
            dual_res = jnp.linalg.norm(xi_new - xi)

            return (xi_new, lam_new), (primal_res, dual_res)

        # Initialize
        carry_init = (xi_init, lam_init)
        carry_final, (primal_resid, dual_resid) = jax.lax.scan(admm_step, carry_init, xs=None, length=self.maxiter)

        xi_final, lam_final = carry_final
        return xi_final, lam_final, primal_resid, dual_resid


def main():
    # Placeholder dimensions


    # Example QP matrices
    key = jax.random.PRNGKey(0)

    foot_x = 0.2
    foot_y = 0.2
    foot_z = -0.5
    FootPositionsInBodyFrame = jnp.array([foot_x,foot_y,foot_z,-foot_x,foot_y,foot_z,foot_x,-foot_y,foot_z,-foot_x,-foot_y,foot_z])
    FootPositionsInBodyFrame = FootPositionsInBodyFrame.reshape(4,3)
    H,g,C,c = TorqueStanceLegController(desired_speed= (0.0,0.0),
                                        desired_twisting_speed= 0.0,
                                        desired_body_height=0.5,
                                        body_mass=30.0,
                                        body_inertia=(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0),
                                        num_legs=4,
                                        friction_coeff=0.2,
                                        timestep=0.05,
                                        horizon = 10
                                        ).getMatrices(BaseRollPitchYaw=(0.0,0.0,0.0),
                                                      AngularVelocityBodyFrame=(0.0,0.0,0.0),
                                                      ComVelocityBodyFrame=(0.0,0.0,0.0),
                                                      FootPositionsInBodyFrame=FootPositionsInBodyFrame,
                                                      FootContacts=(True,True,True,True),
                                                      slope_estimate=(0.0,0.0,0.0),
                                                      RotationBodyWrtWorld=(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0),
                                        )
    cb = -c
    print("H", H.shape)
    print("g", g.shape)
    print("C", C.shape)
    print("c", c.shape)
    print("cb", cb.shape)
    # H = jnp.eye(nvar)
    # g = jnp.ones(nvar)
    # C = jax.random.normal(key, shape=(num_constraints, nvar))
    # c = -jnp.ones(num_constraints)
    # cb = jnp.ones(num_constraints)

    nvar = jnp.shape(H)[0]
    num_constraints = jnp.shape(C)[0]
    

    projector = QuadrupedQPProjector(H, g, C, c, cb, maxiter=1000, rho=1.0)

    xi_init = jnp.zeros(nvar)
    lam_init = jnp.zeros(num_constraints)

    xi_proj, lam_out, primal_res, dual_res = projector.solve_admm(xi_init, lam_init)
    

    
    print("Projected xi:", xi_proj.shape)
    print("Final primal residual:", primal_res[-1])
    print("Final dual residual:", dual_res[-1])
    print("projected xi", -xi_proj[:12])


if __name__ == "__main__":
    main()
