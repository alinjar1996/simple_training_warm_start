import numpy as np
#from qpsolvers import solve_qp


class QPSolver():

    def __init__(self, L_i, K_i, planning_horizon):

        self.L = None
        self.K = None
        self.L_i = L_i
        self.K_i = K_i
        self.planning_horizon = planning_horizon
        self.opts = {'MAXITER':30,'VERBOSE':0,'OUTPUT':2}


    def _createFullCostMatrices(self, decay=0.99):

        n = self.L_i.shape[0]
        m = self.K_i.shape[0]
        self.L = np.zeros((n*self.planning_horizon, n*self.planning_horizon))
        self.K = np.zeros((m*self.planning_horizon, m*self.planning_horizon))

        for i in range(self.planning_horizon):
            self.L[i*n:(i+1)*n, i*n:(i+1)*n] = self.L_i*(decay**i)
            self.K[i*m:(i+1)*m, i*m:(i+1)*m] = self.K_i*(decay**i)

        # L = self.L
        # K = self.K    

        # print("Full cost Matrix printed")    
        # return L, K
   


    # def solveQP(self, A_qp, B_qp, x_init, x_ref, force_constraints):

    #     self._createFullCostMatrices()

    #     # Cost function
    #     H = 2*(B_qp.T@self.L@B_qp + self.K)
    #     g = 2*B_qp.T@self.L@(A_qp@x_init - x_ref)

    #     # Inequality constraints
    #     c, C = force_constraints
        
    #     res = solve_qp(H, g, C, c, solver="clarabel", verbose=False)
    #     # res = solve_qp.run(g, c, H, C, opts=self.opts)

    #     return np.array(res)