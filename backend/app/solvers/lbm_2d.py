import numpy as np

class LbmSolver2D:
    def __init__(
        self,
        nx: int = 128,
        ny: int = 64,
        tau: float = 0.6,
        u_inlet: float = 0.08,
        wind_angle_deg: float = 0.0,
    ):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.u_inlet = u_inlet
        self.wind_angle_deg = wind_angle_deg
        angle_rad = np.radians(wind_angle_deg)
        self.u_inlet_x = float(u_inlet * np.cos(angle_rad))
        self.u_inlet_y = float(u_inlet * np.sin(angle_rad))

        # D2Q9 constants
        self.C = np.array([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [-1, 1], [-1, -1], [1, -1]
        ])
        self.W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    def get_equilibrium(self, rho, u):
        """Calculate the equilibrium distribution function f_eq."""
        f_eq = np.zeros((9, self.ny, self.nx))
        usqr = u[0]**2 + u[1]**2
        for i in range(9):
            cu = self.C[i, 0] * u[0] + self.C[i, 1] * u[1]
            f_eq[i] = self.W[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr)
        return f_eq

    def solve(self, obstacle_mask: np.ndarray, steps: int = 600):
        """Run D2Q9 LBM simulation around binary obstacle mask."""
        # Initialize density and velocity
        rho = np.ones((self.ny, self.nx))
        u = np.zeros((2, self.ny, self.nx))
        u[0, :, :] = self.u_inlet_x
        u[1, :, :] = self.u_inlet_y

        # Initialize f to equilibrium
        f = self.get_equilibrium(rho, u)

        # Simulation loop
        for _ in range(steps):
            # 1. Streaming
            for i in range(9):
                f[i] = np.roll(f[i], shift=(self.C[i, 1], self.C[i, 0]), axis=(0, 1))

            # 2. Boundary conditions
            # Bounce-back on obstacles
            for i in range(9):
                f_bounce = f[self.OPPOSITE[i]]
                f[i] = np.where(obstacle_mask, f_bounce, f[i])

            # Inlet boundary condition (Zou-He velocity boundary at x = 0)
            rho_inlet = 1.0
            u_inlet_vec = np.array([self.u_inlet_x, self.u_inlet_y])
            f_eq_inlet = self.get_equilibrium(rho_inlet, u_inlet_vec)
            for i in [1, 5, 8]:
                f[i, :, 0] = f_eq_inlet[i, :, 0]

            # Outlet boundary condition (Outflow zero gradient at x = nx-1)
            for i in [3, 6, 7]:
                f[i, :, -1] = f[i, :, -2]

            # 3. Compute macro variables
            rho = np.sum(f, axis=0)
            
            # Prevent division by zero
            rho_safe = np.where(rho < 0.1, 0.1, rho)
            
            u[0] = np.sum(f * self.C[:, 0][:, np.newaxis, np.newaxis], axis=0) / rho_safe
            u[1] = np.sum(f * self.C[:, 1][:, np.newaxis, np.newaxis], axis=0) / rho_safe

            # Force zero velocity inside obstacles
            u[0] = np.where(obstacle_mask, 0.0, u[0])
            u[1] = np.where(obstacle_mask, 0.0, u[1])

            # 4. Collision step
            f_eq = self.get_equilibrium(rho, u)
            f = f - (1 / self.tau) * (f - f_eq)

        # Relative pressure: p = rho * c_s^2 = rho / 3
        pressure = (rho - 1.0) / 3.0
        
        return u, pressure
