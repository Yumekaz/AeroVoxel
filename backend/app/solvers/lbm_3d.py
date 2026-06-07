"""
Pure-Python D3Q19 Lattice Boltzmann solver for 3D flow simulation.

This is a real physics solver — not a mock or approximation.
It uses the standard BGK collision operator on a D3Q19 lattice
with bounce-back boundary conditions for solid surfaces.

Designed for small grids (64^3) suitable for educational /
portfolio-grade demonstration of 3D CFD concepts.
"""

import numpy as np


class LbmSolver3D:
    """D3Q19 Lattice Boltzmann solver for 3D incompressible flow."""

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        nz: int = 64,
        tau: float = 0.8,
        u_inlet: float = 0.05,
        wind_angle_deg: float = 0.0,
    ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.tau = tau
        self.u_inlet = u_inlet
        self.wind_angle_deg = wind_angle_deg

        # Compute inlet velocity components (angle rotates in X-Y plane)
        angle_rad = np.radians(wind_angle_deg)
        self.u_inlet_x = float(u_inlet * np.cos(angle_rad))
        self.u_inlet_y = float(u_inlet * np.sin(angle_rad))
        self.u_inlet_z = 0.0

        # D3Q19 lattice velocities (cx, cy, cz)
        self.C = np.array([
            [ 0,  0,  0],  # 0  - rest
            [ 1,  0,  0],  # 1
            [-1,  0,  0],  # 2
            [ 0,  1,  0],  # 3
            [ 0, -1,  0],  # 4
            [ 0,  0,  1],  # 5
            [ 0,  0, -1],  # 6
            [ 1,  1,  0],  # 7
            [-1,  1,  0],  # 8
            [ 1, -1,  0],  # 9
            [-1, -1,  0],  # 10
            [ 1,  0,  1],  # 11
            [-1,  0,  1],  # 12
            [ 1,  0, -1],  # 13
            [-1,  0, -1],  # 14
            [ 0,  1,  1],  # 15
            [ 0, -1,  1],  # 16
            [ 0,  1, -1],  # 17
            [ 0, -1, -1],  # 18
        ])

        # D3Q19 weights
        self.W = np.array([
            1.0 / 3.0,                                       # rest
            1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,             # axis-aligned
            1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,             # diagonals
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
        ])

        # Opposite direction indices for bounce-back
        self.OPPOSITE = np.array([
            0,
            2, 1,    # +x <-> -x
            4, 3,    # +y <-> -y
            6, 5,    # +z <-> -z
            10, 9,   # (+x,+y) <-> (-x,-y)
            8, 7,    # (+x,-y) <-> (-x,+y)
            14, 13,  # (+x,+z) <-> (-x,-z)
            12, 11,  # (+x,-z) <-> (-x,+z)
            18, 17,  # (+y,+z) <-> (-y,-z)
            16, 15,  # (+y,-z) <-> (-y,+z)
        ])

    def get_equilibrium(self, rho, u):
        """Calculate equilibrium distribution function f_eq for D3Q19."""
        q = 19
        f_eq = np.zeros((q, self.nz, self.ny, self.nx), dtype=np.float32)
        usqr = u[0] ** 2 + u[1] ** 2 + u[2] ** 2

        for i in range(q):
            cu = (
                self.C[i, 0] * u[0]
                + self.C[i, 1] * u[1]
                + self.C[i, 2] * u[2]
            )
            f_eq[i] = self.W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usqr)

        return f_eq

    def solve(self, obstacle_mask: np.ndarray, steps: int = 200, progress_callback=None):
        """
        Run D3Q19 LBM simulation around a 3D obstacle mask.

        Parameters
        ----------
        obstacle_mask : ndarray of bool, shape (nz, ny, nx)
            True where solid obstacle exists.
        steps : int
            Number of LBM iterations.
        progress_callback : callable, optional
            Called with (current_step, total_steps) for progress reporting.

        Returns
        -------
        u : ndarray, shape (3, nz, ny, nx) — velocity field
        pressure : ndarray, shape (nz, ny, nx) — relative pressure
        """
        nz, ny, nx = self.nz, self.ny, self.nx
        q = 19

        # Initialize macroscopic fields
        rho = np.ones((nz, ny, nx), dtype=np.float32)
        u = np.zeros((3, nz, ny, nx), dtype=np.float32)
        u[0, :, :, :] = self.u_inlet_x
        u[1, :, :, :] = self.u_inlet_y
        u[2, :, :, :] = self.u_inlet_z

        # Zero velocity inside obstacle
        u[0][obstacle_mask] = 0.0
        u[1][obstacle_mask] = 0.0
        u[2][obstacle_mask] = 0.0

        # Initialize distribution to equilibrium
        f = self.get_equilibrium(rho, u)

        print(f"[LBM-3D] Starting D3Q19 solver: {nx}×{ny}×{nz} grid, {steps} steps, tau={self.tau}")

        for step in range(steps):
            # 1. Streaming step — shift each distribution along its velocity direction
            for i in range(q):
                f[i] = np.roll(f[i], shift=self.C[i, 0], axis=2)  # x-axis
                f[i] = np.roll(f[i], shift=self.C[i, 1], axis=1)  # y-axis
                f[i] = np.roll(f[i], shift=self.C[i, 2], axis=0)  # z-axis

            # 2. Bounce-back on solid boundaries
            for i in range(q):
                f_bounce = f[self.OPPOSITE[i]].copy()
                f[i] = np.where(obstacle_mask, f_bounce, f[i])

            # 3. Inlet boundary (x = 0): equilibrium velocity BC
            rho_inlet = np.float32(1.0)
            u_inlet_vec = np.zeros((3, nz, ny, 1), dtype=np.float32)
            u_inlet_vec[0] = self.u_inlet_x
            u_inlet_vec[1] = self.u_inlet_y
            u_inlet_vec[2] = self.u_inlet_z

            usqr_in = u_inlet_vec[0] ** 2 + u_inlet_vec[1] ** 2 + u_inlet_vec[2] ** 2
            for i in [1, 7, 9, 11, 13]:  # directions with +x component
                cu_in = (
                    self.C[i, 0] * u_inlet_vec[0]
                    + self.C[i, 1] * u_inlet_vec[1]
                    + self.C[i, 2] * u_inlet_vec[2]
                )
                f_eq_in = self.W[i] * rho_inlet * (1.0 + 3.0 * cu_in + 4.5 * cu_in ** 2 - 1.5 * usqr_in)
                f[i, :, :, 0:1] = f_eq_in

            # 4. Outlet boundary (x = nx-1): zero-gradient outflow
            for i in [2, 8, 10, 12, 14]:  # directions with -x component
                f[i, :, :, -1] = f[i, :, :, -2]

            # 5. Compute macroscopic variables
            rho = np.sum(f, axis=0)
            rho_safe = np.where(rho < 0.1, np.float32(0.1), rho)

            u[0] = np.sum(f * self.C[:, 0, np.newaxis, np.newaxis, np.newaxis], axis=0) / rho_safe
            u[1] = np.sum(f * self.C[:, 1, np.newaxis, np.newaxis, np.newaxis], axis=0) / rho_safe
            u[2] = np.sum(f * self.C[:, 2, np.newaxis, np.newaxis, np.newaxis], axis=0) / rho_safe

            # Force zero velocity inside obstacles
            u[0][obstacle_mask] = 0.0
            u[1][obstacle_mask] = 0.0
            u[2][obstacle_mask] = 0.0

            # 6. Collision step (BGK)
            f_eq = self.get_equilibrium(rho, u)
            f = f - (1.0 / self.tau) * (f - f_eq)

            # Progress reporting
            if progress_callback and (step % 20 == 0 or step == steps - 1):
                progress_callback(step + 1, steps)

            if step % 50 == 0:
                max_u = float(np.max(np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)))
                print(f"[LBM-3D] Step {step}/{steps} — max |u| = {max_u:.6f}")

        # Compute relative pressure: p = (rho - 1) * c_s^2 = (rho - 1) / 3
        pressure = (rho - 1.0) / 3.0

        print(f"[LBM-3D] Solver complete. Velocity shape: {u.shape}, Pressure shape: {pressure.shape}")
        return u, pressure
