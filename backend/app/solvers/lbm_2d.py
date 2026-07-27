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
        self.W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
        self.OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Filled after solve(): educational force / Cd proxies (lattice units)
        self.last_force_lu = (0.0, 0.0)
        self.last_force_method = "none"

    def get_equilibrium(self, rho, u):
        """Calculate the equilibrium distribution function f_eq."""
        f_eq = np.zeros((9, self.ny, self.nx))
        usqr = u[0] ** 2 + u[1] ** 2
        for i in range(9):
            cu = self.C[i, 0] * u[0] + self.C[i, 1] * u[1]
            f_eq[i] = self.W[i] * rho * (1 + 3 * cu + 4.5 * cu ** 2 - 1.5 * usqr)
        return f_eq

    def _momentum_exchange_force_fluid_side(
        self, f: np.ndarray, obstacle_mask: np.ndarray
    ) -> tuple[float, float]:
        """Stationary bounce-back momentum exchange on fluid→solid links (educational).

        Uses post-collision populations at fluid nodes: for each lattice direction
        i that points into a solid neighbor, the force on the solid over one step is
        approximately 2 * f_i * c_i (rest-frame wall). Evaluated on the fluid side
        to avoid full-way solid-node double counting.

        This is an educational proxy, not a certified CFD force integration.
        """
        solid = obstacle_mask.astype(bool)
        fluid = ~solid
        fx = 0.0
        fy = 0.0
        for i in range(1, 9):
            cx = int(self.C[i, 0])
            cy = int(self.C[i, 1])
            # solid at x + c_i → roll(solid, -c) brings neighbor solid flag to x
            dest_solid = np.roll(solid, shift=(-cy, -cx), axis=(0, 1))
            links = fluid & dest_solid
            if not np.any(links):
                continue
            fi_sum = float(np.sum(f[i][links]))
            fx += 2.0 * fi_sum * cx
            fy += 2.0 * fi_sum * cy
        return fx, fy

    def solve(
        self,
        obstacle_mask: np.ndarray,
        steps: int = 600,
        force_avg_steps: int = 40,
    ):
        """Run D2Q9 LBM simulation around binary obstacle mask.

        Parameters
        ----------
        obstacle_mask : bool array (ny, nx)
            True = solid.
        steps : int
            Number of LBM timesteps.
        force_avg_steps : int
            Average momentum-exchange force over the last N steps (0 disables).

        Returns
        -------
        u : ndarray (2, ny, nx)
        pressure : ndarray (ny, nx)
            Relative pressure (rho - 1) / 3.

        Side effects
        ------------
        Sets ``self.last_force_lu`` = (Fx, Fy) in lattice units (time-averaged
        momentum exchange over the final ``force_avg_steps``), and
        ``self.last_force_method`` describing the proxy.
        """
        obstacle_mask = obstacle_mask.astype(bool)

        # Initialize density and velocity
        rho = np.ones((self.ny, self.nx))
        u = np.zeros((2, self.ny, self.nx))
        u[0, :, :] = self.u_inlet_x
        u[1, :, :] = self.u_inlet_y

        # Initialize f to equilibrium
        f = self.get_equilibrium(rho, u)

        force_acc = np.zeros(2, dtype=np.float64)
        force_count = 0
        avg_start = max(0, steps - max(0, int(force_avg_steps)))

        # Simulation loop
        for step in range(steps):
            # 1. Streaming
            for i in range(9):
                f[i] = np.roll(f[i], shift=(self.C[i, 1], self.C[i, 0]), axis=(0, 1))

            # 2. Boundary conditions
            # Bounce-back on obstacles (full-way BB)
            # Snapshot opposites first so updates are simultaneous
            f_pre = f.copy()
            for i in range(9):
                f[i] = np.where(obstacle_mask, f_pre[self.OPPOSITE[i]], f[i])

            # Inlet boundary condition (Zou-He style velocity BC at x = 0)
            rho_inlet = 1.0
            u_inlet_vec = np.array([self.u_inlet_x, self.u_inlet_y])
            f_eq_inlet = self.get_equilibrium(rho_inlet, u_inlet_vec)
            for i in [1, 5, 8]:
                f[i, :, 0] = f_eq_inlet[i, :, 0]

            # Outlet boundary condition (outflow zero gradient at x = nx-1)
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

            # 5. Momentum exchange on post-collision fluid→solid links
            if force_avg_steps > 0 and step >= avg_start:
                fx, fy = self._momentum_exchange_force_fluid_side(f, obstacle_mask)
                force_acc[0] += fx
                force_acc[1] += fy
                force_count += 1

        # Relative pressure: p = rho * c_s^2 = rho / 3
        pressure = (rho - 1.0) / 3.0

        if force_count > 0:
            self.last_force_lu = (
                float(force_acc[0] / force_count),
                float(force_acc[1] / force_count),
            )
            self.last_force_method = (
                "momentum_exchange_fluid_side_bb_avg_last_"
                f"{force_count}_steps"
            )
        else:
            self.last_force_lu = (0.0, 0.0)
            self.last_force_method = "disabled"

        return u, pressure


def surface_pressure_force_2d(pressure: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Discrete pressure surface integral on the solid boundary (educational).

    For each solid–fluid face, force on the body is ``-p * n * dA`` with unit
    lattice spacing and ``n`` the outward normal of the solid (into fluid).
    Uses face-adjacent fluid pressure only (no viscous stress).

    Returns (Fx, Fy) in lattice units. Not certified CFD.
    """
    solid = mask.astype(bool)
    p = np.asarray(pressure, dtype=np.float64)
    fx = 0.0
    fy = 0.0

    # Right-facing solid faces: solid[y,x] | fluid[y,x+1], n = (+1, 0)
    face = solid[:, :-1] & (~solid[:, 1:])
    fx += -float(np.sum(p[:, 1:][face]))

    # Left-facing: solid[y,x] | fluid[y,x-1], n = (-1, 0)
    face = solid[:, 1:] & (~solid[:, :-1])
    fx += -float(np.sum(p[:, :-1][face])) * (-1.0)

    # Down-facing in array y+ : solid[y,x] | fluid[y+1,x], n = (0, +1)
    face = solid[:-1, :] & (~solid[1:, :])
    fy += -float(np.sum(p[1:, :][face]))

    # Up-facing in array y- : solid[y,x] | fluid[y-1,x], n = (0, -1)
    face = solid[1:, :] & (~solid[:-1, :])
    fy += -float(np.sum(p[:-1, :][face])) * (-1.0)

    return fx, fy


def surface_viscous_force_proxy_2d(
    velocity: np.ndarray,
    mask: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    """Rough viscous traction proxy from near-wall velocity gradient (educational).

    Uses μ = ρ ν with ρ≈1, ν = (τ−0.5)/3, and ∂u/∂n estimated by one-sided
    difference from the adjacent fluid cell. Highly approximate on coarse grids.
    """
    solid = mask.astype(bool)
    if velocity.ndim == 3 and velocity.shape[0] >= 2:
        ux = np.asarray(velocity[0], dtype=np.float64)
        uy = np.asarray(velocity[1], dtype=np.float64)
    else:
        ux = np.asarray(velocity, dtype=np.float64)
        uy = np.zeros_like(ux)

    nu = max((tau - 0.5) / 3.0, 1e-12)
    mu = nu  # rho ≈ 1
    fx = 0.0
    fy = 0.0

    # For each face, estimate tangential shear ~ mu * du_t / dn (dn = 1 LU)
    # Right face n=(1,0): tangential vel is uy, stress on body ~ -mu * d(uy)/dn for y-force
    # and streamwise viscous contribution from d(ux)/dn
    face = solid[:, :-1] & (~solid[:, 1:])
    if np.any(face):
        # fluid just outside solid along +x
        dux_dn = ux[:, 1:][face]  # solid vel=0, fluid ux ≈ du/dn * 1
        duy_dn = uy[:, 1:][face]
        # traction on body ≈ -sigma · n; viscous sigma_xx ~ 2 mu dux/dx ≈ 2 mu dux_dn
        fx += -float(np.sum(2.0 * mu * dux_dn))
        fy += -float(np.sum(mu * duy_dn))

    face = solid[:, 1:] & (~solid[:, :-1])
    if np.any(face):
        dux_dn = ux[:, :-1][face]  # n = -x, outward; gradient along -n uses fluid at x-1
        duy_dn = uy[:, :-1][face]
        # n = (-1,0); -sigma·n x-comp
        fx += float(np.sum(2.0 * mu * dux_dn))
        fy += float(np.sum(mu * duy_dn))

    face = solid[:-1, :] & (~solid[1:, :])
    if np.any(face):
        dux_dn = ux[1:, :][face]
        duy_dn = uy[1:, :][face]
        fx += -float(np.sum(mu * dux_dn))
        fy += -float(np.sum(2.0 * mu * duy_dn))

    face = solid[1:, :] & (~solid[:-1, :])
    if np.any(face):
        dux_dn = ux[:-1, :][face]
        duy_dn = uy[:-1, :][face]
        fx += float(np.sum(mu * dux_dn))
        fy += float(np.sum(2.0 * mu * duy_dn))

    return fx, fy


def force_to_cd(
    force_x: float,
    u_ref: float,
    char_length: float,
    rho: float = 1.0,
) -> float:
    """Convert streamwise force (LU) to a 2D drag coefficient proxy.

    Cd = 2 * Fx / (ρ U² L) with L = characteristic length (e.g. diameter).
    Educational only — not a certified Cd.
    """
    denom = rho * (max(abs(u_ref), 1e-12) ** 2) * max(char_length, 1e-12)
    return float(2.0 * force_x / denom)


def educational_force_metrics(
    pressure: np.ndarray,
    mask: np.ndarray,
    u_ref: float,
    char_length: float,
    velocity: np.ndarray | None = None,
    tau: float = 0.6,
    momentum_force_lu: tuple[float, float] | None = None,
    momentum_method: str | None = None,
) -> dict:
    """Bundle educational force / Cd proxies from available fields and optional MEM.

    Labels:
      - cd_force_proxy: preferred force-based Cd (MEM if available, else surface)
      - cd_heuristic: wake-area formula is applied by the caller separately
    """
    fx_p, fy_p = surface_pressure_force_2d(pressure, mask)
    fx_v, fy_v = (0.0, 0.0)
    if velocity is not None:
        fx_v, fy_v = surface_viscous_force_proxy_2d(velocity, mask, tau=tau)

    fx_surf = fx_p + fx_v
    fy_surf = fy_p + fy_v
    cd_surf = force_to_cd(fx_surf, u_ref, char_length)

    out = {
        "force_x_pressure_lu": fx_p,
        "force_y_pressure_lu": fy_p,
        "force_x_viscous_proxy_lu": fx_v,
        "force_y_viscous_proxy_lu": fy_v,
        "force_x_surface_lu": fx_surf,
        "force_y_surface_lu": fy_surf,
        "cd_surface_proxy": cd_surf,
        "cd_force_proxy_method": "surface_pressure_plus_viscous_proxy",
        "label": "educational force proxy — not certified CFD",
    }

    # Primary educational proxy: discrete surface pressure + rough viscous traction.
    # Full-way bounce-back MEM is reported as a secondary estimate; on this coarse
    # D2Q9 implementation it often overshoots surface Cd, so it is not the default.
    out["cd_force_proxy"] = cd_surf
    out["force_x_lu"] = fx_surf
    out["force_y_lu"] = fy_surf
    out["cd_force_proxy_method"] = "surface_pressure_plus_viscous_proxy"

    if momentum_force_lu is not None:
        fx_m, fy_m = float(momentum_force_lu[0]), float(momentum_force_lu[1])
        cd_m = force_to_cd(fx_m, u_ref, char_length)
        out["force_x_momentum_exchange_lu"] = fx_m
        out["force_y_momentum_exchange_lu"] = fy_m
        out["cd_momentum_exchange_proxy"] = cd_m
        out["momentum_exchange_method"] = momentum_method or "momentum_exchange_bb"
        out["cd_force_proxy_note"] = (
            "Primary cd_force_proxy is surface integral; "
            "cd_momentum_exchange_proxy is secondary (full-way BB MEM, educational)."
        )

    return out
