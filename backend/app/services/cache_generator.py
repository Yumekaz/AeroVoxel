import os
import numpy as np

# Grid dimensions for precomputed cases
NX = 128
NY = 64
STEPS = 800  # Number of LBM iterations for convergence
TAU = 0.6    # Relaxation time (viscosity controller)
U_INLET = 0.08  # Inlet velocity in lattice units

# D2Q9 constants
C = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Opposite directions for bounce-back
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def get_equilibrium(rho, u):
    """Calculate the equilibrium distribution function f_eq."""
    f_eq = np.zeros((9, NY, NX))
    usqr = u[0]**2 + u[1]**2
    for i in range(9):
        cu = C[i, 0] * u[0] + C[i, 1] * u[1]
        f_eq[i] = W[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usqr)
    return f_eq

def run_lbm_simulation(obstacle_mask):
    """Run D2Q9 LBM simulation around binary obstacle mask."""
    # Initialize density and velocity
    rho = np.ones((NY, NX))
    u = np.zeros((2, NY, NX))
    u[0, :, :] = U_INLET  # Set uniform inlet velocity

    # Initialize f to equilibrium
    f = get_equilibrium(rho, u)

    # Simulation loop
    for _ in range(STEPS):
        # 1. Streaming (shift distributions along lattice velocities)
        for i in range(9):
            f[i] = np.roll(f[i], shift=(C[i, 1], C[i, 0]), axis=(0, 1))

        # 2. Boundary conditions
        # Bounce-back on obstacle boundary
        for i in range(9):
            f_bounce = f[OPPOSITE[i]]
            f[i] = np.where(obstacle_mask, f_bounce, f[i])

        # Inlet (Zou-He boundary condition for velocity boundary x = 0)
        # Simply reset inlet distributions to equilibrium for inlet velocity
        rho_inlet = 1.0  # approximate
        u_inlet = np.array([U_INLET, 0.0])
        f_eq_inlet = get_equilibrium(rho_inlet, u_inlet)
        for i in [1, 5, 8]: # directions pointing into domain from left
            f[i, :, 0] = f_eq_inlet[i, :, 0]

        # Outlet (zero gradient outflow x = NX-1)
        for i in [3, 6, 7]: # directions pointing out of domain to left (from outlet)
            f[i, :, -1] = f[i, :, -2]

        # 3. Macro variables calculation
        rho = np.sum(f, axis=0)
        
        # Prevent division by zero
        rho_safe = np.where(rho < 0.1, 0.1, rho)
        
        u[0] = np.sum(f * C[:, 0][:, np.newaxis, np.newaxis], axis=0) / rho_safe
        u[1] = np.sum(f * C[:, 1][:, np.newaxis, np.newaxis], axis=0) / rho_safe

        # Enforce zero velocity inside obstacle
        u[0] = np.where(obstacle_mask, 0, u[0])
        u[1] = np.where(obstacle_mask, 0, u[1])

        # 4. Collision step
        f_eq = get_equilibrium(rho, u)
        f = f - (1 / TAU) * (f - f_eq)

    # Converged pressure (p = rho * c_s^2 where c_s^2 = 1/3)
    pressure = (rho - 1.0) / 3.0  # Relative pressure difference
    
    return u, pressure

# Shape definitions
def make_car_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    # Ground wall
    # Car is at the bottom (y close to NY-1 in standard array index, but let's make y range 45 to 60)
    for x in range(NX):
        for y in range(NY):
            # Car body: length from 40 to 80, height from 48 to 60 (bottom of domain is y=NY-1)
            # Cabin: length from 50 to 72, height from 38 to 48
            if (40 <= x <= 80 and 48 <= y <= 60):
                # Slanted hood: x from 40 to 50
                if x < 50:
                    slope_height = int(48 + (50 - x) * 1.2)
                    if y >= slope_height:
                        mask[y, x] = True
                # Slanted windshield/rear: x from 72 to 80
                elif x > 72:
                    slope_height = int(48 + (x - 72) * 1.5)
                    if y >= slope_height:
                        mask[y, x] = True
                else:
                    mask[y, x] = True
            elif (50 <= x <= 72 and 38 <= y <= 48):
                # Windshield slant front: x from 50 to 56
                if x < 56:
                    slope_height = int(38 + (56 - x) * 1.6)
                    if y >= slope_height:
                        mask[y, x] = True
                # Rear window slant: x from 66 to 72
                elif x > 66:
                    slope_height = int(38 + (x - 66) * 1.6)
                    if y >= slope_height:
                        mask[y, x] = True
                else:
                    mask[y, x] = True
    return mask

def make_drone_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    # Center body: circle at (64, 32) with radius 8
    # Left arm: (44 to 64, 30 to 34)
    # Right arm: (64 to 84, 30 to 34)
    # Left rotor center: (44, 25 to 39)
    # Right rotor center: (84, 25 to 39)
    cx, cy = 64, 32
    for x in range(NX):
        for y in range(NY):
            # Center body
            if (x - cx)**2 + (y - cy)**2 <= 9**2:
                mask[y, x] = True
            # Left & Right arms
            if 36 <= x <= 92 and 30 <= y <= 34:
                mask[y, x] = True
            # Rotors
            if (x - 36)**2 + (y - cy)**2 <= 4**2:
                mask[y, x] = True
            if (x - 92)**2 + (y - cy)**2 <= 4**2:
                mask[y, x] = True
    return mask

def make_airfoil_mask():
    mask = np.zeros((NY, NX), dtype=bool)
    # Tilted NACA 0012 airfoil profile center (60, 32)
    # Chord length = 40
    # Thickness = t = 0.12
    # Profile formula: y = +/- 5 * t * (0.2969*sqrt(x) - 0.126*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
    # Let's rotate by alpha = 8 degrees (0.14 radians)
    cx, cy = 60, 32
    chord = 36.0
    alpha = np.radians(8.0)
    
    for x in range(NX):
        for y in range(NY):
            # Translate to chord origin (leading edge at 42, 32)
            dx = x - 42
            dy = y - cy
            
            # Rotate back by alpha to align with profile coordinate system
            rx = dx * np.cos(alpha) + dy * np.sin(alpha)
            ry = -dx * np.sin(alpha) + dy * np.cos(alpha)
            
            # Check if inside airfoil boundary
            if 0 <= rx <= chord:
                x_frac = rx / chord
                # NACA 0012 thickness equation
                half_thickness = chord * 5.0 * 0.12 * (
                    0.2969 * np.sqrt(x_frac) - 
                    0.1260 * x_frac - 
                    0.3516 * (x_frac**2) + 
                    0.2843 * (x_frac**3) - 
                    0.1015 * (x_frac**4)
                )
                if abs(ry) <= half_thickness:
                    mask[y, x] = True
    return mask

def generate_and_save_caches():
    """Generate flow fields and save to assets folder."""
    # Define output directories
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "flow"))
    os.makedirs(assets_dir, exist_ok=True)
    
    print(f"Generating aerodynamic caches in: {assets_dir}")
    
    presets = [
        ("sports_car_v1", make_car_mask()),
        ("drone_v1", make_drone_mask()),
        ("airfoil_v1", make_airfoil_mask())
    ]
    
    for name, mask in presets:
        print(f"Running LBM solver for preset: {name}...")
        u, pressure = run_lbm_simulation(mask)
        
        # Save files
        u_path = os.path.join(assets_dir, f"{name}_velocity.npy")
        p_path = os.path.join(assets_dir, f"{name}_pressure.npy")
        m_path = os.path.join(assets_dir, f"{name}_mask.npy")
        
        np.save(u_path, u.astype(np.float32))
        np.save(p_path, pressure.astype(np.float32))
        np.save(m_path, mask.astype(bool))
        
        print(f"Saved: {name} arrays (Velocity shape: {u.shape}, Pressure shape: {pressure.shape})")

if __name__ == "__main__":
    generate_and_save_caches()
