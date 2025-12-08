"""
Ionospheric HF Radio Wave Propagation Physics Model

This module contains the physics functions for modeling ionospheric propagation
of HF (High Frequency) radio waves. It is used by the condx.py marimo notebook.

Physical model components:
- Chapman layer electron density profiles
- Four-layer ionosphere model (D, E, F1, F2)
- Plasma physics functions (plasma frequency, collision frequency)
- Complex refractive index calculation (Appleton-Hartree equation)
- Spherical ray tracing with absorption
- 2D ionosphere with horizontal gradients (tilts)
"""

import numpy as np

# Physical constants
R_E = 6371.0  # Earth radius in km
c = 3.0e8     # Speed of light in m/s
pi = np.pi

# ============================================================================
# CHAPMAN LAYER FUNCTION
# ============================================================================

def chapman_layer(z, Ne_max, h_m, H):
    """
    Calculate electron density profile for a single ionospheric layer
    using the Chapman function.
    
    Parameters:
    -----------
    z : array-like
        Altitude in km
    Ne_max : float
        Peak electron density in electrons/m³
    h_m : float
        Peak altitude in km
    H : float
        Scale height in km
    
    Returns:
    --------
    Ne : ndarray
        Electron density at each altitude in electrons/m³
    """
    xi = (z - h_m) / H
    Ne = Ne_max * np.exp(0.5 * (1 - xi - np.exp(-xi)))
    return Ne


# ============================================================================
# FOUR-LAYER IONOSPHERE MODEL
# ============================================================================

def make_ionosphere_for_foF2(foF2_MHz):
    """
    Create a four-layer ionosphere model (D, E, F1, F2) based on foF2 value.
    
    Parameters:
    -----------
    foF2_MHz : float
        Critical frequency of F2 layer in MHz
    
    Returns:
    --------
    electron_density_func : function
        Function that takes altitude (km) and returns total electron density (electrons/m³)
    """
    # Calculate F2 peak density from foF2
    foF2_Hz = foF2_MHz * 1e6
    Ne_peak_cm3 = (foF2_Hz / 8.98e3) ** 2
    Ne_peak_m3 = Ne_peak_cm3 * 1e6
    
    # F2 altitude varies with foF2 (higher foF2 → lower altitude)
    # Night: ~420 km, Day: ~260 km
    h_F2 = 420.0 - (foF2_MHz - 2.0) * 7.0
    h_F2 = max(250.0, min(420.0, h_F2))
    
    # D layer strength (present during day, absent at night)
    if foF2_MHz < 5.0:
        D_factor = 0.0  # Night: no D layer
    elif foF2_MHz < 8.0:
        D_factor = (foF2_MHz - 5.0) / 3.0  # Dawn/dusk transition
    else:
        D_factor = 1.0  # Day: full D layer
    
    Ne_D_max = 1.5e9 * D_factor  # Calibrated to match real-world absorption
    
    # F1 layer strength (daytime only)
    if foF2_MHz < 4.5:
        F1_factor = 0.0
    elif foF2_MHz < 7.0:
        F1_factor = (foF2_MHz - 4.5) / 2.5
    else:
        F1_factor = 1.0
    
    Ne_F1_max = 3e11 * F1_factor
    
    # E layer strength (stronger during day)
    E_factor = 0.5 + 0.5 * min(1.0, foF2_MHz / 12.0)
    Ne_E_max = 8e10 * E_factor
    
    # Layer parameters
    # D layer: 75 km peak, 8 km scale height
    # E layer: 110 km peak, 15 km scale height
    # F1 layer: 190 km peak, 30 km scale height
    # F2 layer: variable peak altitude, 55 km scale height
    
    def electron_density(z):
        """Total electron density at altitude z (km)"""
        Ne_D = chapman_layer(z, Ne_D_max, 75.0, 8.0)
        Ne_E = chapman_layer(z, Ne_E_max, 110.0, 15.0)
        Ne_F1 = chapman_layer(z, Ne_F1_max, 190.0, 30.0)
        Ne_F2 = chapman_layer(z, Ne_peak_m3, h_F2, 55.0)
        return Ne_D + Ne_E + Ne_F1 + Ne_F2
    
    return electron_density


# ============================================================================
# 2D TILTED IONOSPHERE MODEL
# ============================================================================

def make_tilted_ionosphere_2D(foF2_start_MHz, foF2_end_MHz, tilt_distance_km=3000.0):
    """
    Create a 2D ionosphere with horizontal foF2 gradient.
    
    This models ionospheric tilts such as day/night terminator effects,
    where electron density varies along the propagation path.
    
    Parameters:
    -----------
    foF2_start_MHz : float
        Critical frequency at transmitter location (MHz)
    foF2_end_MHz : float
        Critical frequency at receiver location (MHz)
    tilt_distance_km : float
        Distance over which gradient occurs (default 3000 km)
    
    Returns:
    --------
    electron_density_func : function
        Function that takes (altitude_km, distance_km) and returns electron density (electrons/m³)
    """
    def electron_density_2D(z, x):
        """
        2D electron density function
        
        Parameters:
        -----------
        z : float or array
            Altitude in km
        x : float or array
            Horizontal distance from transmitter in km
        
        Returns:
        --------
        Ne : float or array
            Electron density in electrons/m³
        """
        # Linear interpolation of foF2 with distance
        foF2_local = foF2_start_MHz + (foF2_end_MHz - foF2_start_MHz) * (x / tilt_distance_km)
        foF2_local = np.clip(foF2_local, min(foF2_start_MHz, foF2_end_MHz), 
                            max(foF2_start_MHz, foF2_end_MHz))
        
        # Build ionosphere for local foF2
        local_ionosphere = make_ionosphere_for_foF2(foF2_local)
        return local_ionosphere(z)
    
    return electron_density_2D


# ============================================================================
# PLASMA PHYSICS FUNCTIONS
# ============================================================================

def plasma_frequency_Hz_from_Ne(Ne):
    """
    Calculate plasma frequency from electron density.
    
    Parameters:
    -----------
    Ne : array-like
        Electron density in electrons/m³
    
    Returns:
    --------
    f_p : ndarray
        Plasma frequency in Hz
    """
    # f_p = 8.98 × 10³ × sqrt(Ne) where Ne is in electrons/cm³
    # Convert Ne from m⁻³ to cm⁻³
    Ne_cm3 = Ne / 1e6
    f_p = 8.98e3 * np.sqrt(Ne_cm3)
    return f_p


def collision_frequency_Hz(z):
    """
    Calculate electron-neutral collision frequency at altitude z.
    
    This calibrated profile matches real-world absorption measurements.
    
    Parameters:
    -----------
    z : array-like
        Altitude in km
    
    Returns:
    --------
    nu : ndarray
        Collision frequency in Hz
    """
    z = np.asarray(z)
    nu = np.zeros_like(z, dtype=float)
    
    # D-layer region (< 90 km): High collision rate
    mask_D = z < 90.0
    nu[mask_D] = 3.5e5 * np.exp(-(z[mask_D] - 70.0) / 8.0)
    
    # E/F region (90-150 km): Rapidly decreasing collisions
    mask_E = (z >= 90.0) & (z < 150.0)
    nu[mask_E] = 1e5 * np.exp(-(z[mask_E] - 90.0) / 20.0)
    
    # High altitude (> 150 km): Negligible collisions
    mask_F = z >= 150.0
    nu[mask_F] = 1e3
    
    return nu


def make_refractive_index_func(electron_density_func, is_2D=False):
    """
    Create a refractive index function using the Appleton-Hartree equation.
    
    Parameters:
    -----------
    electron_density_func : function
        Function that returns electron density (electrons/m³)
        - 1D mode: electron_density_func(z) 
        - 2D mode: electron_density_func(z, x)
    is_2D : bool
        If True, electron_density_func expects (z, x) arguments
        If False, electron_density_func expects (z) argument only
    
    Returns:
    --------
    refractive_index : function
        Function that takes (frequency_Hz, altitude_km, [distance_km]) 
        and returns complex refractive index
    """
    def refractive_index(f_Hz, z, x=0.0):
        """
        Complex refractive index from Appleton-Hartree equation.
        
        Parameters:
        -----------
        f_Hz : float
            Radio frequency in Hz
        z : float or array
            Altitude in km
        x : float or array
            Horizontal distance in km (only used in 2D mode)
        
        Returns:
        --------
        n : complex or array of complex
            Complex refractive index (dimensionless)
        """
        # Get electron density
        if is_2D:
            Ne = electron_density_func(z, x)
        else:
            Ne = electron_density_func(z)
        
        # Calculate plasma frequency
        f_p = plasma_frequency_Hz_from_Ne(Ne)
        
        # Calculate collision frequency
        nu = collision_frequency_Hz(z)
        
        # Normalized parameters
        omega = 2 * pi * f_Hz
        omega_p = 2 * pi * f_p
        X = (omega_p / omega) ** 2
        Z = nu / omega
        
        # Appleton-Hartree equation: n² = 1 - X/(1 - jZ)
        n_squared = 1.0 - X / (1.0 - 1j * Z)
        
        # Take square root with correct branch
        # (positive real part for physical solution)
        n = np.sqrt(n_squared + 0j)
        
        # Ensure positive real part
        n = np.where(np.real(n) < 0, -n, n)
        
        return n
    
    return refractive_index


# ============================================================================
# SPHERICAL RAY TRACING
# ============================================================================

def trace_ray_spherical_with_path(f_MHz, elevation_deg, refractive_index_func, 
                                   max_distance_km=10000.0, step_km=2.0):
    """
    Trace a radio wave ray through the ionosphere in spherical geometry.
    Records the complete path for visualization.
    
    Parameters:
    -----------
    f_MHz : float
        Radio frequency in MHz
    elevation_deg : float
        Launch elevation angle in degrees (above horizon)
    refractive_index_func : function
        Function that returns complex refractive index: n = f(freq_Hz, altitude_km)
    max_distance_km : float
        Maximum surface distance to trace (default 10000 km)
    step_km : float
        Step size for ray tracing (default 2 km)
    
    Returns:
    --------
    x_path : ndarray
        Surface distance along ray path (km)
    z_path : ndarray
        Altitude along ray path (km)
    status : str
        "returns" if ray returns to Earth
        "escapes" if ray passes through ionosphere
        "stops" if ray is absorbed or reaches max distance
    """
    f_Hz = f_MHz * 1e6
    psi_0 = elevation_deg * pi / 180.0
    
    # Initial conditions at Earth's surface
    r = R_E
    theta = 0.0
    n_0 = np.real(refractive_index_func(f_Hz, 0.0))
    b = n_0 * r * np.sin(psi_0)  # Ray parameter (conserved)
    
    # Storage for path
    x_path = [0.0]
    z_path = [0.0]
    
    # Ray direction tracking
    going_up = True
    
    # Ray tracing loop
    for _ in range(int(max_distance_km / step_km)):
        z = r - R_E
        
        # Check if returned to ground
        if z < 0 and len(x_path) > 10:
            return np.array(x_path), np.array(z_path), "returns"
        
        # Get refractive index at current position
        n = refractive_index_func(f_Hz, z)
        n_real = np.real(n)
        n_squared = n * n
        n2_real = np.real(n_squared)
        
        # Check if ray is reflected (n² < 0 means total reflection)
        if n2_real <= 0:
            going_up = False
        
        # Calculate ray angle from ray parameter conservation
        sin_psi = b / (n_real * r)
        
        # Check if ray reflects (sin > 1) or escapes (too high)
        if abs(sin_psi) >= 1.0:
            if z > 600 and going_up:
                return np.array(x_path), np.array(z_path), "escapes"
            else:
                # Reflection point - start coming back down
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999
        
        if z > 800 and going_up:
            return np.array(x_path), np.array(z_path), "escapes"
        
        cos_psi = np.sqrt(1 - sin_psi**2)
        
        # Step along ray (with direction: up or down)
        dr = step_km * cos_psi * (1 if going_up else -1)
        d_theta = step_km * sin_psi / r
        
        r += dr
        theta += d_theta
        
        x_path.append(R_E * theta)
        z_path.append(r - R_E)
    
    return np.array(x_path), np.array(z_path), "stops"


def trace_ray_with_absorption(f_MHz, elevation_deg, refractive_index_func,
                              max_distance_km=10000.0, step_km=2.0):
    """
    Trace a radio wave ray and calculate absorption loss.
    
    Uses the Sen-Wyller absorption formula to calculate path loss
    due to electron-neutral collisions in the ionosphere.
    
    Parameters:
    -----------
    f_MHz : float
        Radio frequency in MHz
    elevation_deg : float
        Launch elevation angle in degrees (above horizon)
    refractive_index_func : function
        Function that returns complex refractive index: n = f(freq_Hz, altitude_km)
    max_distance_km : float
        Maximum surface distance to trace (default 10000 km)
    step_km : float
        Step size for ray tracing (default 2 km)
    
    Returns:
    --------
    x_path : ndarray
        Surface distance along ray path (km)
    z_path : ndarray
        Altitude along ray path (km)
    status : str
        "returns", "escapes", or "stops"
    total_loss_dB : float
        Total absorption loss along path in dB
    """
    f_Hz = f_MHz * 1e6
    omega = 2 * pi * f_Hz
    psi_0 = elevation_deg * pi / 180.0
    
    # Initial conditions
    r = R_E
    theta = 0.0
    n_0 = np.real(refractive_index_func(f_Hz, 0.0))
    b = n_0 * r * np.sin(psi_0)
    
    # Storage
    x_path = [0.0]
    z_path = [0.0]
    total_loss_nepers = 0.0
    
    # Ray direction tracking
    going_up = True
    
    for _ in range(int(max_distance_km / step_km)):
        z = r - R_E
        
        if z < 0 and len(x_path) > 10:
            total_loss_dB = total_loss_nepers * 8.686  # Convert Nepers to dB
            return np.array(x_path), np.array(z_path), "returns", total_loss_dB
        
        # Get complex refractive index
        n = refractive_index_func(f_Hz, z)
        n_real = np.real(n)
        n_squared = n * n
        n2_real = np.real(n_squared)
        
        # Check if ray is reflected
        if n2_real <= 0:
            going_up = False
        
        # Calculate absorption coefficient (Sen-Wyller formula)
        # α = (ω/2c) × |Im(n²)| / Re(n)
        alpha = (omega / (2 * c)) * abs(np.imag(n_squared)) / max(n_real, 1e-10)
        
        # Accumulate loss (in Nepers)
        path_length_m = step_km * 1000.0
        total_loss_nepers += alpha * path_length_m
        
        # Ray geometry
        sin_psi = b / (n_real * r)
        
        if abs(sin_psi) >= 1.0:
            if z > 600 and going_up:
                total_loss_dB = total_loss_nepers * 8.686
                return np.array(x_path), np.array(z_path), "escapes", total_loss_dB
            else:
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999
        
        if z > 800 and going_up:
            total_loss_dB = total_loss_nepers * 8.686
            return np.array(x_path), np.array(z_path), "escapes", total_loss_dB
        
        cos_psi = np.sqrt(1 - sin_psi**2)
        
        # Step along ray (with direction: up or down)
        dr = step_km * cos_psi * (1 if going_up else -1)
        d_theta = step_km * sin_psi / r
        
        r += dr
        theta += d_theta
        
        x_path.append(R_E * theta)
        z_path.append(r - R_E)
    
    total_loss_dB = total_loss_nepers * 8.686
    return np.array(x_path), np.array(z_path), "stops", total_loss_dB


# ============================================================================
# 2D RAY TRACING WITH TILTS
# ============================================================================

def trace_ray_2D_with_tilts(f_MHz, elevation_deg, refractive_index_func_2D,
                           max_distance_km=10000.0, step_km=2.0):
    """
    Trace a radio wave ray through a 2D tilted ionosphere.
    
    This version handles horizontal refractive index gradients that cause
    off-great-circle propagation (azimuth deflection).
    
    Parameters:
    -----------
    f_MHz : float
        Radio frequency in MHz
    elevation_deg : float
        Launch elevation angle in degrees (above horizon)
    refractive_index_func_2D : function
        Function that returns complex refractive index: n = f(freq_Hz, altitude_km, distance_km)
    max_distance_km : float
        Maximum surface distance to trace (default 10000 km)
    step_km : float
        Step size for ray tracing (default 2 km)
    
    Returns:
    --------
    x_path : ndarray
        Surface distance along ray path (km)
    z_path : ndarray
        Altitude along ray path (km)
    phi_path : ndarray
        Azimuth deflection along path (degrees off great circle)
    status : str
        "returns", "escapes", or "stops"
    """
    f_Hz = f_MHz * 1e6
    psi_0 = elevation_deg * pi / 180.0
    
    # Initial conditions
    r = R_E
    theta = 0.0
    x = 0.0  # Horizontal distance
    phi = 0.0  # Azimuth angle (radians)
    
    n_0 = np.real(refractive_index_func_2D(f_Hz, 0.0, 0.0))
    b = n_0 * r * np.sin(psi_0)  # Ray parameter
    
    # Storage
    x_path = [0.0]
    z_path = [0.0]
    phi_path = [0.0]
    
    # Ray direction tracking
    going_up = True
    
    for _ in range(int(max_distance_km / step_km)):
        z = r - R_E
        
        if z < 0 and len(x_path) > 10:
            return (np.array(x_path), np.array(z_path), 
                   np.array(phi_path) * 180.0 / pi, "returns")
        
        # Get refractive index at current position
        n = refractive_index_func_2D(f_Hz, z, x)
        n_real = np.real(n)
        n_squared = n * n
        n2_real = np.real(n_squared)
        
        # Check if ray is reflected
        if n2_real <= 0:
            going_up = False
        
        # Calculate horizontal gradient (dn/dx) for azimuth deflection
        dx_sample = 1.0  # km
        if x > 0:
            n_plus = refractive_index_func_2D(f_Hz, z, x + dx_sample)
            n_minus = refractive_index_func_2D(f_Hz, z, x - dx_sample)
            dn_dx = (np.real(n_plus) - np.real(n_minus)) / (2 * dx_sample)
        else:
            n_plus = refractive_index_func_2D(f_Hz, z, x + dx_sample)
            dn_dx = (np.real(n_plus) - n_real) / dx_sample
        
        # Ray angle from ray parameter
        sin_psi = b / (n_real * r)
        
        if abs(sin_psi) >= 1.0:
            if z > 600 and going_up:
                return (np.array(x_path), np.array(z_path),
                       np.array(phi_path) * 180.0 / pi, "escapes")
            else:
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999
        
        if z > 800 and going_up:
            return (np.array(x_path), np.array(z_path),
                   np.array(phi_path) * 180.0 / pi, "escapes")
        
        cos_psi = np.sqrt(1 - sin_psi**2)
        
        # Azimuth deflection from horizontal gradient
        # dphi/ds = -(1/n) × (dn/dx) / sin(psi)
        if abs(sin_psi) > 0.1:  # Avoid division by zero at high angles
            dphi_ds = -(1.0 / n_real) * dn_dx / abs(sin_psi)
            dphi = dphi_ds * step_km
        else:
            dphi = 0.0
        
        # Step along ray (using spherical geometry with direction tracking)
        dr = step_km * cos_psi * (1 if going_up else -1)
        d_theta = step_km * sin_psi / r  # Angular distance along Earth's surface
        
        r += dr
        theta += d_theta
        x += abs(d_theta * R_E)  # Horizontal coordinate follows spherical geometry
        phi += dphi
        
        x_path.append(R_E * theta)
        z_path.append(r - R_E)
        phi_path.append(phi)
    
    return (np.array(x_path), np.array(z_path),
           np.array(phi_path) * 180.0 / pi, "stops")


def trace_ray_2D_with_absorption(f_MHz, elevation_deg, refractive_index_func_2D,
                                 max_distance_km=10000.0, step_km=2.0):
    """
    Trace a radio wave ray through a 2D tilted ionosphere with absorption tracking.
    
    This version handles horizontal refractive index gradients (azimuth deflection)
    and calculates absorption loss using the Sen-Wyller formula.
    
    Parameters:
    -----------
    f_MHz : float
        Radio frequency in MHz
    elevation_deg : float
        Launch elevation angle in degrees (above horizon)
    refractive_index_func_2D : function
        Function that returns complex refractive index: n = f(freq_Hz, altitude_km, distance_km)
    max_distance_km : float
        Maximum surface distance to trace (default 10000 km)
    step_km : float
        Step size for ray tracing (default 2 km)
    
    Returns:
    --------
    x_path : ndarray
        Surface distance along ray path (km)
    z_path : ndarray
        Altitude along ray path (km)
    phi_path : ndarray
        Azimuth deflection along path (degrees off great circle)
    status : str
        "returns", "escapes", or "stops"
    total_loss_dB : float
        Total absorption loss along path in dB
    """
    f_Hz = f_MHz * 1e6
    omega = 2 * pi * f_Hz
    psi_0 = elevation_deg * pi / 180.0
    
    # Initial conditions
    r = R_E
    theta = 0.0
    x = 0.0  # Horizontal distance
    phi = 0.0  # Azimuth angle (radians)
    
    n_0 = np.real(refractive_index_func_2D(f_Hz, 0.0, 0.0))
    b = n_0 * r * np.sin(psi_0)  # Ray parameter
    
    # Storage
    x_path = [0.0]
    z_path = [0.0]
    phi_path = [0.0]
    total_loss_nepers = 0.0
    
    # Ray direction tracking
    going_up = True
    
    for _ in range(int(max_distance_km / step_km)):
        z = r - R_E
        
        if z < 0 and len(x_path) > 10:
            total_loss_dB = total_loss_nepers * 8.686
            return (np.array(x_path), np.array(z_path), 
                   np.array(phi_path) * 180.0 / pi, "returns", total_loss_dB)
        
        # Get refractive index at current position
        n = refractive_index_func_2D(f_Hz, z, x)
        n_real = np.real(n)
        n_squared = n * n
        n2_real = np.real(n_squared)
        
        # Calculate absorption coefficient (Sen-Wyller formula)
        # α = (ω/2c) × |Im(n²)| / Re(n)
        alpha = (omega / (2 * c)) * abs(np.imag(n_squared)) / max(n_real, 1e-10)
        
        # Accumulate loss (in Nepers)
        path_length_m = step_km * 1000.0
        total_loss_nepers += alpha * path_length_m
        
        # Check if ray is reflected
        if n2_real <= 0:
            going_up = False
        
        # Calculate horizontal gradient (dn/dx) for azimuth deflection
        dx_sample = 1.0  # km
        if x > 0:
            n_plus = refractive_index_func_2D(f_Hz, z, x + dx_sample)
            n_minus = refractive_index_func_2D(f_Hz, z, x - dx_sample)
            dn_dx = (np.real(n_plus) - np.real(n_minus)) / (2 * dx_sample)
        else:
            n_plus = refractive_index_func_2D(f_Hz, z, x + dx_sample)
            dn_dx = (np.real(n_plus) - n_real) / dx_sample
        
        # Ray angle from ray parameter
        sin_psi = b / (n_real * r)
        
        if abs(sin_psi) >= 1.0:
            if z > 600 and going_up:
                total_loss_dB = total_loss_nepers * 8.686
                return (np.array(x_path), np.array(z_path),
                       np.array(phi_path) * 180.0 / pi, "escapes", total_loss_dB)
            else:
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999
        
        if z > 800 and going_up:
            total_loss_dB = total_loss_nepers * 8.686
            return (np.array(x_path), np.array(z_path),
                   np.array(phi_path) * 180.0 / pi, "escapes", total_loss_dB)
        
        cos_psi = np.sqrt(1 - sin_psi**2)
        
        # Azimuth deflection from horizontal gradient
        # dphi/ds = -(1/n) × (dn/dx) / sin(psi)
        if abs(sin_psi) > 0.1:  # Avoid division by zero at high angles
            dphi_ds = -(1.0 / n_real) * dn_dx / abs(sin_psi)
            dphi = dphi_ds * step_km
        else:
            dphi = 0.0
        
        # Step along ray (using spherical geometry with direction tracking)
        dr = step_km * cos_psi * (1 if going_up else -1)
        d_theta = step_km * sin_psi / r  # Angular distance along Earth's surface
        
        r += dr
        theta += d_theta
        x += abs(d_theta * R_E)  # Horizontal coordinate follows spherical geometry
        phi += dphi
        
        x_path.append(R_E * theta)
        z_path.append(r - R_E)
        phi_path.append(phi)
    
    total_loss_dB = total_loss_nepers * 8.686
    return (np.array(x_path), np.array(z_path),
           np.array(phi_path) * 180.0 / pi, "stops", total_loss_dB)
