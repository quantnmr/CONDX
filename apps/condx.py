# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium", app_title="Modelling the Ionosphere")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd
    return alt, mo, np, pd, plt


@app.cell
def _(np):
    # ============================================================================
    # PHYSICS MODEL - All functions from model.py
    # ============================================================================
    
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
    
    return (
        R_E, c, pi,
        chapman_layer,
        make_ionosphere_for_foF2,
        make_tilted_ionosphere_2D,
        plasma_frequency_Hz_from_Ne,
        collision_frequency_Hz,
        make_refractive_index_func,
        trace_ray_spherical_with_path,
        trace_ray_with_absorption,
        trace_ray_2D_with_tilts,
        trace_ray_2D_with_absorption
    )


@app.cell
def _(alt, np, pd):
    # ============================================================================
    # PLOTTING FUNCTIONS - All functions from plots.py
    # ============================================================================
    
    LEGEND_STYLE = {
        'symbolStrokeWidth': 4,
        'symbolSize': 200
    }
    
    # Color schemes
    LAYER_COLORS = {
        'domain': ['D layer', 'E layer', 'F1 layer', 'F2 layer', 'Total'],
        'range': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#000000']
    }
    
    LOSS_COLORS = {
        'domain': ['Excellent (<10dB)', 'Good (10-30dB)', 'Weak (30-60dB)', 'Very Weak (>60dB)'],
        'range': ['green', 'orange', 'red', 'darkred']
    }
    
    # ============================================================================
    # ELECTRON DENSITY & PLASMA FREQUENCY VISUALIZATION
    # ============================================================================
    
    def plot_electron_density(z_array, Ne_D, Ne_E, Ne_F1, Ne_F2, Ne_total, 
                              fp_total, foF2):
        """
        Create electron density and plasma frequency visualization.
        
        Parameters:
        -----------
        z_array : ndarray
            Altitude array in km
        Ne_D, Ne_E, Ne_F1, Ne_F2 : ndarray
            Electron densities for each layer in electrons/m³
        Ne_total : ndarray
            Total electron density in electrons/m³
        fp_total : ndarray
            Plasma frequency in MHz
        foF2 : float
            Current foF2 value in MHz
        
        Returns:
        --------
        chart : altair.Chart
            Combined density and plasma frequency visualization
        """
        # Prepare data for electron density by layer
        density_data = []
        for i, z in enumerate(z_array):
            density_data.append({'altitude': z, 'density': Ne_D[i] / 1e9, 'layer': 'D layer'})
            density_data.append({'altitude': z, 'density': Ne_E[i] / 1e9, 'layer': 'E layer'})
            density_data.append({'altitude': z, 'density': Ne_F1[i] / 1e9, 'layer': 'F1 layer'})
            density_data.append({'altitude': z, 'density': Ne_F2[i] / 1e9, 'layer': 'F2 layer'})
            density_data.append({'altitude': z, 'density': Ne_total[i] / 1e9, 'layer': 'Total'})
        
        df_density = pd.DataFrame(density_data)
        
        # Prepare data for plasma frequency
        df_plasma = pd.DataFrame({
            'altitude': z_array,
            'plasma_freq': fp_total
        })
        
        # Create layer selection for hover highlighting
        layer_selection = alt.selection_point(
            fields=['layer'],
            bind='legend',
            on='mouseover',
            nearest=True
        )
        
        # Left chart: Electron density by layer
        density_chart = alt.Chart(df_density).mark_line(size=2).encode(
            x=alt.X('density:Q', 
                    title='Electron Density (× 10⁹ electrons/m³)',
                    scale=alt.Scale(domain=[0, df_density['density'].max() * 1.05])),
            y=alt.Y('altitude:Q', 
                    title='Altitude (km)',
                    scale=alt.Scale(domain=[0, 600])),
            color=alt.Color('layer:N',
                           legend=alt.Legend(title='Layer', **LEGEND_STYLE),
                           scale=alt.Scale(**LAYER_COLORS)),
            strokeDash=alt.condition(
                alt.datum.layer == 'Total',
                alt.value([5, 5]),
                alt.value([0])
            ),
            strokeWidth=alt.condition(
                alt.datum.layer == 'Total',
                alt.value(3),
                alt.value(2)
            ),
            opacity=alt.condition(layer_selection, alt.value(1.0), alt.value(0.3)),
            detail='layer:N',
            tooltip=[
                alt.Tooltip('layer:N', title='Layer'),
                alt.Tooltip('altitude:Q', title='Altitude (km)', format='.0f'),
                alt.Tooltip('density:Q', title='Density (×10⁹ e/m³)', format='.3f')
            ]
        ).add_params(
            layer_selection
        ).properties(
            width=400,
            height=400,
            title=f'Ionospheric Layers (foF2 = {foF2} MHz)'
        )
        
        # Add layer peak labels with bordered boxes
        idx_D = np.argmax(Ne_D)
        idx_E = np.argmax(Ne_E)
        idx_F1 = np.argmax(Ne_F1)
        idx_F2 = np.argmax(Ne_F2)
        
        layer_labels_data = [
            {'density': Ne_D[idx_D] / 1e9, 'altitude': z_array[idx_D], 'label': 'D (~75 km)', 'color': '#1f77b4'},
            {'density': Ne_E[idx_E] / 1e9, 'altitude': z_array[idx_E], 'label': 'E (~110 km)', 'color': '#ff7f0e'},
            {'density': Ne_F1[idx_F1] / 1e9, 'altitude': z_array[idx_F1], 'label': 'F1 (~190 km)', 'color': '#2ca02c'},
            {'density': Ne_F2[idx_F2] / 1e9, 'altitude': z_array[idx_F2], 'label': 'F2 (~300 km)', 'color': '#d62728'}
        ]
        
        # Create bordered label boxes
        box_width = df_density['density'].max() * 0.25
        box_height = 20
        
        label_boxes_data = []
        for row in layer_labels_data:
            label_boxes_data.append({
                'x_min': row['density'] + 2,
                'x_max': row['density'] + box_width,
                'y_min': row['altitude'] - box_height / 2,
                'y_max': row['altitude'] + box_height / 2,
                'x_center': row['density'] + box_width / 2,
                'y_center': row['altitude'],
                'label': row['label'],
                'color': row['color']
            })
        
        df_label_boxes = pd.DataFrame(label_boxes_data)
        
        label_boxes = alt.Chart(df_label_boxes).mark_rect(
            opacity=0.95,
            cornerRadius=4
        ).encode(
            x=alt.X('x_min:Q'),
            x2='x_max:Q',
            y=alt.Y('y_min:Q'),
            y2='y_max:Q',
            color=alt.value('white'),
            stroke=alt.Color('color:N', scale=None, legend=None),
            strokeWidth=alt.value(1.875)
        )
        
        label_text = alt.Chart(df_label_boxes).mark_text(
            align='center',
            baseline='middle',
            fontSize=12,
            fontWeight='bold'
        ).encode(
            x='x_center:Q',
            y='y_center:Q',
            text='label:N',
            color=alt.Color('color:N', scale=None, legend=None)
        )
        
        density_chart_with_labels = density_chart + label_boxes + label_text
        
        # Right chart: Plasma frequency profile
        plasma_chart = alt.Chart(df_plasma).mark_line(size=2.5, color='darkblue').encode(
            x=alt.X('plasma_freq:Q', 
                    title='Plasma Frequency (MHz)',
                    scale=alt.Scale(domain=[0, df_plasma['plasma_freq'].max() * 1.05])),
            y=alt.Y('altitude:Q', 
                    title='Altitude (km)',
                    scale=alt.Scale(domain=[0, 600])),
            order='altitude:Q',
            tooltip=[
                alt.Tooltip('altitude:Q', title='Altitude (km)', format='.0f'),
                alt.Tooltip('plasma_freq:Q', title='Plasma Freq (MHz)', format='.2f')
            ]
        ).properties(
            width=400,
            height=400,
            title='Plasma Frequency Profile'
        )
        
        # Add foF2 reference line
        foF2_line = alt.Chart(pd.DataFrame({'foF2': [foF2]})).mark_rule(
            color='red',
            strokeDash=[5, 5],
            size=2
        ).encode(
            x='foF2:Q'
        )
        
        # Add peak annotation
        peak_idx = np.argmax(fp_total)
        peak_annotation = alt.Chart(pd.DataFrame([{
            'plasma_freq': df_plasma['plasma_freq'].max() * 0.3,
            'altitude': 100,
            'text': f'Peak: {fp_total.max():.1f} MHz\nat {z_array[peak_idx]:.0f} km\n\nfoF2 = {foF2} MHz (red line)'
        }])).mark_text(
            align='left',
            fontSize=11,
            dx=5
        ).encode(
            x='plasma_freq:Q',
            y='altitude:Q',
            text='text:N'
        )
        
        # Combine charts horizontally
        return density_chart_with_labels | (plasma_chart + foF2_line + peak_annotation)
    
    
    # ============================================================================
    # 2D TILT VISUALIZATION
    # ============================================================================
    
    def plot_2D_tilts(ray_data_2D, foF2_start, foF2_end, elevation, tilt_distance_km):
        """
        Create 2D ionospheric tilt visualization (side view + top view).
        
        Parameters:
        -----------
        ray_data_2D : list of dict
            List with keys: frequency, x_path, z_path, phi_path, status
        foF2_start : float
            foF2 at transmitter in MHz
        foF2_end : float
            foF2 at reference distance in MHz
        elevation : float
            Elevation angle in degrees
        tilt_distance_km : float
            Distance over which gradient is defined
        
        Returns:
        --------
        chart : altair.Chart
            Combined side and top view visualization
        """
        # Prepare data
        tilt_path_data = []
        tilt_endpoint_data = []
        
        MAX_ALTITUDE = 600  # km - clip rays at this altitude
        
        for ray in ray_data_2D:
            freq = ray['frequency']
            x_path = ray['x_path']
            z_path = ray['z_path']
            phi_path = ray['phi_path']
            status = ray['status']
            label = f"{freq} MHz ({status})"
            
            # Clip paths at 600 km altitude
            for idx, (x, z, phi) in enumerate(zip(x_path, z_path, phi_path)):
                if z <= MAX_ALTITUDE:
                    tilt_path_data.append({
                        'frequency': freq,
                        'distance_km': x,
                        'altitude_km': z,
                        'azimuth_deg': phi,
                        'status': status,
                        'label': label,
                        'point_index': idx
                    })
                else:
                    # Ray exceeded altitude limit, stop adding points
                    break
            
            # Mark landing point (only if it's within altitude limit)
            if status == "returns" and z_path[-1] <= MAX_ALTITUDE:
                tilt_endpoint_data.append({
                    'frequency': freq,
                    'distance_km': x_path[-1],
                    'altitude_km': z_path[-1],
                    'azimuth_deg': phi_path[-1],
                    'label': label
                })
        
        df_tilt_paths = pd.DataFrame(tilt_path_data)
        df_tilt_endpoints = pd.DataFrame(tilt_endpoint_data)
        
        # Create frequency selection
        tilt_selection = alt.selection_point(
            fields=['label'],
            bind='legend',
            on='mouseover',
            nearest=True
        )
        
        # Sort labels numerically
        tilt_label_freq_map = df_tilt_paths[['label', 'frequency']].drop_duplicates()
        tilt_sorted_labels = tilt_label_freq_map.sort_values('frequency')['label'].tolist()
        
        # SIDE VIEW: Ray paths (altitude vs distance)
        side_view_chart = alt.Chart(df_tilt_paths).mark_line(size=2.5).encode(
            x=alt.X('distance_km:Q', 
                    title='Surface distance (km)',
                    scale=alt.Scale(domain=[0, df_tilt_paths['distance_km'].max() * 1.05])),
            y=alt.Y('altitude_km:Q', 
                    title='Height above ground (km)',
                    scale=alt.Scale(domain=[0, 600])),
            color=alt.Color('label:N',
                           legend=alt.Legend(title='Frequency', **LEGEND_STYLE),
                           scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels)),
            detail='frequency:N',
            opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.3)),
            strokeWidth=alt.condition(tilt_selection, alt.value(4), alt.value(2.5)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('status:N', title='Status')
            ]
        ).add_params(
            tilt_selection
        ).properties(
            width=700,
            height=350,
            title=f'Side View: foF2 gradient from {foF2_start} MHz (TX) to {foF2_end} MHz (at {tilt_distance_km:.0f} km), elev={elevation}°'
        )
        
        # Add endpoints to side view
        if len(df_tilt_endpoints) > 0:
            side_endpoints = alt.Chart(df_tilt_endpoints).mark_point(
                size=100,
                shape='cross',
                filled=True
            ).encode(
                x='distance_km:Q',
                y='altitude_km:Q',
                color=alt.Color('label:N', scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels), legend=None),
                opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.4)),
                size=alt.condition(tilt_selection, alt.value(250), alt.value(100))
            ).add_params(tilt_selection)
            side_view_with_endpoints = side_view_chart + side_endpoints
        else:
            side_view_with_endpoints = side_view_chart
        
        # TOP VIEW: Azimuth deflection
        top_view_chart = alt.Chart(df_tilt_paths).mark_line(size=2.5).encode(
            x=alt.X('distance_km:Q', 
                    title='Surface distance (km)',
                    scale=alt.Scale(domain=[0, df_tilt_paths['distance_km'].max() * 1.05])),
            y=alt.Y('azimuth_deg:Q', 
                    title='Azimuth deflection (degrees off great circle)',
                    scale=alt.Scale(domain=[
                        df_tilt_paths['azimuth_deg'].min() * 1.1 if df_tilt_paths['azimuth_deg'].min() < 0 else 0,
                        df_tilt_paths['azimuth_deg'].max() * 1.1 if df_tilt_paths['azimuth_deg'].max() > 0 else 1
                    ])),
            color=alt.Color('label:N',
                           legend=None,
                           scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels)),
            detail='frequency:N',
            opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.3)),
            strokeWidth=alt.condition(tilt_selection, alt.value(4), alt.value(2.5)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('azimuth_deg:Q', title='Azimuth (°)', format='.2f'),
                alt.Tooltip('distance_km:Q', title='Distance (km)', format='.0f')
            ]
        ).add_params(
            tilt_selection
        ).properties(
            width=700,
            height=250,
            title='Top View: Off-Great-Circle Propagation'
        )
        
        # Add zero line
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            strokeDash=[5, 5],
            color='gray'
        ).encode(y='y:Q')
        
        top_view_with_zero = top_view_chart + zero_line
        
        return side_view_with_endpoints & top_view_with_zero
    
    
    def plot_2D_side_view(ray_data_2D, foF2_start, foF2_end, elevation, tilt_distance_km):
        """
        Create 2D ionospheric tilt side view only (altitude vs distance).
        
        Parameters:
        -----------
        ray_data_2D : list of dict
            List with keys: frequency, x_path, z_path, phi_path, status
        foF2_start : float
            foF2 at transmitter in MHz
        foF2_end : float
            foF2 at reference distance in MHz
        elevation : float
            Elevation angle in degrees
        tilt_distance_km : float
            Distance over which gradient is defined
        
        Returns:
        --------
        chart : altair.Chart
            Side view visualization
        """
        # Prepare data
        tilt_path_data = []
        tilt_endpoint_data = []
        
        MAX_ALTITUDE = 600  # km - clip rays at this altitude
        
        for ray in ray_data_2D:
            freq = ray['frequency']
            x_path = ray['x_path']
            z_path = ray['z_path']
            phi_path = ray['phi_path']
            status = ray['status']
            label = f"{freq} MHz ({status})"
            
            # Clip paths at 600 km altitude
            for idx, (x, z, phi) in enumerate(zip(x_path, z_path, phi_path)):
                if z <= MAX_ALTITUDE:
                    tilt_path_data.append({
                        'frequency': freq,
                        'distance_km': x,
                        'altitude_km': z,
                        'azimuth_deg': phi,
                        'status': status,
                        'label': label,
                        'point_index': idx
                    })
                else:
                    # Ray exceeded altitude limit, stop adding points
                    break
            
            # Mark landing point (only if it's within altitude limit)
            if status == "returns" and z_path[-1] <= MAX_ALTITUDE:
                tilt_endpoint_data.append({
                    'frequency': freq,
                    'distance_km': x_path[-1],
                    'altitude_km': z_path[-1],
                    'azimuth_deg': phi_path[-1],
                    'label': label
                })
        
        df_tilt_paths = pd.DataFrame(tilt_path_data)
        df_tilt_endpoints = pd.DataFrame(tilt_endpoint_data)
        
        # Create frequency selection
        tilt_selection = alt.selection_point(
            fields=['label'],
            bind='legend',
            on='mouseover',
            nearest=True
        )
        
        # Sort labels numerically
        tilt_label_freq_map = df_tilt_paths[['label', 'frequency']].drop_duplicates()
        tilt_sorted_labels = tilt_label_freq_map.sort_values('frequency')['label'].tolist()
        
        # SIDE VIEW: Ray paths (altitude vs distance)
        side_view_chart = alt.Chart(df_tilt_paths).mark_line(size=2.5).encode(
            x=alt.X('distance_km:Q', 
                    title='Surface distance (km)',
                    scale=alt.Scale(domain=[0, df_tilt_paths['distance_km'].max() * 1.05])),
            y=alt.Y('altitude_km:Q', 
                    title='Height above ground (km)',
                    scale=alt.Scale(domain=[0, 600])),
            color=alt.Color('label:N',
                           legend=alt.Legend(title='Frequency', **LEGEND_STYLE),
                           scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels)),
            detail='frequency:N',
            opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.3)),
            strokeWidth=alt.condition(tilt_selection, alt.value(4), alt.value(2.5)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('status:N', title='Status')
            ]
        ).add_params(
            tilt_selection
        ).properties(
            width=800,
            height=400,
            title=f'Side View: foF2 gradient from {foF2_start} MHz (TX) to {foF2_end} MHz (at {tilt_distance_km:.0f} km), elev={elevation}°'
        )
        
        # Add endpoints to side view
        if len(df_tilt_endpoints) > 0:
            side_endpoints = alt.Chart(df_tilt_endpoints).mark_point(
                size=100,
                shape='cross',
                filled=True
            ).encode(
                x='distance_km:Q',
                y='altitude_km:Q',
                color=alt.Color('label:N', scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels), legend=None),
                opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.4)),
                size=alt.condition(tilt_selection, alt.value(250), alt.value(100))
            ).add_params(tilt_selection)
            return side_view_chart + side_endpoints
        else:
            return side_view_chart
    
    
    def plot_2D_top_view(ray_data_2D, foF2_start, foF2_end, elevation, tilt_distance_km):
        """
        Create 2D ionospheric tilt top view only (azimuth deflection).
        
        Parameters:
        -----------
        ray_data_2D : list of dict
            List with keys: frequency, x_path, z_path, phi_path, status
        foF2_start : float
            foF2 at transmitter in MHz
        foF2_end : float
            foF2 at reference distance in MHz
        elevation : float
            Elevation angle in degrees
        tilt_distance_km : float
            Distance over which gradient is defined
        
        Returns:
        --------
        chart : altair.Chart
            Top view visualization
        """
        # Prepare data
        tilt_path_data = []
        
        MAX_ALTITUDE = 600  # km - clip rays at this altitude
        
        for ray in ray_data_2D:
            freq = ray['frequency']
            x_path = ray['x_path']
            z_path = ray['z_path']
            phi_path = ray['phi_path']
            status = ray['status']
            label = f"{freq} MHz ({status})"
            
            # Clip paths at 600 km altitude
            for idx, (x, z, phi) in enumerate(zip(x_path, z_path, phi_path)):
                if z <= MAX_ALTITUDE:
                    tilt_path_data.append({
                        'frequency': freq,
                        'distance_km': x,
                        'altitude_km': z,
                        'azimuth_deg': phi,
                        'status': status,
                        'label': label,
                        'point_index': idx
                    })
                else:
                    # Ray exceeded altitude limit, stop adding points
                    break
        
        df_tilt_paths = pd.DataFrame(tilt_path_data)
        
        # Create frequency selection
        tilt_selection = alt.selection_point(
            fields=['label'],
            bind='legend',
            on='mouseover',
            nearest=True
        )
        
        # Sort labels numerically
        tilt_label_freq_map = df_tilt_paths[['label', 'frequency']].drop_duplicates()
        tilt_sorted_labels = tilt_label_freq_map.sort_values('frequency')['label'].tolist()
        
        # TOP VIEW: Azimuth deflection
        top_view_chart = alt.Chart(df_tilt_paths).mark_line(size=2.5).encode(
            x=alt.X('distance_km:Q', 
                    title='Surface distance (km)',
                    scale=alt.Scale(domain=[0, df_tilt_paths['distance_km'].max() * 1.05])),
            y=alt.Y('azimuth_deg:Q', 
                    title='Azimuth deflection (degrees off great circle)',
                    scale=alt.Scale(domain=[
                        df_tilt_paths['azimuth_deg'].min() * 1.1 if df_tilt_paths['azimuth_deg'].min() < 0 else 0,
                        df_tilt_paths['azimuth_deg'].max() * 1.1 if df_tilt_paths['azimuth_deg'].max() > 0 else 1
                    ])),
            color=alt.Color('label:N',
                           legend=alt.Legend(title='Frequency', **LEGEND_STYLE),
                           scale=alt.Scale(scheme='category10', domain=tilt_sorted_labels)),
            detail='frequency:N',
            opacity=alt.condition(tilt_selection, alt.value(1.0), alt.value(0.3)),
            strokeWidth=alt.condition(tilt_selection, alt.value(4), alt.value(2.5)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('azimuth_deg:Q', title='Azimuth (°)', format='.2f'),
                alt.Tooltip('distance_km:Q', title='Distance (km)', format='.0f'),
                alt.Tooltip('status:N', title='Status')
            ]
        ).add_params(
            tilt_selection
        ).properties(
            width=800,
            height=350,
            title=f'Top View: Off-Great-Circle Propagation (foF2 {foF2_start}→{foF2_end} MHz over {tilt_distance_km:.0f} km, elev={elevation}°)'
        )
        
        # Add zero line
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            strokeDash=[5, 5],
            color='gray'
        ).encode(y='y:Q')
        
        return top_view_chart + zero_line
    
    
    # ============================================================================
    # 2D ABSORPTION VISUALIZATION
    # ============================================================================
    
    def plot_2D_absorption(ray_data_2D_absorption, foF2_start, foF2_end, elevation, tilt_distance_km):
        """
        Create 2D absorption visualization (side view with loss + bar chart).
        
        Parameters:
        -----------
        ray_data_2D_absorption : list of dict
            List with keys: frequency, x_path, z_path, phi_path, status, loss_dB
        foF2_start : float
            foF2 at transmitter in MHz
        foF2_end : float
            foF2 at reference distance in MHz
        elevation : float
            Elevation angle in degrees
        tilt_distance_km : float
            Distance over which gradient is defined
        
        Returns:
        --------
        chart : altair.Chart
            Combined side view and bar chart visualization
        """
        # Prepare data
        path_data = []
        endpoint_data = []
        loss_summary_data = []
        
        MAX_ALTITUDE = 600  # km - clip rays at this altitude
        
        for ray in ray_data_2D_absorption:
            freq = ray['frequency']
            x_path = ray['x_path']
            z_path = ray['z_path']
            status = ray['status']
            loss_dB = ray['loss_dB']
            
            # Assign color category based on loss
            if loss_dB < 10:
                loss_category = 'Excellent (<10dB)'
            elif loss_dB < 30:
                loss_category = 'Good (10-30dB)'
            elif loss_dB < 60:
                loss_category = 'Weak (30-60dB)'
            else:
                loss_category = 'Very Weak (>60dB)'
            
            # Clip paths at 600 km altitude
            for idx, (x, z) in enumerate(zip(x_path, z_path)):
                if z <= MAX_ALTITUDE:
                    path_data.append({
                        'frequency': freq,
                        'distance_km': x,
                        'altitude_km': z,
                        'loss_dB': loss_dB,
                        'status': status,
                        'loss_category': loss_category,
                        'point_index': idx
                    })
                else:
                    # Ray exceeded altitude limit, stop adding points
                    break
            
            # Mark endpoint (only if it's within altitude limit)
            if status == "returns" and z_path[-1] <= MAX_ALTITUDE:
                endpoint_data.append({
                    'frequency': freq,
                    'distance_km': x_path[-1],
                    'altitude_km': z_path[-1],
                    'loss_dB': loss_dB,
                    'loss_category': loss_category
                })
                
                # Add to summary for bar chart
                loss_summary_data.append({
                    'frequency': freq,
                    'loss_dB': loss_dB,
                    'distance_km': x_path[-1],
                    'loss_category': loss_category
                })
        
        df_paths = pd.DataFrame(path_data)
        df_endpoints = pd.DataFrame(endpoint_data)
        df_loss = pd.DataFrame(loss_summary_data)
        
        # Create frequency selection
        freq_selection = alt.selection_point(
            fields=['frequency'],
            bind='legend',
            on='mouseover',
            nearest=True
        )
        
        # SIDE VIEW: Ray paths color-coded by absorption loss
        side_view_chart = alt.Chart(df_paths).mark_line(size=2.5).encode(
            x=alt.X('distance_km:Q', 
                    title='Surface distance (km)',
                    scale=alt.Scale(domain=[0, df_paths['distance_km'].max() * 1.05])),
            y=alt.Y('altitude_km:Q', 
                    title='Height above ground (km)',
                    scale=alt.Scale(domain=[0, 600])),
            color=alt.Color('loss_category:N',
                           legend=alt.Legend(title='Signal Strength', **LEGEND_STYLE),
                           scale=alt.Scale(**LOSS_COLORS)),
            detail='frequency:N',
            opacity=alt.condition(freq_selection, alt.value(1.0), alt.value(0.3)),
            strokeWidth=alt.condition(freq_selection, alt.value(4), alt.value(2.5)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('loss_dB:Q', title='Total Loss (dB)', format='.1f'),
                alt.Tooltip('status:N', title='Status')
            ]
        ).add_params(
            freq_selection
        ).properties(
            width=800,
            height=350,
            title=f'Ray Paths with Absorption: foF2 {foF2_start}→{foF2_end} MHz over {tilt_distance_km:.0f} km, elev={elevation}°'
        )
        
        # Add endpoints to side view
        if len(df_endpoints) > 0:
            side_endpoints = alt.Chart(df_endpoints).mark_point(
                size=100,
                shape='cross',
                filled=True
            ).encode(
                x='distance_km:Q',
                y='altitude_km:Q',
                color=alt.Color('loss_category:N', scale=alt.Scale(**LOSS_COLORS), legend=None),
                opacity=alt.condition(freq_selection, alt.value(1.0), alt.value(0.4)),
                size=alt.condition(freq_selection, alt.value(250), alt.value(100))
            ).add_params(freq_selection)
            side_view_with_endpoints = side_view_chart + side_endpoints
        else:
            side_view_with_endpoints = side_view_chart
        
        # BAR CHART: Absorption loss by frequency
        if len(df_loss) > 0:
            bar_chart = alt.Chart(df_loss).mark_bar().encode(
                x=alt.X('frequency:O', 
                        title='Frequency (MHz)',
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y('loss_dB:Q', 
                        title='Total Path Loss (dB)',
                        scale=alt.Scale(domain=[0, max(df_loss['loss_dB'].max() * 1.1, 10)])),
                color=alt.Color('loss_category:N',
                               scale=alt.Scale(**LOSS_COLORS),
                               legend=None),
                opacity=alt.condition(freq_selection, alt.value(1.0), alt.value(0.4)),
                strokeWidth=alt.condition(freq_selection, alt.value(2), alt.value(0)),
                stroke=alt.value('black'),
                tooltip=[
                    alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                    alt.Tooltip('loss_dB:Q', title='Loss (dB)', format='.1f'),
                    alt.Tooltip('distance_km:Q', title='Distance (km)', format='.0f'),
                    alt.Tooltip('loss_category:N', title='Signal Quality')
                ]
            ).add_params(
                freq_selection
            ).properties(
                width=800,
                height=250,
                title='Absorption Loss by Frequency (hover to highlight)'
            )
            
            # Add text labels on bars
            text = alt.Chart(df_loss).mark_text(
                align='center',
                baseline='bottom',
                dy=-5,
                fontSize=10,
                fontWeight='bold'
            ).encode(
                x=alt.X('frequency:O'),
                y=alt.Y('loss_dB:Q'),
                text=alt.Text('loss_dB:Q', format='.1f'),
                opacity=alt.condition(freq_selection, alt.value(1.0), alt.value(0.5))
            ).add_params(
                freq_selection
            )
            
            loss_chart = bar_chart + text
        else:
            # No data message
            loss_chart = alt.Chart(pd.DataFrame({'text': ['No rays returned to Earth']})).mark_text(
                size=16,
                align='center'
            ).encode(
                text='text:N'
            ).properties(
                width=800,
                height=250,
                title='Absorption Loss by Frequency'
            )
        
        return side_view_with_endpoints & loss_chart
    
    return (
        LEGEND_STYLE, LAYER_COLORS, LOSS_COLORS,
        plot_electron_density,
        plot_2D_side_view,
        plot_2D_top_view,
        plot_2D_absorption
    )


@app.cell
def _(mo):
    # Main title always visible
    _title = mo.md("# Understanding HF Radio Wave Propagation Through the Ionosphere")

    # Collapsible introduction section (closed by default)
    _introduction = mo.accordion({
        "📖 Introduction & Physics Background (click to expand)": mo.md(r"""
        ## Interactive Educational Notebook

        This notebook demonstrates how **High Frequency (HF) radio waves** (1.8-30 MHz) propagate through Earth's ionosphere to enable long-distance communication. Use the interactive visualizations below to explore how different conditions affect radio propagation.

        ---

        ## Key Concepts

        **Ionospheric Skip Propagation:**
        - HF radio waves launched from Earth's surface can **reflect off the ionosphere** (a layer of ionized atmosphere 60-600 km altitude)
        - The wave "skips" back to Earth at a distant location, enabling over-the-horizon communication
        - Skip distance depends on **frequency**, **launch angle**, and **ionospheric conditions**

        **Critical Parameters:**
        - **foF2 (critical frequency)**: Maximum frequency that can be reflected by the F2 layer at vertical incidence
          - Higher foF2 → ionosphere can reflect higher frequencies
          - Varies with solar activity, time of day, season (typical range: 2-25 MHz in this model)
          - **Night** (2-5 MHz): D-layer absent, F2 layer at high altitude (~400 km)
          - **Dawn/Dusk** (5-10 MHz): D-layer building, F1 layer emerging
          - **Day** (10-25 MHz): All layers present, F2 layer at lower altitude (~260 km)
        - **Elevation angle**: Launch angle of the radio wave above the horizon
          - Low angles (5-20°) → long skip distances (2000+ km)
          - High angles (60-85°) → short skip distances (< 500 km)

        ---

        ## The Physics Behind the Model

        ### 1. Refractive Index (Wave Bending & Absorption)

        The complex refractive index determines both wave bending and absorption using the **Appleton-Hartree equation**:

        $$n^2 = 1 - \frac{X}{1 - jZ}$$

        where:
        - $X = (\omega_p/\omega)^2$ = normalized plasma frequency (depends on electron density)
        - $Z = \nu/\omega$ = normalized collision frequency (determines absorption)
        - $\omega_p = 2\pi \cdot 8.98 \times 10^3 \sqrt{N_e}$ (plasma frequency from electron density $N_e$)
        - $\nu(z)$ = altitude-dependent electron-neutral collision frequency

        ### 2. Snell's Law in Spherical Coordinates

        The **ray parameter** $b = n \cdot r \cdot \sin(\psi)$ is conserved along the ray path, where:
        - $n$ = refractive index
        - $r$ = radial distance from Earth's center
        - $\psi$ = angle between ray and radial direction

        As waves travel into regions of higher electron density, $n$ decreases. To conserve $b$, the ray bends. When $n^2 < 0$, the wave reflects back toward Earth.

        ### 3. Absorption Loss (Sen-Wyller Formula)

        Radio waves lose energy through collisions. The absorption coefficient is:

        $$\alpha = \frac{\omega}{2c} \cdot \frac{|\text{Im}(n^2)|}{\text{Re}(n)}$$

        Total path loss: $\text{Loss (dB)} = 8.686 \int \alpha(s) \, ds$

        Key behaviors (calibrated to match real-world measurements):
        - **Lower frequencies** (1.8-7 MHz) suffer **more absorption** in D-layer (∝ 1/f²)
        - **Mid-HF** (14-21 MHz) has **lowest loss** for long-distance paths
        - **D-layer** (70-90 km) causes most absorption due to high collision frequency

        ### 4. Chapman Layer Electron Density

        Each ionospheric layer (D, E, F1, F2) uses the **Chapman function**:

        $$N_e(z) = N_{max} \exp\left[\frac{1}{2}\left(1 - \xi - e^{-\xi}\right)\right]$$

        where $\xi = (z - h_m)/H$ with $h_m$ = peak altitude, $H$ = scale height.

        ---

        ## How to Use This Notebook

        **This notebook is divided into two parts:**

        ### Part 1: 1D Model (Vertical Variations Only)

        1. **Electron Density Profile** - Shows how electron density varies with altitude across the four ionospheric layers (D, E, F1, F2)
        2. **Interactive Controls** - Sliders to adjust foF2 (ionospheric conditions) and elevation angle (launch angle)
        3. **Ray Path Visualization** - Shows how radio waves at different frequencies propagate through the ionosphere
        4. **Signal Loss Analysis** - Displays absorption loss for each frequency with color-coded signal strength

        **Try experimenting with the sliders to see:**
        - How higher foF2 allows higher frequencies to propagate
        - How lower elevation angles create longer skip distances
        - Why some frequencies work better than others (lower absorption loss)
        - How rays at different frequencies bend differently through the ionosphere

        ### Part 2: 2D Model (Horizontal Gradients)

        The second part extends the model to include **horizontal ionospheric gradients** that create fascinating propagation effects:

        5. **Ionospheric Tilts** - Model horizontal foF2 variations (day/night terminator, etc.)
        6. **Off-Great-Circle Propagation** - Visualize how rays bend sideways when encountering horizontal gradients
        7. **2D Absorption Loss** - See how signal strength varies through tilted ionosphere

        **Try experimenting with the 2D sliders to see:**
        - Day/night terminator effects (15 MHz → 4 MHz over 3000 km)
        - Off-great-circle propagation (sideways ray bending)
        - How horizontal gradients affect absorption loss
        - Pedersen ray behavior with high elevation angles
        """)
    })

    mo.vstack([_title, _introduction])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Electron Density Profile Through the Ionosphere

    The following plot shows how electron density varies with altitude in the ionosphere. The ionosphere consists of four main layers (D, E, F1, F2), each with different characteristics:

    - **D layer (~75 km)**: Lowest density, but high absorption due to frequent electron collisions
    - **E layer (~110 km)**: Regular daytime layer with moderate density
    - **F1 layer (~190 km)**: Daytime layer that merges with F2 at night
    - **F2 layer (~300 km)**: Highest peak density, most important for HF propagation

    The F2 layer peak density varies with the foF2 parameter (critical frequency). Higher foF2 means higher electron density, which allows higher radio frequencies to be reflected.
    """)
    return


@app.cell
def _(
    chapman_layer,
    foF2_slider,
    make_ionosphere_for_foF2,
    np,
    plasma_frequency_Hz_from_Ne,
    plots,
):
    # Get current foF2 value from slider
    _density_foF2 = foF2_slider.value

    # Build ionosphere for this foF2
    _total_density_func = make_ionosphere_for_foF2(_density_foF2)

    # Calculate F2 peak density from foF2
    _foF2_Hz = _density_foF2 * 1e6
    _Ne_peak_cm3 = (_foF2_Hz / 8.98e3) ** 2
    _Ne_peak_m3 = _Ne_peak_cm3 * 1e6

    # Generate altitude array
    _z_array = np.linspace(0, 600, 1000)

    # Calculate layer parameters matching make_ionosphere_for_foF2
    _h_F2 = 420.0 - (_density_foF2 - 2.0) * 7.0
    _h_F2 = max(250.0, min(420.0, _h_F2))

    if _density_foF2 < 5.0:
        _D_factor = 0.0
    elif _density_foF2 < 8.0:
        _D_factor = (_density_foF2 - 5.0) / 3.0
    else:
        _D_factor = 1.0
    _Ne_D_max = 1.5e9 * _D_factor

    if _density_foF2 < 4.5:
        _F1_factor = 0.0
    elif _density_foF2 < 7.0:
        _F1_factor = (_density_foF2 - 4.5) / 2.5
    else:
        _F1_factor = 1.0
    _Ne_F1_max = 3e11 * _F1_factor

    _E_factor = 0.5 + 0.5 * min(1.0, _density_foF2 / 12.0)
    _Ne_E_max = 8e10 * _E_factor

    # Calculate individual layer contributions
    _Ne_D = chapman_layer(_z_array, _Ne_D_max, 75.0, 8.0)
    _Ne_E = chapman_layer(_z_array, _Ne_E_max, 110.0, 15.0)
    _Ne_F1 = chapman_layer(_z_array, _Ne_F1_max, 190.0, 30.0)
    _Ne_F2 = chapman_layer(_z_array, _Ne_peak_m3, _h_F2, 55.0)
    _Ne_total = _total_density_func(_z_array)

    # Calculate plasma frequency
    _fp_total = plasma_frequency_Hz_from_Ne(_Ne_total) / 1e6  # Convert to MHz

    # Use plots module to create visualization
    plot_electron_density(_z_array, _Ne_D, _Ne_E, _Ne_F1, _Ne_F2, _Ne_total, _fp_total, _density_foF2)
    return


@app.cell
def _(mo):
    # ============================================================================
    # UI CONTROLS: 1D MODEL SLIDERS
    # ============================================================================

    # 1D mode slider (single foF2)
    foF2_slider = mo.ui.slider(
        start=2.0,
        stop=25.0,
        step=0.1,
        value=12.0,
        label="foF2 (MHz)"
    )

    # Elevation angle slider
    elevation_slider = mo.ui.slider(
        start=1,
        stop=89,
        step=1,
        value=30,
        label="Elevation Angle (degrees)"
    )

    # Display descriptive text
    mo.vstack([
        mo.md("## Interactive Controls - 1D Model"),
        mo.md("""
        Adjust the sliders below to explore how different ionospheric conditions affect radio propagation:
        - **foF2**: Critical frequency of the F2 layer (affects ionization level)
        - **Elevation angle**: Launch angle of the radio wave above the horizon
        """),
    ])
    return elevation_slider, foF2_slider


@app.cell
def _(elevation_slider, foF2_slider, mo):
    # ============================================================================
    # SLIDER DISPLAY WITH VALUES AND DAY/NIGHT INDICATOR (1D MODEL)
    # ============================================================================

    # Get current slider values
    _current_foF2 = foF2_slider.value
    _current_elev = elevation_slider.value

    # Determine ionospheric condition based on foF2 value
    if _current_foF2 < 5.0:
        _condition = "🌙 **Night conditions** (D-layer absent, F2 high altitude)"
        _color = "blue"
    elif _current_foF2 < 10.0:
        _condition = "🌅 **Dawn/Dusk transition** (D-layer building, intermediate conditions)"
        _color = "orange"
    else:
        _condition = "☀️ **Daytime conditions** (D-layer present, F2 low altitude)"
        _color = "gold"

    # Display sliders with value labels and condition indicator
    mo.vstack([
        mo.md(f"**foF2 ({_current_foF2} MHz)**"),
        foF2_slider,
        mo.md(f"<span style='color: {_color}; font-weight: bold;'>{_condition}</span>"),
        mo.md(""),  # Spacer
        mo.md(f"**Elevation Angle ({_current_elev}°)**"),
        elevation_slider,
    ])
    return


@app.cell
def _(
    alt,
    elevation_slider,
    foF2_slider,
    make_ionosphere_for_foF2,
    make_refractive_index_func,
    pd,
    trace_ray_spherical_with_path,
):
    #import altair as alt
    #import pandas as pd

    # ============================================================================
    # INTERACTIVE RAY PATH PLOT (ALTAIR)
    # ============================================================================

    # Get current slider values
    _interactive_foF2 = foF2_slider.value
    _interactive_elev = elevation_slider.value

    # Build ionosphere model for selected foF2
    _interactive_electron_density = make_ionosphere_for_foF2(_interactive_foF2)
    _interactive_refractive_index = make_refractive_index_func(_interactive_electron_density)

    # Amateur radio HF bands (using typical frequencies)
    _interactive_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Prepare data for Altair
    _path_data_interactive = []
    _endpoint_data_interactive = []

    _MAX_ALTITUDE = 600  # km - clip rays at this altitude

    for _f_interactive in _interactive_freqs:
        _x_int, _z_int, _status_int = trace_ray_spherical_with_path(
            _f_interactive, _interactive_elev, _interactive_refractive_index,
            max_distance_km=8000.0, step_km=5.0
        )

        # Add path data (clip at 600 km altitude)
        for _idx, (_x, _z) in enumerate(zip(_x_int, _z_int)):
            if _z <= _MAX_ALTITUDE:
                _path_data_interactive.append({
                    'frequency': _f_interactive,
                    'distance_km': _x,
                    'altitude_km': _z,
                    'status': _status_int,
                    'label': f"{_f_interactive} MHz ({_status_int})",
                    'point_index': _idx
                })
            else:
                # Ray exceeded altitude limit, stop adding points
                break

        # Mark landing point for returning rays (only if within altitude limit)
        if _status_int == "returns" and _z_int[-1] <= _MAX_ALTITUDE:
            _endpoint_data_interactive.append({
                'frequency': _f_interactive,
                'distance_km': _x_int[-1],
                'altitude_km': _z_int[-1],
                'label': f"{_f_interactive} MHz ({_status_int})"
            })

    _df_paths_interactive = pd.DataFrame(_path_data_interactive)
    _df_endpoints_interactive = pd.DataFrame(_endpoint_data_interactive)

    # Create frequency selection for hover highlighting
    _freq_selection_interactive = alt.selection_point(
        fields=['label'],
        bind='legend',
        on='mouseover',
        nearest=True
    )

    # Get unique labels and sort them numerically by frequency
    _label_freq_map = _df_paths_interactive[['label', 'frequency']].drop_duplicates()
    _sorted_labels = _label_freq_map.sort_values('frequency')['label'].tolist()

    # Create ray path chart
    _ray_chart_interactive = alt.Chart(_df_paths_interactive).mark_line(size=2.5).encode(
        x=alt.X('distance_km:Q', 
                title='Surface distance (km)',
                scale=alt.Scale(domain=[0, _df_paths_interactive['distance_km'].max() * 1.05])),
        y=alt.Y('altitude_km:Q', 
                title='Height above ground (km)',
                scale=alt.Scale(domain=[0, 600])),
        color=alt.Color('label:N',
                       legend=alt.Legend(title='Frequency (MHz)', columns=1, symbolStrokeWidth=4, symbolSize=200),
                       scale=alt.Scale(scheme='category10', domain=_sorted_labels)),
        detail='frequency:N',
        opacity=alt.condition(_freq_selection_interactive, alt.value(1.0), alt.value(0.3)),
        strokeWidth=alt.condition(_freq_selection_interactive, alt.value(4), alt.value(2.5)),
        tooltip=[
            alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
            alt.Tooltip('status:N', title='Status')
        ]
    ).add_params(
        _freq_selection_interactive
    ).properties(
        width=800,
        height=400,
        title=f'Ray Paths: Elevation = {_interactive_elev}°, foF2 = {_interactive_foF2} MHz (hover over legend or rays to highlight)'
    )

    # Add endpoint markers for returning rays
    if len(_df_endpoints_interactive) > 0:
        _endpoint_chart_interactive = alt.Chart(_df_endpoints_interactive).mark_point(
            shape='cross',
            filled=True,
            size=150
        ).encode(
            x='distance_km:Q',
            y='altitude_km:Q',
            color=alt.Color('label:N', legend=None, scale=alt.Scale(scheme='category10')),
            size=alt.condition(_freq_selection_interactive, alt.value(300), alt.value(150)),
            opacity=alt.condition(_freq_selection_interactive, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('distance_km:Q', title='Skip Distance (km)', format='.0f'),
                alt.Tooltip('label:N', title='Status')
            ]
        ).add_params(
            _freq_selection_interactive
        )

        _combined_chart_interactive = _ray_chart_interactive + _endpoint_chart_interactive
    else:
        _combined_chart_interactive = _ray_chart_interactive

    _combined_chart_interactive
    return


@app.cell
def _(
    alt,
    elevation_slider,
    foF2_slider,
    make_ionosphere_for_foF2,
    make_refractive_index_func,
    pd,
    trace_ray_with_absorption,
):


    _loss_foF2 = foF2_slider.value
    _loss_elev = elevation_slider.value

    _loss_electron_density = make_ionosphere_for_foF2(_loss_foF2)
    _loss_refractive_index = make_refractive_index_func(_loss_electron_density)

    # Amateur radio HF bands (using typical frequencies)
    _loss_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Prepare data for Altair
    _path_data = []
    _loss_summary_data = []

    _MAX_ALTITUDE_LOSS = 600  # km - clip rays at this altitude

    for _f_loss in _loss_freqs:
        _x_loss, _z_loss, _status_loss, _total_loss = trace_ray_with_absorption(
            _f_loss, _loss_elev, _loss_refractive_index,
            max_distance_km=8000.0, step_km=5.0
        )

        # Assign color category based on loss
        if _total_loss < 10:
            _loss_category = 'Excellent (<10dB)'
            _color_val = 1
        elif _total_loss < 30:
            _loss_category = 'Good (10-30dB)'
            _color_val = 2
        elif _total_loss < 60:
            _loss_category = 'Weak (30-60dB)'
            _color_val = 3
        else:
            _loss_category = 'Very Weak (>60dB)'
            _color_val = 4

        # Add path data (clip at 600 km altitude)
        for _idx, (_x, _z) in enumerate(zip(_x_loss, _z_loss)):
            if _z <= _MAX_ALTITUDE_LOSS:
                _path_data.append({
                    'frequency': _f_loss,
                    'distance_km': _x,
                    'altitude_km': _z,
                    'loss_dB': _total_loss,
                    'status': _status_loss,
                    'loss_category': _loss_category,
                    'color_val': _color_val,
                    'point_index': _idx
                })
            else:
                # Ray exceeded altitude limit, stop adding points
                break

        # Add summary data for bar chart (only if within altitude limit)
        if _status_loss == "returns" and _z_loss[-1] <= _MAX_ALTITUDE_LOSS:
            _loss_summary_data.append({
                'frequency': _f_loss,
                'loss_dB': _total_loss,
                'distance_km': _x_loss[-1],
                'loss_category': _loss_category
            })

    _df_paths = pd.DataFrame(_path_data)
    _df_loss = pd.DataFrame(_loss_summary_data)

    # Create animated ray path chart with frequency selection
    _freq_selection = alt.selection_point(
        fields=['frequency'],
        bind='legend',
        on='mouseover',
        nearest=True
    )

    _ray_chart = alt.Chart(_df_paths).mark_line(size=2.5).encode(
        x=alt.X('distance_km:Q', 
                title='Surface distance (km)',
                scale=alt.Scale(domain=[0, _df_paths['distance_km'].max() * 1.05])),
        y=alt.Y('altitude_km:Q', 
                title='Height above ground (km)',
                scale=alt.Scale(domain=[0, 600])),
        color=alt.Color('loss_category:N',
                       scale=alt.Scale(
                           domain=['Excellent (<10dB)', 'Good (10-30dB)', 'Weak (30-60dB)', 'Very Weak (>60dB)'],
                           range=['green', 'orange', 'red', 'darkred']
                       ),
                       legend=alt.Legend(title='Signal Strength', symbolStrokeWidth=4, symbolSize=200)),
        detail='frequency:N',
        opacity=alt.condition(_freq_selection, alt.value(1.0), alt.value(0.3)),
        strokeWidth=alt.condition(_freq_selection, alt.value(4), alt.value(2)),
        tooltip=[
            alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
            alt.Tooltip('loss_dB:Q', title='Total Loss (dB)', format='.1f'),
            alt.Tooltip('status:N', title='Status')
        ]
    ).add_params(
        _freq_selection
    ).properties(
        width=800,
        height=350,
        title=f'Ray Paths with Absorption Loss: Elevation = {_loss_elev}°, foF2 = {_loss_foF2} MHz (hover over legend or rays to highlight)'
    )

    # Add endpoint markers for returning rays with interactivity
    _endpoints = _df_paths[_df_paths['status'] == 'returns'].groupby('frequency').tail(1)
    _endpoint_chart = alt.Chart(_endpoints).mark_point(
        shape='cross',
        filled=True
    ).encode(
        x='distance_km:Q',
        y='altitude_km:Q',
        color=alt.Color('loss_category:N',
                       scale=alt.Scale(
                           domain=['Excellent (<10dB)', 'Good (10-30dB)', 'Weak (30-60dB)', 'Very Weak (>60dB)'],
                           range=['green', 'orange', 'red', 'darkred']
                       ),
                       legend=None),
        size=alt.condition(_freq_selection, alt.value(250), alt.value(100)),
        opacity=alt.condition(_freq_selection, alt.value(1.0), alt.value(0.6))
    ).add_params(
        _freq_selection
    )

    # Create interactive bar chart linked to ray paths
    if len(_df_loss) > 0:
        _bar_chart = alt.Chart(_df_loss).mark_bar().encode(
            x=alt.X('frequency:O', 
                    title='Frequency (MHz)',
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y('loss_dB:Q', 
                    title='Total Path Loss (dB)',
                    scale=alt.Scale(domain=[0, max(_df_loss['loss_dB'].max() * 1.1, 10)])),
            color=alt.Color('loss_category:N',
                           scale=alt.Scale(
                               domain=['Excellent (<10dB)', 'Good (10-30dB)', 'Weak (30-60dB)', 'Very Weak (>60dB)'],
                               range=['green', 'orange', 'red', 'darkred']
                           ),
                           legend=None),
            opacity=alt.condition(_freq_selection, alt.value(1.0), alt.value(0.4)),
            strokeWidth=alt.condition(_freq_selection, alt.value(2), alt.value(0)),
            stroke=alt.value('black'),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('loss_dB:Q', title='Loss (dB)', format='.1f'),
                alt.Tooltip('distance_km:Q', title='Distance (km)', format='.0f'),
                alt.Tooltip('loss_category:N', title='Signal Quality')
            ]
        ).add_params(
            _freq_selection
        ).properties(
            width=800,
            height=250,
            title='Absorption Loss by Frequency (hover over bar to highlight corresponding ray)'
        )

        # Add text labels on bars with conditional opacity
        _text = alt.Chart(_df_loss).mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=10,
            fontWeight='bold'
        ).encode(
            x=alt.X('frequency:O'),
            y=alt.Y('loss_dB:Q'),
            text=alt.Text('loss_dB:Q', format='.1f'),
            opacity=alt.condition(_freq_selection, alt.value(1.0), alt.value(0.5))
        ).add_params(
            _freq_selection
        )

        _loss_chart = _bar_chart + _text
    else:
        # No data message
        _loss_chart = alt.Chart(pd.DataFrame({'text': ['No rays returned to Earth']})).mark_text(
            size=16,
            align='center'
        ).encode(
            text='text:N'
        ).properties(
            width=800,
            height=250,
            title='Absorption Loss by Frequency'
        )

    # Combine charts vertically
    (_ray_chart + _endpoint_chart) & _loss_chart
    return


@app.cell
def _(mo):
    # ============================================================================
    # SECTION DIVIDER: 2D MODEL WITH IONOSPHERIC TILTS
    # ============================================================================
    mo.md(r"""
    ---

    # Part 2: 2D Model with Ionospheric Tilts

    The sections above showed a **1D ionosphere model** where electron density varies only with altitude. In reality, the ionosphere also has **horizontal gradients** that cause fascinating propagation effects.

    This 2D model includes:
    - **Horizontal foF2 gradient**: Varies linearly from transmitter to 3000 km distance
    - **Off-great-circle propagation**: Radio waves bend sideways when encountering horizontal gradients
    - **Day/night terminator simulation**: Model the sharp density change at sunrise/sunset boundaries
    - **Pedersen rays**: High-angle rays that return at moderate distances due to horizontal gradients

    ---
    """)
    return


@app.cell
def _(mo):
    # ============================================================================
    # UI CONTROLS: 2D MODEL SLIDERS
    # ============================================================================

    # 2D tilt mode sliders (distance-based gradient)
    foF2_at_tx_slider = mo.ui.slider(
        start=2.0,
        stop=25.0,
        step=0.1,
        value=15.0,
        label="foF2 at Transmitter (0 km)"
    )

    foF2_at_distance_slider = mo.ui.slider(
        start=2.0,
        stop=25.0,
        step=0.1,
        value=4.0,
        label="foF2 at Reference Distance"
    )

    # Distance slider for gradient reference point
    tilt_distance_slider = mo.ui.slider(
        start=200.0,
        stop=8000.0,
        step=20.0,
        value=3000.0,
        label="Reference Distance (km)"
    )

    # Elevation angle slider for 2D model
    elevation_slider_2D = mo.ui.slider(
        start=1,
        stop=89,
        step=1,
        value=30,
        label="Elevation Angle (degrees)"
    )

    # Display descriptive text
    mo.vstack([
        mo.md("## Interactive Controls - 2D Model"),
        mo.md("""
        Adjust the sliders below to explore how horizontal ionospheric gradients affect radio propagation:
        - **foF2 at Transmitter (0 km)**: Critical frequency at the starting point
        - **foF2 at Reference Distance**: Critical frequency at the reference distance (creates linear gradient)
        - **Reference Distance**: Distance over which the foF2 gradient occurs (500-8000 km)
        - **Elevation angle**: Launch angle of the radio wave above the horizon

        **Suggested experiments**:
        - **Day/night terminator**: Set foF2 at TX = 15 MHz, foF2 at distance = 4 MHz, distance = 3000 km (sharp terminator)
        - **Gradual transition**: Set foF2 at TX = 12 MHz, foF2 at distance = 6 MHz, distance = 6000 km (gentle gradient)
        - **Short-range gradient**: Set distance = 1000 km for steep ionospheric tilt over short distances
        """),
    ])
    return (
        elevation_slider_2D,
        foF2_at_distance_slider,
        foF2_at_tx_slider,
        tilt_distance_slider,
    )


@app.cell
def _(
    elevation_slider_2D,
    foF2_at_distance_slider,
    foF2_at_tx_slider,
    mo,
    tilt_distance_slider,
):
    # ============================================================================
    # SLIDER DISPLAY WITH VALUES (2D MODEL)
    # ============================================================================

    # Get current slider values
    _current_foF2_tx = foF2_at_tx_slider.value
    _current_foF2_dist = foF2_at_distance_slider.value
    _current_distance = tilt_distance_slider.value
    _current_elev_2D = elevation_slider_2D.value

    # Display sliders with value labels
    mo.vstack([
        mo.md(f"**foF2 at Transmitter ({_current_foF2_tx} MHz)**"),
        foF2_at_tx_slider,
        mo.md(""),  # Spacer
        mo.md(f"**foF2 at Reference Distance ({_current_foF2_dist} MHz)**"),
        foF2_at_distance_slider,
        mo.md(""),
        mo.md(f"**Reference Distance ({_current_distance:.0f} km)**"),
        tilt_distance_slider,
        mo.md(""),
        mo.md(f"**Gradient**: {_current_foF2_tx} MHz → {_current_foF2_dist} MHz over {_current_distance:.0f} km ({(_current_foF2_dist - _current_foF2_tx)/_current_distance:.4f} MHz/km)"),
        mo.md(""),  # Spacer
        mo.md(f"**Elevation Angle ({_current_elev_2D}°)**"),
        elevation_slider_2D,
    ])
    return


@app.cell
def _(
    elevation_slider_2D,
    foF2_at_distance_slider,
    foF2_at_tx_slider,
    make_refractive_index_func,
    make_tilted_ionosphere_2D,
    plots,
    tilt_distance_slider,
    trace_ray_2D_with_tilts,
):
    # ============================================================================
    # 2D IONOSPHERIC TILTS - SIDE VIEW VISUALIZATION
    # ============================================================================

    # Get current slider values
    _tilt_foF2_tx = foF2_at_tx_slider.value
    _tilt_foF2_dist = foF2_at_distance_slider.value
    _tilt_distance = tilt_distance_slider.value
    _tilt_elev = elevation_slider_2D.value

    # Build 2D tilted ionosphere with distance-based gradient
    _tilt_electron_density = make_tilted_ionosphere_2D(_tilt_foF2_tx, _tilt_foF2_dist, _tilt_distance)
    _tilt_refractive_index = make_refractive_index_func(_tilt_electron_density, is_2D=True)

    # Trace rays for all HF frequencies
    _tilt_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Collect ray data
    _ray_data_2D = []
    for _f_tilt in _tilt_freqs:
        _x_tilt, _z_tilt, _phi_tilt, _status_tilt = trace_ray_2D_with_tilts(
            _f_tilt, _tilt_elev, _tilt_refractive_index,
            max_distance_km=8000.0, step_km=5.0
        )
        _ray_data_2D.append({
            'frequency': _f_tilt,
            'x_path': _x_tilt,
            'z_path': _z_tilt,
            'phi_path': _phi_tilt,
            'status': _status_tilt
        })

    # Create side view visualization using plots module
    plot_2D_side_view(_ray_data_2D, _tilt_foF2_tx, _tilt_foF2_dist, _tilt_elev, _tilt_distance)
    return


@app.cell
def _():
    # trace_ray_2D_with_absorption already imported above
    return


@app.cell
def _(
    elevation_slider_2D,
    foF2_at_distance_slider,
    foF2_at_tx_slider,
    make_refractive_index_func,
    make_tilted_ionosphere_2D,
    plots,
    tilt_distance_slider,
    trace_ray_2D_with_tilts,
):
    # ============================================================================
    # 2D IONOSPHERIC TILTS - TOP VIEW (AZIMUTH DEFLECTION)
    # ============================================================================

    # Get current slider values
    _azimuth_foF2_tx = foF2_at_tx_slider.value
    _azimuth_foF2_dist = foF2_at_distance_slider.value
    _azimuth_distance = tilt_distance_slider.value
    _azimuth_elev = elevation_slider_2D.value

    # Build 2D tilted ionosphere with distance-based gradient
    _azimuth_electron_density = make_tilted_ionosphere_2D(_azimuth_foF2_tx, _azimuth_foF2_dist, _azimuth_distance)
    _azimuth_refractive_index = make_refractive_index_func(_azimuth_electron_density, is_2D=True)

    # Trace rays for all HF frequencies
    _azimuth_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Collect ray data
    _ray_data_2D_azimuth = []
    for _f_azimuth in _azimuth_freqs:
        _x_azimuth, _z_azimuth, _phi_azimuth, _status_azimuth = trace_ray_2D_with_tilts(
            _f_azimuth, _azimuth_elev, _azimuth_refractive_index,
            max_distance_km=8000.0, step_km=5.0
        )
        _ray_data_2D_azimuth.append({
            'frequency': _f_azimuth,
            'x_path': _x_azimuth,
            'z_path': _z_azimuth,
            'phi_path': _phi_azimuth,
            'status': _status_azimuth
        })

    # Create top view visualization using plots module
    plot_2D_top_view(_ray_data_2D_azimuth, _azimuth_foF2_tx, _azimuth_foF2_dist, _azimuth_elev, _azimuth_distance)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
