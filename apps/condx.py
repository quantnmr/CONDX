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
    # Import required libraries for ionospheric HF radio wave propagation modeling
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np


@app.cell
def _(np):
    # Physical constants
    c = 3.0e8         # Speed of light in m/s
    pi = np.pi        
    R_E = 6371.0      # Earth's radius in km (spherical Earth model)
    return R_E, pi


@app.cell
def _(mo):
    mo.md(r"""
    # Understanding HF Radio Wave Propagation Through the Ionosphere

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
      - Varies with solar activity, time of day, season (typical range: 3-15 MHz)
    - **Elevation angle**: Launch angle of the radio wave above the horizon
      - Low angles (10-20°) → long skip distances (2000+ km)
      - High angles (60-80°) → short skip distances (< 500 km)

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

    **Below you'll find three interactive sections:**

    1. **Electron Density Profile** - Shows how electron density varies with altitude across the four ionospheric layers (D, E, F1, F2)

    2. **Interactive Controls** - Sliders to adjust foF2 (ionospheric conditions) and elevation angle (launch angle)

    3. **Ray Path Visualization** - Shows how radio waves at different frequencies propagate through the ionosphere

    4. **Signal Loss Analysis** - Displays absorption loss for each frequency with color-coded signal strength

    **Try experimenting with the sliders to see:**
    - How higher foF2 allows higher frequencies to propagate
    - How lower elevation angles create longer skip distances
    - Why some frequencies work better than others (lower absorption loss)
    - How rays at different frequencies bend differently through the ionosphere
    """)
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
    alt,
    chapman_layer,
    foF2_slider,
    make_ionosphere_for_foF2,
    np,
    pd,
    plasma_frequency_Hz_from_Ne,
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

    # Calculate individual layer contributions
    _Ne_D = chapman_layer(_z_array, 1.5e9, 75.0, 8.0)
    _Ne_E = chapman_layer(_z_array, 8e10, 110.0, 15.0)
    _Ne_F1 = chapman_layer(_z_array, 3e11, 190.0, 30.0)
    _Ne_F2 = chapman_layer(_z_array, _Ne_peak_m3, 300.0, 55.0)
    _Ne_total = _total_density_func(_z_array)

    # Also calculate plasma frequency for reference
    _fp_total = plasma_frequency_Hz_from_Ne(_Ne_total) / 1e6  # Convert to MHz

    # Prepare data for Altair - electron density by layer
    _density_data = []
    for _i, _z in enumerate(_z_array):
        _density_data.append({'altitude': _z, 'density': _Ne_D[_i] / 1e9, 'layer': 'D layer'})
        _density_data.append({'altitude': _z, 'density': _Ne_E[_i] / 1e9, 'layer': 'E layer'})
        _density_data.append({'altitude': _z, 'density': _Ne_F1[_i] / 1e9, 'layer': 'F1 layer'})
        _density_data.append({'altitude': _z, 'density': _Ne_F2[_i] / 1e9, 'layer': 'F2 layer'})
        _density_data.append({'altitude': _z, 'density': _Ne_total[_i] / 1e9, 'layer': 'Total'})

    _df_density = pd.DataFrame(_density_data)

    # Prepare data for plasma frequency plot (single curve, not multiple layers)
    _df_plasma = pd.DataFrame({
        'altitude': _z_array,
        'plasma_freq': _fp_total
    })

    # Create layer selection for hover highlighting
    _layer_selection = alt.selection_point(
        fields=['layer'],
        bind='legend',
        on='mouseover',
        nearest=True
    )

    # Left chart: Electron density by layer
    _density_chart = alt.Chart(_df_density).mark_line(size=2).encode(
        x=alt.X('density:Q', 
                title='Electron Density (× 10⁹ electrons/m³)',
                scale=alt.Scale(domain=[0, _df_density['density'].max() * 1.05])),
        y=alt.Y('altitude:Q', 
                title='Altitude (km)',
                scale=alt.Scale(domain=[0, 600])),
        color=alt.Color('layer:N',
                       legend=alt.Legend(title='Layer'),
                       scale=alt.Scale(
                           domain=['D layer', 'E layer', 'F1 layer', 'F2 layer', 'Total'],
                           range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#000000']
                       )),
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
        opacity=alt.condition(_layer_selection, alt.value(1.0), alt.value(0.3)),
        detail='layer:N',
        tooltip=[
            alt.Tooltip('layer:N', title='Layer'),
            alt.Tooltip('altitude:Q', title='Altitude (km)', format='.0f'),
            alt.Tooltip('density:Q', title='Density (×10⁹ e/m³)', format='.3f')
        ]
    ).add_params(
        _layer_selection
    ).properties(
        width=400,
        height=400,
        title=f'Ionospheric Layers (foF2 = {_density_foF2} MHz)'
    )

    # Add layer peak labels
    _idx_D = np.argmax(_Ne_D)
    _idx_E = np.argmax(_Ne_E)
    _idx_F1 = np.argmax(_Ne_F1)
    _idx_F2 = np.argmax(_Ne_F2)

    _layer_labels = pd.DataFrame([
        {'density': _Ne_D[_idx_D] / 1e9, 'altitude': _z_array[_idx_D], 'label': 'D (~75 km)', 'color': '#1f77b4'},
        {'density': _Ne_E[_idx_E] / 1e9, 'altitude': _z_array[_idx_E], 'label': 'E (~110 km)', 'color': '#ff7f0e'},
        {'density': _Ne_F1[_idx_F1] / 1e9, 'altitude': _z_array[_idx_F1], 'label': 'F1 (~190 km)', 'color': '#2ca02c'},
        {'density': _Ne_F2[_idx_F2] / 1e9, 'altitude': _z_array[_idx_F2], 'label': 'F2 (~300 km)', 'color': '#d62728'}
    ])

    # Create bordered label boxes with text
    # Add computed box coordinates
    _box_width_val = _df_density['density'].max() * 0.25
    _box_height_val = 20

    _layer_labels_boxes = []
    for _, row in _layer_labels.iterrows():
        _layer_labels_boxes.append({
            'x_min': row['density'] + 2,
            'x_max': row['density'] + _box_width_val,
            'y_min': row['altitude'] - _box_height_val / 2,
            'y_max': row['altitude'] + _box_height_val / 2,
            'x_center': row['density'] + _box_width_val / 2,
            'y_center': row['altitude'],
            'label': row['label'],
            'color': row['color']
        })
    _df_label_boxes = pd.DataFrame(_layer_labels_boxes)

    # Create background boxes
    _label_boxes = alt.Chart(_df_label_boxes).mark_rect(
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

    _label_text = alt.Chart(_df_label_boxes).mark_text(
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

    _density_chart_with_labels = _density_chart + _label_boxes + _label_text

    # Right chart: Plasma frequency profile (explicitly sort and use order encoding)
    _plasma_chart = alt.Chart(_df_plasma).mark_line(size=2.5, color='darkblue').encode(
        x=alt.X('plasma_freq:Q', 
                title='Plasma Frequency (MHz)',
                scale=alt.Scale(domain=[0, _df_plasma['plasma_freq'].max() * 1.05])),
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
    _foF2_line = alt.Chart(pd.DataFrame({'foF2': [_density_foF2]})).mark_rule(
        color='red',
        strokeDash=[5, 5],
        size=2
    ).encode(
        x='foF2:Q'
    )

    # Add peak annotation
    _peak_idx = np.argmax(_fp_total)
    _peak_annotation = alt.Chart(pd.DataFrame([{
        'plasma_freq': _df_plasma['plasma_freq'].max() * 0.3,
        'altitude': 100,
        'text': f'Peak: {_fp_total.max():.1f} MHz\nat {_z_array[_peak_idx]:.0f} km\n\nfoF2 = {_density_foF2} MHz (red line)'
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
    (_density_chart_with_labels | (_plasma_chart + _foF2_line + _peak_annotation))
    return


@app.cell
def _(R_E, np, pi):
    # ============================================================================
    # IONOSPHERE MODEL FUNCTIONS
    # ============================================================================

    def chapman_layer(z_km, Nmax, hm, H):
        """
        Chapman layer function - models electron density profile for a single ionospheric layer.

        The Chapman function describes how electron density varies with altitude in a layer
        that's produced by solar radiation absorption in the atmosphere.

        Parameters:
        -----------
        z_km : array-like
            Altitude in km
        Nmax : float
            Peak electron density in electrons/m^3
        hm : float
            Peak altitude in km (where maximum density occurs)
        H : float
            Scale height in km (controls layer thickness/width)

        Returns:
        --------
        Ne : array
            Electron density in electrons/m^3 at each altitude
        """
        z = np.asarray(z_km)
        xi = (z - hm) / H
        Ne = Nmax * np.exp(0.5 * (1.0 - xi - np.exp(-xi)))
        Ne[z < 0] = 0.0  # No ionosphere below ground level
        return Ne

    def make_ionosphere_for_foF2(foF2_MHz):
        """
        Constructs a 4-layer ionosphere model (D, E, F1, F2) for a given foF2 value.

        The ionosphere consists of multiple layers at different altitudes:
        - D layer (~75 km): Lowest layer, high absorption, mainly affects lower HF
        - E layer (~110 km): Regular daytime layer, supports some HF skip
        - F1 layer (~190 km): Daytime only, merges with F2 at night
        - F2 layer (~300 km): Highest/most important for HF, peak density varies with foF2

        Parameters:
        -----------
        foF2_MHz : float
            Critical frequency of F2 layer in MHz. This is the maximum frequency that
            can be reflected by the F2 layer at vertical incidence. Typical values:
            - Night/low solar activity: 3-6 MHz
            - Day/moderate activity: 6-12 MHz  
            - Day/high solar activity: 12-15+ MHz

        Returns:
        --------
        total_electron_density : function
            Function that takes altitude z_km and returns total electron density
        """
        # Convert foF2 to peak electron density using the plasma frequency relation
        # fp(MHz) = 8.98 * sqrt(Ne_cm3), so Ne = (fp/8.98)^2
        foF2_Hz = foF2_MHz * 1e6
        Ne_peak_cm3 = (foF2_Hz / 8.98e3) ** 2
        Ne_peak_m3 = Ne_peak_cm3 * 1e6

        def total_electron_density(z_km):
            """
            Returns total electron density by summing all four ionospheric layers.

            Layer definitions (Nmax, hm, H):
            - D layer:  1.5e9 electrons/m³,  75 km peak,  8 km scale height (calibrated to match real-world absorption)
            - E layer:  8e10 electrons/m³, 110 km peak, 15 km scale height
            - F1 layer: 3e11 electrons/m³, 190 km peak, 30 km scale height
            - F2 layer: Variable (from foF2), 300 km peak, 55 km scale height
            """
            z = np.asarray(z_km)
            Ne_total = (
                chapman_layer(z, 1.5e9, 75.0, 8.0) +      # D layer (reduced from 5e9 to match real absorption)
                chapman_layer(z, 8e10, 110.0, 15.0) +     # E layer
                chapman_layer(z, 3e11, 190.0, 30.0) +     # F1 layer
                chapman_layer(z, Ne_peak_m3, 300.0, 55.0) # F2 layer (varies with foF2)
            )
            return Ne_total

        return total_electron_density

    def plasma_frequency_Hz_from_Ne(Ne_m3):
        """
        Calculate plasma frequency from electron density.

        The plasma frequency is the natural oscillation frequency of electrons in the
        ionosphere. Radio waves below the plasma frequency are reflected.

        Formula: fp(Hz) = 8.98e3 * sqrt(Ne_cm³)

        Parameters:
        -----------
        Ne_m3 : float or array
            Electron density in electrons/m³

        Returns:
        --------
        fp : float or array
            Plasma frequency in Hz
        """
        Ne_cm3 = Ne_m3 * 1e-6
        return 8.98e3 * np.sqrt(Ne_cm3)

    def collision_frequency_Hz(z_km):
        """
        Calculate electron-neutral collision frequency (affects radio wave absorption).

        Collisions between electrons and neutral particles cause radio wave energy loss.
        Higher collision rates → more absorption. Collision frequency decreases rapidly
        with altitude as the atmosphere thins.

        Based on literature: typical D-layer collision frequency ~1-2×10^7 Hz at 70-80 km

        Parameters:
        -----------
        z_km : float or array
            Altitude in km

        Returns:
        --------
        nu : float or array
            Collision frequency in Hz
        """
        z = np.asarray(z_km)
        nu = np.zeros_like(z, dtype=float)
        # Below 90 km (D-layer region): high collisions, exponentially decreasing
        # This is where most absorption occurs for low frequencies
        # Calibrated to match real-world absorption: ~35 dB at 1.8 MHz, ~25 dB at 3.5 MHz
        mask_low = z < 90.0
        nu[mask_low] = 3.5e5 * np.exp(-(z[mask_low] - 70.0) / 8.0)
        # 90-150 km: rapidly decreasing collisions
        mask_mid = (z >= 90.0) & (z < 150.0)
        nu[mask_mid] = 1e5 * np.exp(-(z[mask_mid] - 90.0) / 20.0)
        # Above 150 km: very low collision rate (negligible absorption)
        nu[z >= 150.0] = 1e3
        return nu

    def make_refractive_index_func(total_electron_density):
        """
        Creates a function to calculate the complex refractive index of the ionosphere.

        The refractive index determines how radio waves bend and are absorbed as they
        propagate through the ionosphere. It depends on:
        - Electron density (plasma frequency)
        - Radio wave frequency
        - Collision frequency (absorption)

        Returns:
        --------
        refractive_index_complex : function
            Function that takes (z_km, f_Hz) and returns (n, n²) as complex values
        """
        def refractive_index_complex(z_km, f_Hz):
            """
            Calculate complex refractive index using Appleton-Hartree equation (simplified).

            The real part of n determines ray bending (refraction)
            The imaginary part determines absorption

            When n² < 0, the wave cannot propagate (reflection occurs)
            """
            Ne = total_electron_density(z_km)
            fp = plasma_frequency_Hz_from_Ne(Ne)
            omega_p = 2.0 * pi * fp  # Angular plasma frequency
            omega = 2.0 * pi * f_Hz   # Angular wave frequency

            X = (omega_p / omega) ** 2  # Normalized plasma frequency
            nu = collision_frequency_Hz(z_km)
            Z = nu / omega               # Normalized collision frequency

            # Appleton-Hartree formula (no magnetic field, ordinary wave)
            n2 = 1.0 - X / (1.0 - 1j * Z)

            # Prevent numerical issues with very small refractive indices
            n2 = np.where(np.real(n2) > 1e-6, n2, 1e-6 + 0j)
            n = np.sqrt(n2)
            return n, n2
        return refractive_index_complex

    # ============================================================================
    # RAY TRACING FUNCTION
    # ============================================================================

    def trace_ray_spherical_fast(f_MHz, elev_deg, refractive_index_complex,
                                  s_max_km=6000.0, ds_km=10.0, z_top_km=600.0):
        """
        Trace an HF radio ray through the ionosphere using spherical geometry.

        This function simulates the path of a radio wave launched from Earth's surface
        at a given elevation angle and frequency. It uses Snell's law in spherical
        coordinates to determine how the wave bends through the varying ionosphere.

        Parameters:
        -----------
        f_MHz : float
            Radio frequency in MHz
        elev_deg : float
            Launch elevation angle in degrees (0° = horizontal, 90° = vertical)
        refractive_index_complex : function
            Function to calculate refractive index at any altitude
        s_max_km : float
            Maximum ray path length in km before giving up
        ds_km : float
            Step size for ray tracing in km
        z_top_km : float
            Altitude above which ray is considered to have escaped

        Returns:
        --------
        distance : float
            Great circle distance traveled (skip distance) in km
        status : str
            "returns" = ray returned to Earth (successful skip)
            "escapes" = ray passed through ionosphere without returning
            "stops" = ray was absorbed or max distance reached
        """
        f_Hz = f_MHz * 1e6

        # Initial position in spherical coordinates (r, theta)
        r = R_E           # Start at Earth's surface
        theta = 0.0       # Angular position along Earth

        # Calculate ray parameter b = n*r*sin(ψ) (conserved by Snell's law in spherical geometry)
        # ψ is the angle between the ray and the radial direction
        psi0 = np.radians(90.0 - elev_deg)  # Convert elevation to ray angle
        n0_arr, _ = refractive_index_complex(np.array([0.0]), f_Hz)
        b = float(np.real(n0_arr[0])) * r * np.sin(psi0)  # Ray parameter (constant along path)

        s = 0.0           # Path length traveled
        going_up = True   # Ray initially going upward

        # Step along ray path
        while s < s_max_km:
            z = r - R_E   # Current altitude

            # Check if ray has returned to ground
            if z < 0:
                break

            # Get refractive index at current altitude
            n_arr, n2_arr = refractive_index_complex(np.array([z]), f_Hz)
            n_r = float(np.real(n_arr[0]))
            n2_r = float(np.real(n2_arr[0]))

            # If n² ≤ 0, wave cannot propagate → reflection point
            if n2_r <= 0:
                going_up = False

            # Apply Snell's law: sin(ψ) = b/(n*r)
            sin_psi = b / (n_r * r)
            if abs(sin_psi) >= 1:  # Ray turning around (reflection)
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999  # Clamp to valid range

            cos_psi = np.sqrt(1.0 - sin_psi**2)

            # Calculate incremental changes in position
            dr = ds_km * cos_psi * (1 if going_up else -1)  # Radial step
            dtheta = ds_km * sin_psi / r                     # Angular step

            r_new = r + dr
            theta_new = theta + dtheta

            # Check if ray has returned to Earth's surface
            if not going_up and r_new < R_E:
                # Interpolate exact ground intersection point
                frac = (r - R_E) / (r - r_new)
                theta = theta + frac * dtheta
                return R_E * theta, "returns"  # Skip distance = arc length

            r, theta = r_new, theta_new
            s += ds_km

            # Check if ray escaped through top of ionosphere
            if going_up and (r - R_E) > z_top_km:
                return R_E * theta, "escapes"

        # Ray didn't return or escape within distance limit
        return R_E * theta, "stops"
    def trace_ray_spherical_with_path(f_MHz, elev_deg, refractive_index_complex,
                                       s_max_km=6000.0, ds_km=10.0, z_top_km=600.0):
        """
        Trace an HF radio ray and record its entire path for visualization.

        Similar to trace_ray_spherical_fast but records (x, z) coordinates along the path.

        Returns:
        --------
        x_km : list
            Surface distance (great circle distance) at each point along path
        z_km : list
            Altitude above ground at each point along path
        status : str
            "returns", "escapes", or "stops"
        """
        f_Hz = f_MHz * 1e6

        # Initial position
        r = R_E
        theta = 0.0

        # Calculate ray parameter
        psi0 = np.radians(90.0 - elev_deg)
        n0_arr, _ = refractive_index_complex(np.array([0.0]), f_Hz)
        b = float(np.real(n0_arr[0])) * r * np.sin(psi0)

        s = 0.0
        going_up = True

        # Record path
        x_path = [R_E * theta]
        z_path = [r - R_E]

        while s < s_max_km:
            z = r - R_E

            if z < 0:
                break

            n_arr, n2_arr = refractive_index_complex(np.array([z]), f_Hz)
            n_r = float(np.real(n_arr[0]))
            n2_r = float(np.real(n2_arr[0]))

            if n2_r <= 0:
                going_up = False

            sin_psi = b / (n_r * r)
            if abs(sin_psi) >= 1:
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999

            cos_psi = np.sqrt(1.0 - sin_psi**2)
            dr = ds_km * cos_psi * (1 if going_up else -1)
            dtheta = ds_km * sin_psi / r

            r_new = r + dr
            theta_new = theta + dtheta

            if not going_up and r_new < R_E:
                frac = (r - R_E) / (r - r_new)
                theta_final = theta + frac * dtheta
                x_path.append(R_E * theta_final)
                z_path.append(0.0)
                return x_path, z_path, "returns"

            r, theta = r_new, theta_new
            s += ds_km

            # Record this point
            x_path.append(R_E * theta)
            z_path.append(r - R_E)

            if going_up and (r - R_E) > z_top_km:
                return x_path, z_path, "escapes"

        return x_path, z_path, "stops"

    def trace_ray_with_absorption(f_MHz, elev_deg, refractive_index_complex,
                                    s_max_km=6000.0, ds_km=10.0, z_top_km=600.0):
        """
        Trace an HF radio ray and calculate total signal loss due to absorption.

        This function traces the ray path and accumulates absorption loss by integrating
        the imaginary part of the refractive index along the path. The imaginary part
        of n represents wave attenuation due to collisions in the ionosphere.

        Parameters:
        -----------
        f_MHz : float
            Radio frequency in MHz
        elev_deg : float
            Launch elevation angle in degrees
        refractive_index_complex : function
            Function to calculate refractive index at any altitude
        s_max_km : float
            Maximum ray path length in km
        ds_km : float
            Step size for ray tracing in km
        z_top_km : float
            Altitude above which ray is considered to have escaped

        Returns:
        --------
        x_km : list
            Surface distance at each point along path
        z_km : list
            Altitude at each point along path
        status : str
            "returns", "escapes", or "stops"
        total_loss_dB : float
            Total absorption loss in dB along the ray path
        """
        f_Hz = f_MHz * 1e6
        omega = 2.0 * pi * f_Hz
        c_km_per_s = 3.0e5  # Speed of light in km/s

        # Initial position
        r = R_E
        theta = 0.0

        # Calculate ray parameter
        psi0 = np.radians(90.0 - elev_deg)
        n0_arr, _ = refractive_index_complex(np.array([0.0]), f_Hz)
        b = float(np.real(n0_arr[0])) * r * np.sin(psi0)

        s = 0.0
        going_up = True

        # Record path and absorption
        x_path = [R_E * theta]
        z_path = [r - R_E]
        total_absorption = 0.0  # Accumulated absorption (Nepers)

        while s < s_max_km:
            z = r - R_E

            if z < 0:
                break

            n_arr, n2_arr = refractive_index_complex(np.array([z]), f_Hz)
            n_r = float(np.real(n_arr[0]))
            n2_r = float(np.real(n2_arr[0]))
            n2_i = float(np.imag(n2_arr[0]))

            # Use Sen-Wyller absorption formula directly
            # α (Nepers/km) = (ω/2c) × |Im(n²)| / Re(n)
            # This correctly gives absorption ∝ 1/f² at high frequencies
            if n_r > 1e-6 and abs(n2_i) > 1e-12:
                alpha = (omega / (2.0 * c_km_per_s)) * abs(n2_i) / n_r
            else:
                alpha = 0.0

            # Accumulate absorption over this step: ∫ α ds
            total_absorption += alpha * ds_km

            if n2_r <= 0:
                going_up = False

            sin_psi = b / (n_r * r)
            if abs(sin_psi) >= 1:
                going_up = False
                sin_psi = np.sign(sin_psi) * 0.999

            cos_psi = np.sqrt(1.0 - sin_psi**2)
            dr = ds_km * cos_psi * (1 if going_up else -1)
            dtheta = ds_km * sin_psi / r

            r_new = r + dr
            theta_new = theta + dtheta

            if not going_up and r_new < R_E:
                # Ray returns to Earth
                frac = (r - R_E) / (r - r_new)
                theta_final = theta + frac * dtheta
                x_path.append(R_E * theta_final)
                z_path.append(0.0)

                # Convert absorption from Nepers to dB: dB = 8.686 × Nepers
                total_loss_dB = 8.686 * total_absorption
                return x_path, z_path, "returns", total_loss_dB

            r, theta = r_new, theta_new
            s += ds_km

            # Record this point
            x_path.append(R_E * theta)
            z_path.append(r - R_E)

            if going_up and (r - R_E) > z_top_km:
                # Ray escapes
                total_loss_dB = 8.686 * total_absorption
                return x_path, z_path, "escapes", total_loss_dB

        # Ray stops
        total_loss_dB = 8.686 * total_absorption
        return x_path, z_path, "stops", total_loss_dB
    return (
        chapman_layer,
        make_ionosphere_for_foF2,
        make_refractive_index_func,
        plasma_frequency_Hz_from_Ne,
        trace_ray_spherical_with_path,
        trace_ray_with_absorption,
    )


@app.cell(hide_code=True)
def _(mo):
    # ============================================================================
    # INTERACTIVE RAY PATH CONTROLS
    # ============================================================================

    # Create sliders for interactive ray path visualization
    # Using on_change=False means sliders only update when mouse is released
    foF2_slider = mo.ui.slider(
        start=3,
        stop=20,
        step=1,
        value=12,
        label="foF2 (MHz)",
        full_width=True
    )

    elevation_slider = mo.ui.slider(
        start=5,
        stop=85,
        step=5,
        value=30,
        label="Elevation Angle (degrees)",
        full_width=True
    )

    # Display sliders in a vertical stack
    mo.vstack([
        mo.md("## Interactive Ray Path Visualization"),
        mo.md("Adjust the sliders to see how ray paths change with different ionospheric conditions and launch angles."),
        foF2_slider,
        elevation_slider
    ])
    return elevation_slider, foF2_slider


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

    for _f_interactive in _interactive_freqs:
        _x_int, _z_int, _status_int = trace_ray_spherical_with_path(
            _f_interactive, _interactive_elev, _interactive_refractive_index
        )

        # Add path data (each point along the ray path)
        for _idx, (_x, _z) in enumerate(zip(_x_int, _z_int)):
            _path_data_interactive.append({
                'frequency': _f_interactive,
                'distance_km': _x,
                'altitude_km': _z,
                'status': _status_int,
                'label': f"{_f_interactive} MHz ({_status_int})",
                'point_index': _idx
            })

        # Mark landing point for returning rays
        if _status_int == "returns":
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
        fields=['frequency'],
        bind='legend',
        on='mouseover',
        nearest=True
    )

    # Sort frequencies for legend ordering
    _freq_order = sorted(_df_paths_interactive['frequency'].unique())

    # Create ray path chart
    _ray_chart_interactive = alt.Chart(_df_paths_interactive).mark_line(size=2.5).encode(
        x=alt.X('distance_km:Q', 
                title='Surface distance (km)',
                scale=alt.Scale(domain=[0, _df_paths_interactive['distance_km'].max() * 1.05])),
        y=alt.Y('altitude_km:Q', 
                title='Height above ground (km)',
                scale=alt.Scale(domain=[0, 600])),
        color=alt.Color('frequency:N',
                       legend=alt.Legend(title='Frequency (MHz)', columns=1),
                       scale=alt.Scale(scheme='category10'),
                       sort=_freq_order),
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
            color=alt.Color('frequency:N', legend=None, scale=alt.Scale(scheme='category10'), sort=_freq_order),
            size=alt.condition(_freq_selection_interactive, alt.value(300), alt.value(150)),
            opacity=alt.condition(_freq_selection_interactive, alt.value(1.0), alt.value(0.7)),
            tooltip=[
                alt.Tooltip('frequency:Q', title='Frequency (MHz)'),
                alt.Tooltip('distance_km:Q', title='Skip Distance (km)', format='.0f'),
                alt.Tooltip('status:N', title='Status')
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
    elevation_slider,
    foF2_slider,
    make_ionosphere_for_foF2,
    make_refractive_index_func,
    trace_ray_with_absorption,
):
    import altair as alt
    import pandas as pd

    _loss_foF2 = foF2_slider.value
    _loss_elev = elevation_slider.value

    _loss_electron_density = make_ionosphere_for_foF2(_loss_foF2)
    _loss_refractive_index = make_refractive_index_func(_loss_electron_density)

    # Amateur radio HF bands (using typical frequencies)
    _loss_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Prepare data for Altair
    _path_data = []
    _loss_summary_data = []

    for _f_loss in _loss_freqs:
        _x_loss, _z_loss, _status_loss, _total_loss = trace_ray_with_absorption(
            _f_loss, _loss_elev, _loss_refractive_index
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

        # Add path data (each point along the ray path)
        for _idx, (_x, _z) in enumerate(zip(_x_loss, _z_loss)):
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

        # Add summary data for bar chart
        if _status_loss == "returns":
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
                       legend=alt.Legend(title='Signal Strength')),
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
    return alt, pd


@app.cell
def _():
    # Export functions needed by new cell
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
