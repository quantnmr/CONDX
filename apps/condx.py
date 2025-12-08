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
    import altair as alt
    import pandas as pd
    import model
    import plots
    return alt, mo, np, pd, plots


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
    plots.plot_electron_density(_z_array, _Ne_D, _Ne_E, _Ne_F1, _Ne_F2, _Ne_total, _fp_total, _density_foF2)
    return


@app.cell
def _():
    # Import physics model functions from external module
    from model import (
        chapman_layer,
        make_ionosphere_for_foF2,
        make_refractive_index_func,
        make_tilted_ionosphere_2D,
        plasma_frequency_Hz_from_Ne,
        trace_ray_2D_with_tilts,
        trace_ray_spherical_with_path,
        trace_ray_with_absorption,
        R_E,
        pi
    )
    return (
        chapman_layer,
        make_ionosphere_for_foF2,
        make_refractive_index_func,
        make_tilted_ionosphere_2D,
        plasma_frequency_Hz_from_Ne,
        trace_ray_2D_with_tilts,
        trace_ray_spherical_with_path,
        trace_ray_with_absorption,
    )


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
    plots.plot_2D_side_view(_ray_data_2D, _tilt_foF2_tx, _tilt_foF2_dist, _tilt_elev, _tilt_distance)
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
):
    # ============================================================================
    # 2D ABSORPTION LOSS VISUALIZATION
    # ============================================================================

    # Import the absorption tracing function
    from model import trace_ray_2D_with_absorption

    # Get current slider values
    _abs_foF2_tx = foF2_at_tx_slider.value
    _abs_foF2_dist = foF2_at_distance_slider.value
    _abs_distance = tilt_distance_slider.value
    _abs_elev = elevation_slider_2D.value

    # Build 2D tilted ionosphere
    _abs_electron_density = make_tilted_ionosphere_2D(_abs_foF2_tx, _abs_foF2_dist, _abs_distance)
    _abs_refractive_index = make_refractive_index_func(_abs_electron_density, is_2D=True)

    # Trace rays with absorption tracking
    _abs_freqs = [1.8, 3.5, 5.3, 7.0, 10.1, 14.0, 18.068, 21.0, 24.89, 28.0]

    # Collect ray data with absorption
    _ray_data_2D_abs = []
    for _f_abs in _abs_freqs:
        _x_abs, _z_abs, _phi_abs, _status_abs, _loss_abs = trace_ray_2D_with_absorption(
            _f_abs, _abs_elev, _abs_refractive_index,
            max_distance_km=8000.0, step_km=5.0
        )
        _ray_data_2D_abs.append({
            'frequency': _f_abs,
            'x_path': _x_abs,
            'z_path': _z_abs,
            'phi_path': _phi_abs,
            'status': _status_abs,
            'loss_dB': _loss_abs
        })

    # Create visualization using plots module
    plots.plot_2D_absorption(_ray_data_2D_abs, _abs_foF2_tx, _abs_foF2_dist, _abs_elev, _abs_distance)
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
    plots.plot_2D_top_view(_ray_data_2D_azimuth, _azimuth_foF2_tx, _azimuth_foF2_dist, _azimuth_elev, _azimuth_distance)
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
