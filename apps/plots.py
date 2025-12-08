"""
Plotting Functions for CondX Ionospheric Propagation Model

This module contains all Altair chart creation functions used by the condx.py
marimo notebook for interactive visualizations.
"""

import altair as alt
import pandas as pd
import numpy as np

# ============================================================================
# CHART STYLING CONSTANTS
# ============================================================================

# Standard legend styling for consistency across all charts
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
