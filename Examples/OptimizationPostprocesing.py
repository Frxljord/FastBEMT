from __future__ import annotations
from os import name
import base64, io, pickle, re
import numpy as np
import matplotlib as mpl
from shapely import geometry

from FastBEMT.JobParameters import LowFidelityParameters
from FastBEMT.Propeller import Propeller
from FastBEMT.Stress import BladeStressCalculator
mpl.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, no_update, callback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from FastBEMT.Section import compute_com

# Assuming this exists in your environment
from FastBEMT.DataLoader import load_propeller_dict

def _repo_root() -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / 'pyproject.toml').exists(): return parent
    raise FileNotFoundError('Repo root not found.')

def _load_run_result(path: Path):
    with path.open('rb') as f:
        res = pickle.load(f)
    return np.asarray(res.F), np.asarray(res.G)

# --- RE-ADDING YOUR GEOMETRY BUILDER ---
def _build_geometry_surface(geometry, sigma_c, sigma_b):
        r = np.asarray(geometry['r'])
        chord = np.asarray(geometry['chord'])
        twist = np.radians(np.asarray(geometry['twist']))
        airfoils = geometry['airfoil']
        n_sections = len(r)
        n_points = airfoils[0].shape[0]

        sigma_c = np.asarray(sigma_c)
        sigma_b = np.asarray(sigma_b)
        
        # Expand sigma_c if needed to match sigma_b dimensions
        if sigma_b.ndim == 2 and sigma_c.ndim == 1:
            sigma_c = sigma_c[:, np.newaxis]
        
        if sigma_b.ndim == 1:
            sigma_total = sigma_c + sigma_b
            sigma_total = sigma_total[:, np.newaxis]
        else:
            sigma_total = sigma_b + sigma_c

        X = np.zeros((n_sections, n_points))
        Y = np.zeros((n_sections, n_points))
        Z = np.zeros((n_sections, n_points))
        S = np.zeros((n_sections, n_points))
        
        # Build blade surface coordinates and stress values
        for i in range(n_sections):
            coords = np.asarray(airfoils[i])
            x_local = coords[:, 0] * chord[i]
            z_local = coords[:, 1] * chord[i]
            x_com, z_com = compute_com(x_local, z_local)
            x_local = x_local - x_com
            z_local = z_local - z_com
            
            if hasattr(geometry, 'com_shift_forward') and hasattr(geometry, 'com_shift_up'):
                x_local = x_local + geometry.com_shift_forward[i] * chord[i]
                z_local = z_local + geometry.com_shift_up[i] * chord[i]
            
            cos_t = np.cos(twist[i])
            sin_t = np.sin(twist[i])
            x_rot = x_local * cos_t + z_local * sin_t
            z_rot = -x_local * sin_t + z_local * cos_t
            
            X[i, :] = x_rot
            Y[i, :] = r[i]
            Z[i, :] = z_rot

            if sigma_total.ndim == 2 and sigma_total.shape[1] == n_points:
                S[i, :] = sigma_total[i, :]
            else:
                S[i, :] = sigma_total[i]
        
        S_mpa = S / 1e6

        return X, Y, Z, S_mpa

def _preview_data_uri(name, geometry, obj, constraint, sigma_c, sigma_b):
    # Ensure we use a non-GUI backend
    plt.switch_backend('Agg') 
    
    X, Y, Z, S_mpa = _build_geometry_surface(geometry, sigma_c, sigma_b)
    
    fig = plt.figure(figsize=(4.5,3))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap_obj = mpl.colormaps['viridis']
    norm = mpl.colors.Normalize(vmin=np.nanmin(S_mpa), vmax=np.nanmax(S_mpa))
    ax.plot_surface(
        X, Y, Z,
        facecolors=cmap_obj(norm(S_mpa)),
        rstride=1, cstride=1,
        linewidth=0, antialiased=True,
        shade=False, edgecolor='none',
    )
    
    ax.set_axis_off()
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    
    # Add colorbar
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(S_mpa)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.4, pad=-0.05, aspect=10)
    cbar.set_label('Stress [MPa]')
    
    ax.view_init(elev=15, azim=-30)
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    
    plt.title(f"{name}\nFoM: {-obj[0]:.4f} | Area: {obj[1]*300:.4f}", fontsize=9)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    
    # Cleanup to prevent the "RuntimeError" and memory growth
    fig.clf()
    plt.close(fig)
    
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# --- DASH APP ---

def build_pareto_dash(run_id: str):
    params = LowFidelityParameters(
        rpm=7000,
        a_inf=343,
        rho=1.225,
        mu=1.81e-5,
        n_blades=2,
        p_ref=2e-5,
        revolutions=5,
        num_obs_times_per_rev=100,
        device='cuda',
    )
    
    root = _repo_root()
    path = root / 'Datasets' / 'Propellers' / f'{run_id}.pkl'
    propellers_list = load_propeller_dict(run_id)

    stress_data = {} 

    for idx, (propeller_name, blade_dict) in enumerate(propellers_list):
        print(f'\nProcessing {propeller_name}...')
        propeller = Propeller(geometry=blade_dict, params=params)
        propeller.run_bemt(v_inf=0)
        propeller.compute_total_forces()
        
        stress_calc = BladeStressCalculator(propeller=propeller)
        # Store the actual stress arrays returned by the calculator
        s_c, s_b = stress_calc.blade_stress_report(material_rho=2700, show=False) 
        stress_data[idx] = (s_c, s_b)
    
    F, G = _load_run_result(path)
    
    # Pareto Logic
    feasible_mask = np.all(G <= 0, axis=1) if G is not None else np.ones(F.shape[0], dtype=bool)
    feasible_idx = np.where(feasible_mask)[0]
    front_rel_idx = NonDominatedSorting().do(F[feasible_mask], only_non_dominated_front=True)
    front_indices = feasible_idx[front_rel_idx]
    
    # Sort for cleaner line plotting
    front_indices = front_indices[np.argsort(F[front_indices, 1])]
    
    fig = go.Figure()

    # Add the Pareto scatter trace
    fig.add_trace(go.Scatter(
        x=F[front_indices, 1]*300, # Area
        y=-F[front_indices, 0], # FoM
        mode='markers+lines',
        marker=dict(size=12, color='#1f77b4', symbol='diamond', line=dict(width=1, color='white')),
        customdata=front_indices.tolist(), 
        hoverinfo="none",
        name="Pareto Front"
    ))

    # Add Improvement Arrows (Annotations)
    fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", text="Higher FoM", showarrow=True, arrowhead=2, ax=0, ay=40)
    fig.add_annotation(x=0.95, y=0.1, xref="paper", yref="paper", text="Smaller Area", showarrow=True, arrowhead=2, ax=40, ay=0)

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Contour Area [m^2]",
        yaxis_title="FoM [-]",
        width=1000, height=700,
        hoverlabel=dict(bgcolor="white")
    )

    app = Dash(__name__)
    app.layout = html.Div([
        html.H3(f"Pareto Front: {run_id}", style={'fontFamily': 'Arial', 'marginLeft': '20px'}),
        dcc.Graph(id="graph", figure=fig, clear_on_unhover=True, config={'displayModeBar': False}),
        dcc.Tooltip(id="tooltip", background_color="white", border_color="#ddd"),
    ])

    @callback(
        Output("tooltip", "show"),
        Output("tooltip", "bbox"),
        Output("tooltip", "children"),
        Input("graph", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None: return False, no_update, no_update
        
        pt = hoverData["points"][0]
        # Get index back from customdata
        orig_idx = pt.get("customdata")
        
        if orig_idx is None: return False, no_update, no_update

        name, geometry = propellers_list[orig_idx]
        s_c, s_b = stress_data[orig_idx]
        img_uri = _preview_data_uri(
            name, geometry, F[orig_idx], G[orig_idx], s_c, s_b
        )        
        return True, pt["bbox"], html.Div([
            html.Img(src=img_uri, style={"width": "100%", "borderRadius": "5px"})], 
            style={'width': '400px'})

    return app

if __name__ == "__main__":
    app = build_pareto_dash('Epsilon_50_50_latent32_Corrected_Stress/run_001')
    app.run(debug=True)