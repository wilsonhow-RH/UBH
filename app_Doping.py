import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines
import matplotlib.tri as mtri
import scipy.ndimage as ndimage
import time
import imageio

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "physics2026": # Change this to your desired password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter Password to access the UHV Dashboard:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter Password to access the UHV Dashboard:", type="password", on_change=password_entered, key="password")
        st.error("Incorrect Password")
        return False
    return True

if not check_password():
    st.stop() # Stops the rest of the code from running until password is correct
# ---------------------------

# ... (The rest of your Moire dashboard code goes here) ...

st.set_page_config(page_title="UHV-bonded Heterostructure Physics Dashboard", layout="wide")

st.title("UHV-bonded Heterostructure Physics Dashboard")
st.markdown("Explore the topology, geometry, scattering, local doping level and many-body interactions of 2D UHV-bonded heterostructures (by Gemini & Ruihua He, 5/1/26).")

# ==========================================
# 1. PARAMETERS & PRE-COMPUTATION
# ==========================================
@st.cache_data
def generate_base_grids():
    a_sto, a_fese, a_mos2, a_bise, a_g = 3.905, 3.905, 3.15, 4.14, 2.46          
    base_grid = 40  
    max_grid = base_grid * 6 

    x_sq = np.arange(-max_grid, max_grid, a_sto)
    y_sq = np.arange(-max_grid, max_grid, a_sto)
    xx_sq, yy_sq = np.meshgrid(x_sq, y_sq)
    pts_sq_base = np.vstack([xx_sq.ravel(), yy_sq.ravel()]).T

    v1_m = np.array([a_mos2, 0])
    v2_m = np.array([a_mos2 * np.cos(np.pi/3), a_mos2 * np.sin(np.pi/3)])
    n_m = np.arange(-int(max_grid/a_mos2)*2, int(max_grid/a_mos2)*2)
    nn_m, mm_m = np.meshgrid(n_m, n_m)
    pts_mos2_base = (nn_m.reshape(-1, 1) * v1_m + mm_m.reshape(-1, 1) * v2_m)
    pts_mos2_base = pts_mos2_base[(np.abs(pts_mos2_base[:, 0]) < max_grid*1.5) & (np.abs(pts_mos2_base[:, 1]) < max_grid*1.5)]

    V_bise = np.array([[a_bise, a_bise*0.5], [0, a_bise*np.sqrt(3)/2]])
    invV_bise = np.linalg.inv(V_bise)
    n_b = np.arange(-int(max_grid/a_bise)*2, int(max_grid/a_bise)*2)
    nn_b, mm_b = np.meshgrid(n_b, n_b)
    pts_bise_base = (nn_b.reshape(-1, 1) * V_bise[:,0] + mm_b.reshape(-1, 1) * V_bise[:,1])
    pts_bise_base = pts_bise_base[(np.abs(pts_bise_base[:, 0]) < max_grid*1.5) & (np.abs(pts_bise_base[:, 1]) < max_grid*1.5)]

    V_g = np.array([[a_g, a_g*0.5], [0, a_g*np.sqrt(3)/2]])
    invV_g = np.linalg.inv(V_g)
    n_g = np.arange(-int(max_grid/a_g)*2, int(max_grid/a_g)*2)
    nn_g, mm_g = np.meshgrid(n_g, n_g)
    pts_grap_base = (nn_g.reshape(-1, 1) * V_g[:,0] + mm_g.reshape(-1, 1) * V_g[:,1])
    pts_grap_base = pts_grap_base[(np.abs(pts_grap_base[:, 0]) < max_grid*1.5) & (np.abs(pts_grap_base[:, 1]) < max_grid*1.5)]

    N_fft = 1024
    L_fft = 400.0  
    x_fft = np.linspace(-L_fft/2, L_fft/2, N_fft)
    y_fft = np.linspace(-L_fft/2, L_fft/2, N_fft)
    X_fft, Y_fft = np.meshgrid(x_fft, y_fft)
    q_freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=(L_fft/N_fft))) * 2 * np.pi
    window_1d = np.hanning(N_fft)
    window_2d = window_1d[:, np.newaxis] * window_1d[np.newaxis, :]
    
    return pts_sq_base, pts_mos2_base, pts_bise_base, pts_grap_base, V_bise, invV_bise, V_g, invV_g, X_fft, Y_fft, q_freq, window_2d

pts_sq_base, pts_mos2_base, pts_bise_base, pts_grap_base, V_bise, invV_bise, V_g, invV_g, X_fft, Y_fft, q_freq, window_2d = generate_base_grids()
a_sto, a_fese, a_mos2, a_bise, a_g = 3.905, 3.905, 3.15, 4.14, 2.46
base_grid = 40  

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def min_hex_dist(frac_pts, V_mat):
    f_floor = np.floor(frac_pts)
    d00 = np.linalg.norm((frac_pts - f_floor).dot(V_mat.T), axis=1)
    d10 = np.linalg.norm((frac_pts - (f_floor + np.array([1, 0]))).dot(V_mat.T), axis=1)
    d01 = np.linalg.norm((frac_pts - (f_floor + np.array([0, 1]))).dot(V_mat.T), axis=1)
    d11 = np.linalg.norm((frac_pts - (f_floor + np.array([1, 1]))).dot(V_mat.T), axis=1)
    return np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))

def get_square_density(a, X_grid, Y_grid):
    return 2.0 + np.cos(2 * np.pi / a * X_grid) + np.cos(2 * np.pi / a * Y_grid)

def get_hex_density(a, X_grid, Y_grid, theta_deg):
    th = np.radians(theta_deg)
    Xr = X_grid * np.cos(th) + Y_grid * np.sin(th)
    Yr = -X_grid * np.sin(th) + Y_grid * np.cos(th)
    q = 4 * np.pi / (np.sqrt(3) * a)
    return 3.0 + np.cos(q * Yr) + np.cos(q * (np.sqrt(3)/2 * Xr - 0.5 * Yr)) + np.cos(q * (-np.sqrt(3)/2 * Xr - 0.5 * Yr))

def get_square_G(a):
    q = 2 * np.pi / a
    return np.array([[q, 0], [-q, 0], [0, q], [0, -q]])

def get_hex_G(a, theta_deg):
    q = 4 * np.pi / (np.sqrt(3) * a)
    base_G = np.array([[0, q], [0, -q], [q*np.sqrt(3)/2, -q*0.5], [-q*np.sqrt(3)/2, q*0.5], [-q*np.sqrt(3)/2, -q*0.5], [q*np.sqrt(3)/2, q*0.5]])
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base_G.dot(R.T)

# ==========================================
# 3. MASTER UNIFIED PLOTTING FUNCTION
# ==========================================
def create_unified_plot(system_mode, theta_deg, zoom_factor, q_max, view_mode, show_boundaries, mid_panel_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay):
    current_fov = base_grid * zoom_factor
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5), dpi=100)
    fig.patch.set_facecolor('#1a1a1a')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
    
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base_size = max(5, 50 / (zoom_factor ** 0.5))

    N_den = 400
    x_den = np.linspace(-current_fov, current_fov, N_den)
    y_den = np.linspace(-current_fov, current_fov, N_den)
    X_den, Y_den = np.meshgrid(x_den, y_den)

    # ------------------------------------------
    # DATA ROUTING BY SYSTEM
    # ------------------------------------------
    if 'Hex-on-Square' in system_mode:
        if 'SrTiO₃' in system_mode:
            title_str, a_sub = r"MoS$_2$ on SrTiO$_3$(100)", a_sto
            label1, label2 = r"Layer 1 (SrTiO$_3$)", r"Layer 2 (MoS$_2$)"
        else:
            title_str, a_sub = r"1ML MoS$_2$ on 1ML FeSe(100)", a_fese
            label1, label2 = r"Layer 1 (FeSe)", r"Layer 2 (MoS$_2$)"
            
        decay_L = 0.25 * a_sub
        mask_sq = (np.abs(pts_sq_base[:, 0]) < current_fov) & (np.abs(pts_sq_base[:, 1]) < current_fov)
        vis_base = pts_sq_base[mask_sq]
        mask_mos2 = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_mos2].dot(R.T)
        
        nx, ny = np.round(vis_top[:, 0] / a_sub) * a_sub, np.round(vis_top[:, 1] / a_sub) * a_sub
        dist_co = np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - ny)**2)
        cx, cy = np.floor(vis_top[:, 0] / a_sub) * a_sub + a_sub/2, np.floor(vis_top[:, 1] / a_sub) * a_sub + a_sub/2
        dist_ho = np.sqrt((vis_top[:, 0] - cx)**2 + (vis_top[:, 1] - cy)**2)
        dist_br = np.minimum(np.sqrt((vis_top[:, 0] - cx)**2 + (vis_top[:, 1] - ny)**2), np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - cy)**2))
        
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/decay_L)**2), np.exp(-(dist_br/(decay_L*0.8))**2)
        
        T_total = get_square_density(a_sub, X_den, Y_den) * get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_fft = get_square_density(a_sub, X_fft, Y_fft) * get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        G1_pts, G2_pts = get_square_G(a_sub), get_hex_G(a_mos2, theta_deg)

    elif 'Bi₂Se₃' in system_mode:
        title_str, decay_L = r"1ML MoS$_2$ on 6QL Bi$_2$Se$_3$", 0.25 * a_bise
        label1, label2 = r"Layer 1 (Bi$_2$Se$_3$)", r"Layer 2 (MoS$_2$)"
        
        mask_sub = (np.abs(pts_bise_base[:, 0]) < current_fov) & (np.abs(pts_bise_base[:, 1]) < current_fov)
        vis_base = pts_bise_base[mask_sub]
        mask_top = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_top].dot(R.T)
        
        f = vis_top.dot(invV_bise.T)
        dist_co = min_hex_dist(f, V_bise)
        dist_ho = np.minimum(min_hex_dist(f - np.array([1/3, 1/3]), V_bise), min_hex_dist(f - np.array([2/3, 2/3]), V_bise))
        dist_br = np.minimum(np.minimum(min_hex_dist(f - np.array([0.5, 0.0]), V_bise), min_hex_dist(f - np.array([0.0, 0.5]), V_bise)), min_hex_dist(f - np.array([0.5, -0.5]), V_bise))
        
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/(decay_L*1.3))**2), np.exp(-(dist_br/(decay_L*0.8))**2)

        T_total = get_hex_density(a_bise, X_den, Y_den, 0.0) * get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_fft = get_hex_density(a_bise, X_fft, Y_fft, 0.0) * get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        G1_pts, G2_pts = get_hex_G(a_bise, 0.0), get_hex_G(a_mos2, theta_deg)

    else:
        title_str, decay_L = "Magic-Angle Twisted Bilayer Graphene", 0.25 * a_g
        label1, label2 = "Layer 1 (Graphene)", "Layer 2 (Rotated)"
        
        mask_sub = (np.abs(pts_grap_base[:, 0]) < current_fov) & (np.abs(pts_grap_base[:, 1]) < current_fov)
        vis_base = pts_grap_base[mask_sub]
        mask_top = (np.abs(pts_grap_base[:, 0]) < current_fov*1.5) & (np.abs(pts_grap_base[:, 1]) < current_fov*1.5)
        vis_top = pts_grap_base[mask_top].dot(R.T)
        
        f = vis_top.dot(invV_g.T)
        dist_co = min_hex_dist(f, V_g)
        dist_ho = np.minimum(min_hex_dist(f - np.array([1/3, 1/3]), V_g), min_hex_dist(f - np.array([2/3, 2/3]), V_g))
        dist_br = np.minimum(np.minimum(min_hex_dist(f - np.array([0.5, 0.0]), V_g), min_hex_dist(f - np.array([0.0, 0.5]), V_g)), min_hex_dist(f - np.array([0.5, -0.5]), V_g))
        
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/(decay_L*1.3))**2), np.exp(-(dist_br/(decay_L*0.8))**2)

        T_total = get_hex_density(a_g, X_den, Y_den, 0.0) * get_hex_density(a_g, X_den, Y_den, theta_deg)
        T_fft = get_hex_density(a_g, X_fft, Y_fft, 0.0) * get_hex_density(a_g, X_fft, Y_fft, theta_deg)
        G1_pts, G2_pts = get_hex_G(a_g, 0.0), get_hex_G(a_g, theta_deg)

    # ------------------------------------------
    # NEW: ANALYTICAL FWHM AND COVERAGE
    # ------------------------------------------
    fwhm_factor = 2 * np.sqrt(np.log(2))
    w_co = decay_L * fwhm_factor
    w_ho = decay_L * 1.3 * fwhm_factor
    w_br = decay_L * 0.8 * fwhm_factor

    strict_mask = (np.abs(vis_top[:, 0]) <= current_fov) & (np.abs(vis_top[:, 1]) <= current_fov)
    score_co_strict = score_co[strict_mask]
    score_ho_strict = score_ho[strict_mask]
    score_br_strict = score_br[strict_mask]

    n_total = max(1, len(score_co_strict))
    cov_co = np.sum(score_co_strict >= 0.5) / n_total * 100
    cov_ho = np.sum(score_ho_strict >= 0.5) / n_total * 100
    cov_br = np.sum(score_br_strict >= 0.5) / n_total * 100

    # ------------------------------------------
    # PANEL 1: REGISTRY DOMAINS & BOUNDARIES
    # ------------------------------------------
    ax1.set_xlim(-current_fov, current_fov)
    ax1.set_ylim(-current_fov, current_fov)
    ax1.set_title(f"Topology (Registry Map)\n{title_str}", color='white', fontsize=13)
    ax1.set_xlabel(r"Distance ($\AA$)", color='white')
    ax1.set_ylabel(r"Distance ($\AA$)", color='white')
    
    if view_mode == 'Raw Lattices':
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='dodgerblue', alpha=0.5)
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=2, color='crimson', alpha=0.5)
    else:
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='gray', alpha=0.3)
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=0.5, color='black', alpha=0.05)
        show_all, show_clean = (view_mode == 'Show All Registries'), (view_mode == 'Coincident + Hollow')
        
        # Plot Scatter Domains
        if show_all or show_clean or view_mode == 'Coincident Only':
            c_co = np.zeros((len(vis_top), 4)); c_co[:, 0], c_co[:, 1], c_co[:, 2], c_co[:, 3] = 1.0, 0.2, 0.3, score_co
            ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=base_size * score_co, c=c_co, edgecolors='none')
        if show_all or show_clean or view_mode == 'Hollow Only':
            c_ho = np.zeros((len(vis_top), 4)); c_ho[:, 0], c_ho[:, 1], c_ho[:, 2], c_ho[:, 3] = 0.1, 0.6, 1.0, score_ho
            ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=base_size * score_ho, c=c_ho, edgecolors='none')
        if show_all or view_mode == 'Bridge Only':
            c_br = np.zeros((len(vis_top), 4)); c_br[:, 0], c_br[:, 1], c_br[:, 2], c_br[:, 3] = 0.2, 0.8, 0.2, score_br
            ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=base_size * score_br, c=c_br, edgecolors='none')

        # NEW: Draw 50% Fall-off Contour Boundaries via Triangulation
        if show_boundaries:
            try:
                triang = mtri.Triangulation(vis_top[:, 0], vis_top[:, 1])
                if show_all or show_clean or view_mode == 'Coincident Only':
                    ax1.tricontour(triang, score_co, levels=[0.5], colors='#ff6666', linewidths=1.5, linestyles='solid')
                if show_all or show_clean or view_mode == 'Hollow Only':
                    ax1.tricontour(triang, score_ho, levels=[0.5], colors='#66b3ff', linewidths=1.5, linestyles='solid')
                if show_all or view_mode == 'Bridge Only':
                    ax1.tricontour(triang, score_br, levels=[0.5], colors='#66ff66', linewidths=1.5, linestyles='solid')
            except Exception:
                pass # Fail silently if zoom is too extreme for triangulation

    sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar1 = fig.colorbar(sm, ax=ax1, shrink=0.78, pad=0.04)
    cbar1.ax.set_visible(False)

    # ------------------------------------------
    # SHARED Z-MAP: THE PHYSICS SOLVER ENGINE
    # ------------------------------------------
    T_norm = (T_total - np.min(T_total)) / (np.max(T_total) - np.min(T_total) + 1e-10)
    Z_0 = user_zmin + T_norm * (user_zmax - user_zmin) 
    
    if relax_mode == "Rigid Lattices (No Relaxation)":
        Z_map = Z_0.copy()
        
    elif relax_mode == "Fast Proxy (Algebraic Shift)":
        compression = 0.2 * (w2 - w1)**2 
        Z_map = np.clip(Z_0 - compression, a_min=2.0, a_max=None)
        
    else:
        Z_map = Z_0.copy()
        lr = 0.05 
        A_elec = 0.5 * (w2 - w1)**2 
        
        show_bar = not hasattr(st.session_state, 'is_rendering_video') or not st.session_state.is_rendering_video
        if show_bar:
            my_bar = st.progress(0, text="Solving Euler-Lagrange PDE...")
            
        iterations = 120
        for i in range(iterations):
            laplacian = ndimage.laplace(Z_map)
            F_elastic = k_elastic * laplacian
            F_vdw = -k_vdw * (Z_map - Z_0)
            F_elec = - A_elec / (Z_map**2)
            
            Z_map = Z_map + lr * (F_elastic + F_vdw + F_elec)
            Z_map = np.clip(Z_map, 1.5, 5.0) 
            
            if show_bar and i % 10 == 0:
                my_bar.progress(int((i / iterations) * 100), text=f"Minimizing Free Energy: Iteration {i}/{iterations}")
                
        if show_bar:
            my_bar.empty()

    final_zmin, final_zmax = np.min(Z_map), np.max(Z_map)

    # ------------------------------------------
    # PANEL 2: MIDDLE PANEL ROUTING
    # ------------------------------------------
    if mid_panel_mode == 'Geometry (Density)':
        T_enhanced = T_total ** 2.5 
        vmin = np.percentile(T_enhanced, den_contrast)
        vmax = np.percentile(T_enhanced, 100 - den_contrast)
        
        im2 = ax2.imshow(T_enhanced, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Geometry (Kinematic Density)\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='white', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.78, pad=0.04)
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Relative Interfacial Density', color='white')
        
    elif mid_panel_mode == 'Local Doping (Δn)':
        Z_meters = Z_map * 1e-10
        epsilon_0, e_charge = 8.854e-12, 1.602e-19
        delta_W = w2 - w1 if (w2 - w1) != 0 else 1e-6
        
        delta_n = (epsilon_0 * delta_W) / (e_charge * Z_meters) / 1e4 
        vmin = np.percentile(delta_n, den_contrast)
        vmax = np.percentile(delta_n, 100 - den_contrast)
        
        im2 = ax2.imshow(delta_n, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Local Doping in Layer 2: $\Delta n$ (cm$^{{-2}}$)\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='#ffcc00', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.78, pad=0.04)
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Carrier Density $\Delta n$ (cm$^{-2}$)', color='white')

    elif mid_panel_mode == 'e-ph Coupling (g)':
        g_map = eph_g0 * np.exp(-(Z_map - user_zmin) / eph_decay)
        vmin = np.percentile(g_map, den_contrast)
        vmax = np.percentile(g_map, 100 - den_contrast)

        im2 = ax2.imshow(g_map, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Evanescent e-ph Coupling: $g(\mathbf{{r}})$\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='#00ffcc', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.78, pad=0.04)
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Coupling Strength $g$ (meV)', color='white')
        
    ax2.set_xlim(-current_fov, current_fov)
    ax2.set_ylim(-current_fov, current_fov)
    ax2.set_xlabel(r"Distance ($\AA$)", color='white')
    
    # ------------------------------------------
    # PANEL 3: LEED FFT
    # ------------------------------------------
    T_centered_windowed = (T_fft - np.mean(T_fft)) * window_2d
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(T_centered_windowed)))**2 + 1e-10
    q_min, q_max_fft = q_freq[0], q_freq[-1]
    
    im3 = ax3.imshow(intensity, extent=[q_min, q_max_fft, q_min, q_max_fft], origin='lower', cmap='viridis', norm=LogNorm(vmin=np.max(intensity)*1e-7, vmax=np.max(intensity)))
    ax3.scatter(G1_pts[:, 0], G1_pts[:, 1], facecolors='none', edgecolors='cyan', s=120, linewidths=1.5, marker='o', label=label1)
    ax3.scatter(G2_pts[:, 0], G2_pts[:, 1], facecolors='none', edgecolors='red', s=120, linewidths=1.5, marker='s', label=label2)
    
    g1_A = G1_pts[np.argmax(G1_pts[:, 1])]
    g1_B = G1_pts[np.argmax(G1_pts[:, 0])]
    g2_A = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_A, axis=1))]
    g2_B = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_B, axis=1))]

    for v1, v2 in [(g1_A, g2_A), (g1_B, g2_B)]:
        ax3.annotate("", xy=v1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="cyan", lw=1.5))
        ax3.annotate("", xy=v2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
        if np.linalg.norm(v2 - v1) > 1e-5: 
            ax3.annotate("", xy=v2, xytext=v1, arrowprops=dict(arrowstyle="-|>", color="yellow", lw=1.5, ls="--"))
    
    ax3.plot([], [], color='yellow', linestyle='--', lw=1.5, label=r'Moiré Vector $\mathbf{q}_M$')

    ax3.set_xlim(-q_max, q_max)
    ax3.set_ylim(-q_max, q_max)
    ax3.set_title(f"Scattering (Simulated LEED)\nTwist: {theta_deg}" + r"$^\circ$", color='white', fontsize=13)
    ax3.set_xlabel(r"$q_x$ ($\AA^{-1}$)", color='white')
    
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.78, pad=0.04)
    cbar3.ax.tick_params(colors='white')
    cbar3.set_label('Scattering Intensity (a.u.)', color='white')
    
    if view_mode != 'Raw Lattices':
        legend_elements = []
        if show_all or show_clean or view_mode == 'Coincident Only':
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1.0, 0.2, 0.3), markersize=9, label=f'Coincident (W: {w_co:.1f}Å, Cov: {cov_co:.1f}%)'))
        if show_all or show_clean or view_mode == 'Hollow Only':
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.1, 0.6, 1.0), markersize=9, label=f'Hollow (W: {w_ho:.1f}Å, Cov: {cov_ho:.1f}%)'))
        if show_all or view_mode == 'Bridge Only':
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.2, 0.8, 0.2), markersize=9, label=f'Bridge (W: {w_br:.1f}Å, Cov: {cov_br:.1f}%)'))
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.8)
        
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.8)
    plt.tight_layout()
    return fig

# ==========================================
# 4. STREAMLIT UI CONTROLS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    system_mode = st.selectbox("System:", ['MoS₂/SrTiO₃ (Hex-on-Square)', 'MoS₂/FeSe(100) (Hex-on-Square)', 'MoS₂/Bi₂Se₃ (Hex-on-Hex)', 'MoS₂/Graphene (Hex-on-Hex)', 'MATBG (Hex-on-Hex)'])
    view_mode = st.selectbox("Topology View:", ['Show All Registries', 'Coincident + Hollow', 'Coincident Only', 'Hollow Only', 'Bridge Only', 'Raw Lattices'])
    show_boundaries = st.checkbox("Show Domain Boundaries (50% Fall-off)", value=False)

with col2:
    mid_panel_mode = st.radio("Middle Panel Metric:", ["Geometry (Density)", "Local Doping (Δn)", "e-ph Coupling (g)"], horizontal=True)
    den_cmap = st.selectbox("Panel 2 Color:", ['magma', 'viridis', 'plasma', 'cividis', 'gray', 'bone', 'coolwarm'])

with col3:
    theta_deg = st.slider("Twist Angle (deg):", 0.0, 45.0, 0.0, 0.1)
    zoom_factor = st.slider("FOV Zoom (x):", 1.0, 5.0, 1.0, 0.5)
    q_max = st.slider("q-space Zoom (Å⁻¹):", 1.0, 8.0, 4.0, 0.5)
    den_contrast = st.slider("Contrast Clip (%):", 0.0, 20.0, 0.0, 1.0)

# Render the Plot
fig = create_unified_plot(system_mode, theta_deg, zoom_factor, q_max, view_mode, show_boundaries, mid_panel_mode, den_cmap, den_contrast, "Rigid Lattices (No Relaxation)", 4.2, 4.5, 3.1, 3.6, 0.0, 0.0, 80.0, 0.5) 

# --- EXPANDER FOR ADVANCED PHYSICS PARAMETERS ---
with st.expander("⚙️ Advanced Physics Parameters (Interfacial Mechanics & e-ph Coupling)", expanded=True):
    
    st.markdown("**1. Interfacial Mechanics & Doping Model**")
    
    relax_mode = st.selectbox("Mechanical Relaxation Model:", [
        "Rigid Lattices (No Relaxation)", 
        "Fast Proxy (Algebraic Shift)", 
        "Continuum Mechanics (PDE Solver)"
    ])
    
    intrinsic_z = {
        'MoS₂/SrTiO₃ (Hex-on-Square)': (3.1, 3.6),
        'MoS₂/FeSe(100) (Hex-on-Square)': (3.2, 3.6),
        'MoS₂/Bi₂Se₃ (Hex-on-Hex)': (3.2, 3.6),
        'MoS₂/Graphene (Hex-on-Hex)': (3.3, 3.6),
        'MATBG (Hex-on-Hex)': (3.35, 3.6)
    }
    base_zmin, base_zmax = intrinsic_z[system_mode]

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        w1 = st.number_input("Layer 1 Work Function (eV)", value=4.2, step=0.1)
    with pcol2:
        w2 = st.number_input("Layer 2 Work Function (eV)", value=4.5, step=0.1)

    with pcol3:
        user_zmin = st.number_input("Base Unrelaxed Min Gap (Å)", value=float(base_zmin), step=0.1)
    with pcol4:
        user_zmax = st.number_input("Base Unrelaxed Max Gap (Å)", value=float(base_zmax), step=0.1)
        
    k_elastic, k_vdw = 0.0, 0.0
    if relax_mode == "Continuum Mechanics (PDE Solver)":
        st.markdown("*Continuum Tuning Parameters (Determines spatial smoothness vs structural pinning):*")
        scol1, scol2 = st.columns(2)
        with scol1:
            k_elastic = st.slider("Elastic Bending Rigidity (κ)", 0.0, 2.0, 0.5, 0.1)
        with scol2:
            k_vdw = st.slider("vdW Spring Stiffness ($k_{vdW}$)", 0.1, 5.0, 1.0, 0.1)

    st.markdown("---")
    st.markdown("**2. Local Electron-Phonon Coupling Model**")
    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    with ecol1:
        eph_g0 = st.number_input("Base Coupling at min gap (meV)", value=80.0, step=5.0)
    with ecol2:
        eph_decay = st.number_input("Evanescent Decay Length $\lambda$ (Å)", value=0.5, step=0.1)

# Re-render plot with proper params if they were modified in the expander
fig = create_unified_plot(system_mode, theta_deg, zoom_factor, q_max, view_mode, show_boundaries, mid_panel_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay)
st.pyplot(fig)


# --- VIDEO GENERATOR (CINEMATIC TOOLS) ---
st.markdown("---")
st.markdown("### 🎥 Cinematic Tools")

if st.button("Generate Twist Angle Scan Video (0° to 45°)"):
    st.session_state.is_rendering_video = True
    vid_progress = st.progress(0, text="Rendering frame 1 of 46...")
    
    frames = []
    for ang in range(46):
        fig_frame = create_unified_plot(system_mode, float(ang), zoom_factor, q_max, view_mode, show_boundaries, mid_panel_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay)
        
        # FIXED: Streamlit-Cloud safe RGB buffer extraction
        fig_frame.canvas.draw()
        img_rgba = np.asarray(fig_frame.canvas.buffer_rgba())
        img = img_rgba[:, :, :3] # Slice out the Alpha channel to get pure RGB
        frames.append(img)
        
        plt.close(fig_frame) 
        vid_progress.progress(int((ang + 1) / 46 * 100), text=f"Rendering frame {ang+1} of 46...")
    
    vid_progress.progress(100, text="Encoding MP4 Video...")
    imageio.mimsave("moire_twist_scan.mp4", frames, fps=8, macro_block_size=None)
    vid_progress.empty()
    st.session_state.is_rendering_video = False
    
    st.success("Video Generated Successfully!")
    st.video("moire_twist_scan.mp4", autoplay=True, loop=True)
    
    with open("moire_twist_scan.mp4", "rb") as file:
        st.download_button(
            label="💾 Save Video to Computer",
            data=file,
            file_name=f"twist_scan_{system_mode[:4]}.mp4",
            mime="video/mp4"
        )
