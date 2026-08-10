import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.tri as mtri
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.ndimage as ndimage
import imageio

st.set_page_config(page_title="UHV-bonded Heterostructure Physics Dashboard", layout="wide")

st.markdown("# UHV-bonded Heterostructure Physics Dashboard <span style='font-size: 20px; font-weight: normal; color: #888888;'>v. May 16, 2026</span>", unsafe_allow_html=True)
st.markdown("Explore the topology, geometry, scattering, local doping level and many-body interactions of 2D UHV-bonded heterostructures.")

# --- PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "physics2026": 
            st.session_state["password_correct"] = True
            del st.session_state["password"]  
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Password to access the UHV Dashboard:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Password to access the UHV Dashboard:", type="password", on_change=password_entered, key="password")
        st.error("Incorrect Password")
        return False
    return True

if not check_password():
    st.stop()

# ==========================================
# 1. PARAMETERS & PRE-COMPUTATION
# ==========================================
@st.cache_resource
def generate_base_grids():
    a_sto, a_fese, a_mos2, a_bise, a_g = 3.905, 3.905, 3.15, 4.14, 2.46          
    a_cux, a_cuy = 3.61, 2.553  
    base_grid = 40  
    max_grid = base_grid * 6 

    x_sq = np.arange(-max_grid, max_grid, a_sto)
    y_sq = np.arange(-max_grid, max_grid, a_sto)
    xx_sq, yy_sq = np.meshgrid(x_sq, y_sq)
    pts_sq_base = np.vstack([xx_sq.ravel(), yy_sq.ravel()]).T

    x_rect = np.arange(-max_grid, max_grid, a_cux)
    y_rect = np.arange(-max_grid, max_grid, a_cuy)
    xx_rect, yy_rect = np.meshgrid(x_rect, y_rect)
    pts_rect_base = np.vstack([xx_rect.ravel(), yy_rect.ravel()]).T

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

    N_fft = 512  
    L_fft = 400.0  
    x_fft = np.linspace(-L_fft/2, L_fft/2, N_fft)
    y_fft = np.linspace(-L_fft/2, L_fft/2, N_fft)
    X_fft, Y_fft = np.meshgrid(x_fft, y_fft)
    q_freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=(L_fft/N_fft))) * 2 * np.pi
    window_1d = np.hanning(N_fft)
    window_2d = window_1d[:, np.newaxis] * window_1d[np.newaxis, :]
    
    return pts_sq_base, pts_rect_base, pts_mos2_base, pts_bise_base, pts_grap_base, V_bise, invV_bise, V_g, invV_g, X_fft, Y_fft, q_freq, window_2d

cached_data = generate_base_grids()
a_sto, a_fese, a_mos2, a_bise, a_g = 3.905, 3.905, 3.15, 4.14, 2.46
a_cux, a_cuy = 3.61, 2.553
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

def calculate_hex_registry_distances(vis_top, V_sub, invV_sub):
    f = vis_top.dot(invV_sub.T)
    dist_co = min_hex_dist(f, V_sub)
    dist_ho = np.minimum(min_hex_dist(f - np.array([1/3, 1/3]), V_sub), min_hex_dist(f - np.array([2/3, 2/3]), V_sub))
    dist_br = np.minimum(np.minimum(min_hex_dist(f - np.array([0.5, 0.0]), V_sub), min_hex_dist(f - np.array([0.0, 0.5]), V_sub)), min_hex_dist(f - np.array([0.5, -0.5]), V_sub))
    return dist_co, dist_ho, dist_br

def get_square_density(a, X_grid, Y_grid):
    return 2.0 + np.cos(2 * np.pi / a * X_grid) + np.cos(2 * np.pi / a * Y_grid)

def get_rect_density(ax, ay, X_grid, Y_grid):
    return 2.0 + np.cos(2 * np.pi / ax * X_grid) + np.cos(2 * np.pi / ay * Y_grid)

def get_hex_density(a, X_grid, Y_grid, theta_deg):
    th = np.radians(theta_deg)
    Xr = X_grid * np.cos(th) + Y_grid * np.sin(th)
    Yr = -X_grid * np.sin(th) + Y_grid * np.cos(th)
    q = 4 * np.pi / (np.sqrt(3) * a)
    return 3.0 + np.cos(q * Yr) + np.cos(q * (np.sqrt(3)/2 * Xr - 0.5 * Yr)) + np.cos(q * (-np.sqrt(3)/2 * Xr - 0.5 * Yr))

def get_reflected_coords(X, Y, phi_deg):
    phi = np.radians(phi_deg)
    c, s = np.cos(2*phi), np.sin(2*phi)
    Xr = X * c + Y * s
    Yr = X * s - Y * c
    return Xr, Yr

def get_square_G(a):
    q = 2 * np.pi / a
    return np.array([[q, 0], [-q, 0], [0, q], [0, -q]])

def get_rect_G(ax, ay):
    qx = 2 * np.pi / ax
    qy = 2 * np.pi / ay
    return np.array([[qx, 0], [-qx, 0], [0, qy], [0, -qy]])

def get_hex_G(a, theta_deg):
    q = 4 * np.pi / (np.sqrt(3) * a)
    base_G = np.array([[0, q], [0, -q], [q*np.sqrt(3)/2, -q*0.5], [-q*np.sqrt(3)/2, q*0.5], [-q*np.sqrt(3)/2, -q*0.5], [q*np.sqrt(3)/2, q*0.5]])
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base_G.dot(R.T)

def get_square_bz(a, theta_deg=0.0):
    q = 2 * np.pi / a
    base_bz = np.array([[q/2, q/2], [-q/2, q/2], [-q/2, -q/2], [q/2, -q/2], [q/2, q/2]])
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base_bz.dot(R.T)

def get_rect_bz(ax, ay, theta_deg=0.0):
    qx = 2 * np.pi / ax
    qy = 2 * np.pi / ay
    base_bz = np.array([[qx/2, qy/2], [-qx/2, qy/2], [-qx/2, -qy/2], [qx/2, -qy/2], [qx/2, qy/2]])
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base_bz.dot(R.T)

def get_hex_bz(a, theta_deg=0.0):
    q = 4 * np.pi / (np.sqrt(3) * a)
    R_bz = q / np.sqrt(3)
    angles = np.radians([0, 60, 120, 180, 240, 300, 360])
    base_bz = np.array([[R_bz * np.cos(ang), R_bz * np.sin(ang)] for ang in angles])
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return base_bz.dot(R.T)

def get_glassy_field_continuous(X, Y, coupling):
    if coupling == 0:
        return np.zeros_like(X), np.zeros_like(Y)
    
    rng = np.random.default_rng(42)
    Ux = np.zeros_like(X)
    Uy = np.zeros_like(Y)
    
    N_waves = 40 
    k0 = 2 * np.pi / 120.0 
    
    ks = rng.normal(k0, k0*0.2, N_waves)
    thetas = rng.uniform(0, 2*np.pi, N_waves)
    phases = rng.uniform(0, 2*np.pi, N_waves)
    
    for i in range(N_waves):
        kx = ks[i] * np.cos(thetas[i])
        ky = ks[i] * np.sin(thetas[i])
        wave = np.sin(X*kx + Y*ky + phases[i])
        
        Ux += -ky * wave
        Uy +=  kx * wave
        
    norm = np.max(np.sqrt(Ux**2 + Uy**2)) + 1e-10
    amp = coupling * 1.5 
    return (Ux / norm) * amp, (Uy / norm) * amp

# ==========================================
# 3. MASTER UNIFIED PLOTTING FUNCTION
# ==========================================
def create_unified_plot(fig, cached_data, system_mode, theta_deg, zoom_factor, q_max, k_max, view_mode, boundary_mode, mid_panel_mode, panel3_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay, interfacial_state, strain_coupling, is_video_frame=False):
    pts_sq_base, pts_rect_base, pts_mos2_base, pts_bise_base, pts_grap_base, V_bise, invV_bise, V_g, invV_g, X_fft, Y_fft, q_freq, window_2d = cached_data

    show_fs_panel = ('FeSe' in system_mode)

    if fig is None:
        if show_fs_panel:
            fig = Figure(figsize=(24, 18), dpi=100)
        else:
            fig = Figure(figsize=(21, 6.5), dpi=100)
        FigureCanvasAgg(fig) 
    else:
        fig.clf()
        
    fig.patch.set_facecolor('#1a1a1a')
    
    if is_video_frame:
        fig.text(0.02, 0.96, f"Twist Angle: {theta_deg:.1f}°", color='#ffcc00', fontsize=14, fontweight='bold', va='top', ha='left')
    
    if show_fs_panel:
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 0.04, 1, 0.14, 1], height_ratios=[1, 2.0], wspace=0.0, hspace=0.35, left=0.08, right=0.88, bottom=0.05, top=0.95)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[0, 4])
        ax4 = fig.add_subplot(gs[1, :]) 
        axes = [ax1, ax2, ax3, ax4]
    else:
        gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.04, 1, 0.14, 1], wspace=0.0, left=0.08, right=0.88, bottom=0.1, top=0.88)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[4])
        axes = [ax1, ax2, ax3]
        ax4 = None
    
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
    
    current_fov = base_grid * zoom_factor
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    base_size = max(5, 50 / (zoom_factor ** 0.5))

    N_den = 512 
    x_den = np.linspace(-current_fov, current_fov, N_den)
    y_den = np.linspace(-current_fov, current_fov, N_den)
    X_den, Y_den = np.meshgrid(x_den, y_den)

    # ------------------------------------------
    # VARIABLE STANDARDIZATION & DATA ROUTING
    # ------------------------------------------
    fft_render_mode = "rigid"
    T_sub_fft = None
    T_top_fft = None
    T_fft_engine = None
    layer2_Cq = 0.01  
    
    if 'Hex-on-Square' in system_mode:
        if 'SrTiO₃' in system_mode:
            title_str, a_sub = r"MoS$_2$ on SrTiO$_3$", a_sto
            label1, label2 = r"Layer 1 (SrTiO$_3$)", r"Layer 2 (MoS$_2$)"
        else:
            title_str, a_sub = r"1ML MoS$_2$ on 1ML FeSe", a_fese
            label1, label2 = r"Layer 1 (FeSe)", r"Layer 2 (MoS$_2$)"
            
        decay_L = 0.25 * a_sub
        mask_sq = (np.abs(pts_sq_base[:, 0]) < current_fov) & (np.abs(pts_sq_base[:, 1]) < current_fov)
        vis_base = pts_sq_base[mask_sq]
        mask_mos2 = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_mos2].dot(R.T)
        
        T_sub_den = get_square_density(a_sub, X_den, Y_den)
        T_sub_fft = get_square_density(a_sub, X_fft, Y_fft)
        T_top_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        
        if "Misfit Dislocation Glass" in interfacial_state and strain_coupling > 0:
            Ux_den, Uy_den = get_glassy_field_continuous(X_den, Y_den, strain_coupling)
            Ux_pts, Uy_pts = get_glassy_field_continuous(vis_top[:,0], vis_top[:,1], strain_coupling)
            
            T_top_den = get_hex_density(a_mos2, X_den - Ux_den, Y_den - Uy_den, theta_deg)
            T_total = T_sub_den * T_top_den
            
            vis_top[:,0] -= Ux_pts
            vis_top[:,1] -= Uy_pts
            
            T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            T_fft_engine = T_sub_fft * T_top_fft 
            fft_render_mode = "glass"

        elif "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            weight = strain_coupling * 0.4
            
            T_base_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
            Xr_m0, Yr_m0 = get_reflected_coords(X_den, Y_den, 0.0)
            Xr_m45, Yr_m45 = get_reflected_coords(X_den, Y_den, 45.0)
            T_m0_den = get_hex_density(a_mos2, Xr_m0, Yr_m0, theta_deg)
            T_m45_den = get_hex_density(a_mos2, Xr_m45, Yr_m45, theta_deg)
            T_top_den_qc = T_base_den + weight * T_m0_den + weight * T_m45_den
            
            Xr_f1, Yr_f1 = get_reflected_coords(X_den, Y_den, theta_deg)
            Xr_f2, Yr_f2 = get_reflected_coords(X_den, Y_den, theta_deg + 30.0)
            T_fese_m1 = get_square_density(a_sub, Xr_f1, Yr_f1)
            T_fese_m2 = get_square_density(a_sub, Xr_f2, Yr_f2)
            T_sub_den_qc = T_sub_den + weight * T_fese_m1 + weight * T_fese_m2
            
            T_total = T_sub_den_qc * T_top_den_qc
            
            T_base_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            Xf_m0, Yf_m0 = get_reflected_coords(X_fft, Y_fft, 0.0)
            Xf_m45, Yf_m45 = get_reflected_coords(X_fft, Y_fft, 45.0)
            T_m0_fft = get_hex_density(a_mos2, Xf_m0, Yf_m0, theta_deg)
            T_m45_fft = get_hex_density(a_mos2, Xf_m45, Yf_m45, theta_deg)
            T_top_fft = T_base_fft + (weight * 6.0 * T_m0_fft) + (weight * 6.0 * T_m45_fft)
            
            Xf_f1, Yf_f1 = get_reflected_coords(X_fft, Y_fft, theta_deg)
            Xf_f2, Yf_f2 = get_reflected_coords(X_fft, Y_fft, theta_deg + 30.0)
            T_fese_m1_fft = get_square_density(a_sub, Xf_f1, Yf_f1)
            T_fese_m2_fft = get_square_density(a_sub, Xf_f2, Yf_f2)
            T_sub_fft = T_sub_fft + (weight * 6.0 * T_fese_m1_fft) + (weight * 6.0 * T_fese_m2_fft)
            
            T_fft_engine = T_sub_fft * T_top_fft
            fft_render_mode = "quasicrystal"

        else:
            T_total = T_sub_den * get_hex_density(a_mos2, X_den, Y_den, theta_deg)
            T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            T_fft_engine = T_sub_fft + T_top_fft
            fft_render_mode = "rigid"
        
        nx, ny = np.round(vis_top[:, 0] / a_sub) * a_sub, np.round(vis_top[:, 1] / a_sub) * a_sub
        dist_co = np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - ny)**2)
        cx, cy = np.floor(vis_top[:, 0] / a_sub) * a_sub + a_sub/2, np.floor(vis_top[:, 1] / a_sub) * a_sub + a_sub/2
        dist_ho = np.sqrt((vis_top[:, 0] - cx)**2 + (vis_top[:, 1] - cy)**2)
        dist_br = np.minimum(np.sqrt((vis_top[:, 0] - cx)**2 + (vis_top[:, 1] - ny)**2), np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - cy)**2))
        
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/decay_L)**2), np.exp(-(dist_br/(decay_L*0.8))**2)
        
        G1_pts, G2_pts = get_square_G(a_sub), get_hex_G(a_mos2, theta_deg)
        BZ1_pts, BZ2_pts = get_square_bz(a_sub, 0.0), get_hex_bz(a_mos2, theta_deg)

    elif 'Hex-on-Rect' in system_mode:
        title_str, a_sub_x, a_sub_y = r"MoS$_2$ on Cu(110)", a_cux, a_cuy
        label1, label2 = r"Layer 1 (Cu)", r"Layer 2 (MoS$_2$)"
        decay_L = 0.25 * a_sub_x
        
        mask_rect = (np.abs(pts_rect_base[:, 0]) < current_fov) & (np.abs(pts_rect_base[:, 1]) < current_fov)
        vis_base = pts_rect_base[mask_rect]
        mask_mos2 = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_mos2].dot(R.T)
        
        T_sub_den = get_rect_density(a_sub_x, a_sub_y, X_den, Y_den)
        T_sub_fft = get_rect_density(a_sub_x, a_sub_y, X_fft, Y_fft)
        T_top_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        
        if "Misfit Dislocation Glass" in interfacial_state and strain_coupling > 0:
            Ux_den, Uy_den = get_glassy_field_continuous(X_den, Y_den, strain_coupling)
            Ux_pts, Uy_pts = get_glassy_field_continuous(vis_top[:,0], vis_top[:,1], strain_coupling)
            
            T_top_den = get_hex_density(a_mos2, X_den - Ux_den, Y_den - Uy_den, theta_deg)
            T_total = T_sub_den * T_top_den
            
            vis_top[:,0] -= Ux_pts
            vis_top[:,1] -= Uy_pts
            
            T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            T_fft_engine = T_sub_fft * T_top_fft 
            fft_render_mode = "glass"

        elif "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            weight = strain_coupling * 0.4
            
            T_base_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
            Xr_m0, Yr_m0 = get_reflected_coords(X_den, Y_den, 0.0)
            Xr_m90, Yr_m90 = get_reflected_coords(X_den, Y_den, 90.0)
            T_m0_den = get_hex_density(a_mos2, Xr_m0, Yr_m0, theta_deg)
            T_m90_den = get_hex_density(a_mos2, Xr_m90, Yr_m90, theta_deg)
            T_top_den_qc = T_base_den + weight * T_m0_den + weight * T_m90_den
            
            Xr_f1, Yr_f1 = get_reflected_coords(X_den, Y_den, theta_deg)
            Xr_f2, Yr_f2 = get_reflected_coords(X_den, Y_den, theta_deg + 30.0)
            T_cu_m1 = get_rect_density(a_sub_x, a_sub_y, Xr_f1, Yr_f1)
            T_cu_m2 = get_rect_density(a_sub_x, a_sub_y, Xr_f2, Yr_f2)
            T_sub_den_qc = T_sub_den + weight * T_cu_m1 + weight * T_cu_m2
            
            T_total = T_sub_den_qc * T_top_den_qc
            
            T_base_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            Xf_m0, Yf_m0 = get_reflected_coords(X_fft, Y_fft, 0.0)
            Xf_m90, Yf_m90 = get_reflected_coords(X_fft, Y_fft, 90.0)
            T_m0_fft = get_hex_density(a_mos2, Xf_m0, Yf_m0, theta_deg)
            T_m90_fft = get_hex_density(a_mos2, Xf_m90, Yf_m90, theta_deg)
            T_top_fft = T_base_fft + (weight * 6.0 * T_m0_fft) + (weight * 6.0 * T_m90_fft)
            
            Xf_f1, Yf_f1 = get_reflected_coords(X_fft, Y_fft, theta_deg)
            Xf_f2, Yf_f2 = get_reflected_coords(X_fft, Y_fft, theta_deg + 30.0)
            T_cu_m1_fft = get_rect_density(a_sub_x, a_sub_y, Xf_f1, Yf_f1)
            T_cu_m2_fft = get_rect_density(a_sub_x, a_sub_y, Xf_f2, Yf_f2)
            T_sub_fft = T_sub_fft + (weight * 6.0 * T_cu_m1_fft) + (weight * 6.0 * T_cu_m2_fft)
            
            T_fft_engine = T_sub_fft * T_top_fft
            fft_render_mode = "quasicrystal"

        else:
            T_total = T_sub_den * get_hex_density(a_mos2, X_den, Y_den, theta_deg)
            T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
            T_fft_engine = T_sub_fft + T_top_fft
            fft_render_mode = "rigid"
        
        nx = np.round(vis_top[:, 0] / a_sub_x) * a_sub_x
        ny = np.round(vis_top[:, 1] / a_sub_y) * a_sub_y
        dist_co = np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - ny)**2)
        
        hx = np.floor(vis_top[:, 0] / a_sub_x) * a_sub_x + a_sub_x/2
        hy = np.floor(vis_top[:, 1] / a_sub_y) * a_sub_y + a_sub_y/2
        dist_ho = np.sqrt((vis_top[:, 0] - hx)**2 + (vis_top[:, 1] - hy)**2)
        
        dist_br = np.minimum(
            np.sqrt((vis_top[:, 0] - hx)**2 + (vis_top[:, 1] - ny)**2),
            np.sqrt((vis_top[:, 0] - nx)**2 + (vis_top[:, 1] - hy)**2)
        )
        
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/decay_L)**2), np.exp(-(dist_br/(decay_L*0.8))**2)
        
        G1_pts, G2_pts = get_rect_G(a_sub_x, a_sub_y), get_hex_G(a_mos2, theta_deg)
        BZ1_pts, BZ2_pts = get_rect_bz(a_sub_x, a_sub_y, 0.0), get_hex_bz(a_mos2, theta_deg)

    elif 'Bi₂Se₃' in system_mode:
        title_str, decay_L = r"1ML MoS$_2$ on 6QL Bi$_2$Se$_3$", 0.25 * a_bise
        label1, label2 = r"Layer 1 (Bi$_2$Se$_3$)", r"Layer 2 (MoS$_2$)"
        V_sub, invV_sub = V_bise, invV_bise
        
        mask_sub = (np.abs(pts_bise_base[:, 0]) < current_fov) & (np.abs(pts_bise_base[:, 1]) < current_fov)
        vis_base = pts_bise_base[mask_sub]
        mask_top = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_top].dot(R.T)
        
        T_sub_den = get_hex_density(a_bise, X_den, Y_den, 0.0)
        T_sub_fft = get_hex_density(a_bise, X_fft, Y_fft, 0.0)
        T_top_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        
        T_total = T_sub_den * T_top_den
        T_fft_engine = T_sub_fft + T_top_fft
        
        dist_co, dist_ho, dist_br = calculate_hex_registry_distances(vis_top, V_sub, invV_sub)
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/(decay_L*1.3))**2), np.exp(-(dist_br/(decay_L*0.8))**2)

        G1_pts, G2_pts = get_hex_G(a_bise, 0.0), get_hex_G(a_mos2, theta_deg)
        BZ1_pts, BZ2_pts = get_hex_bz(a_bise, 0.0), get_hex_bz(a_mos2, theta_deg)

    elif 'Graphene' in system_mode:
        title_str, decay_L = r"1ML MoS$_2$ on Graphene", 0.25 * a_g 
        label1, label2 = r"Layer 1 (Graphene)", r"Layer 2 (MoS$_2$)"
        V_sub, invV_sub = V_g, invV_g
        
        mask_sub = (np.abs(pts_grap_base[:, 0]) < current_fov) & (np.abs(pts_grap_base[:, 1]) < current_fov)
        vis_base = pts_grap_base[mask_sub]
        mask_top = (np.abs(pts_mos2_base[:, 0]) < current_fov*1.5) & (np.abs(pts_mos2_base[:, 1]) < current_fov*1.5)
        vis_top = pts_mos2_base[mask_top].dot(R.T)
        
        T_sub_den = get_hex_density(a_g, X_den, Y_den, 0.0)
        T_sub_fft = get_hex_density(a_g, X_fft, Y_fft, 0.0)
        T_top_den = get_hex_density(a_mos2, X_den, Y_den, theta_deg)
        T_top_fft = get_hex_density(a_mos2, X_fft, Y_fft, theta_deg)
        
        T_total = T_sub_den * T_top_den
        T_fft_engine = T_sub_fft + T_top_fft
        
        dist_co, dist_ho, dist_br = calculate_hex_registry_distances(vis_top, V_sub, invV_sub)
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/(decay_L*1.3))**2), np.exp(-(dist_br/(decay_L*0.8))**2)

        G1_pts, G2_pts = get_hex_G(a_g, 0.0), get_hex_G(a_mos2, theta_deg)
        BZ1_pts, BZ2_pts = get_hex_bz(a_g, 0.0), get_hex_bz(a_mos2, theta_deg)

    else: 
        title_str, decay_L = "Magic-Angle Twisted Bilayer Graphene", 0.25 * a_g
        label1, label2 = "Layer 1 (Graphene)", "Layer 2 (Rotated)"
        layer2_Cq = 0.002 
        V_sub, invV_sub = V_g, invV_g
        
        mask_sub = (np.abs(pts_grap_base[:, 0]) < current_fov) & (np.abs(pts_grap_base[:, 1]) < current_fov)
        vis_base = pts_grap_base[mask_sub]
        mask_top = (np.abs(pts_grap_base[:, 0]) < current_fov*1.5) & (np.abs(pts_grap_base[:, 1]) < current_fov*1.5)
        vis_top = pts_grap_base[mask_top].dot(R.T)
        
        T_sub_den = get_hex_density(a_g, X_den, Y_den, 0.0)
        T_sub_fft = get_hex_density(a_g, X_fft, Y_fft, 0.0)
        T_top_den = get_hex_density(a_g, X_den, Y_den, theta_deg)
        T_top_fft = get_hex_density(a_g, X_fft, Y_fft, theta_deg)
        
        T_total = T_sub_den * T_top_den
        T_fft_engine = T_sub_fft + T_top_fft
        
        dist_co, dist_ho, dist_br = calculate_hex_registry_distances(vis_top, V_sub, invV_sub)
        score_co, score_ho, score_br = np.exp(-(dist_co/decay_L)**2), np.exp(-(dist_ho/(decay_L*1.3))**2), np.exp(-(dist_br/(decay_L*0.8))**2)

        G1_pts, G2_pts = get_hex_G(a_g, 0.0), get_hex_G(a_g, theta_deg)
        BZ1_pts, BZ2_pts = get_hex_bz(a_g, 0.0), get_hex_bz(a_g, theta_deg)

    # ------------------------------------------
    # EXACT CONTINUOUS ATOMIC STM TOPOGRAPHY (BENCHMARK)
    # ------------------------------------------
    sigma_atom = 1.0
    xi_stack = 0.6
    A_M = 1.1

    h_x = 0.5 + 0.5 * np.cos(np.pi * vis_top[:, 0] / current_fov)
    h_y = 0.5 + 0.5 * np.cos(np.pi * vis_top[:, 1] / current_fov)
    window_pts = h_x * h_y

    A_i = (1.0 + A_M * np.exp(-(dist_co**2) / (2 * xi_stack**2))) * window_pts

    r_cut = 3.5 * sigma_atom
    dx = (2 * current_fov) / N_den

    Z_stm_exact = np.zeros((N_den, N_den))

    for i in range(len(vis_top)):
        ax_pos, ay_pos = vis_top[i, 0], vis_top[i, 1]
        
        ix_min = max(0, int((ax_pos - r_cut + current_fov) / dx))
        ix_max = min(N_den, int((ax_pos + r_cut + current_fov) / dx) + 1)
        iy_min = max(0, int((ay_pos - r_cut + current_fov) / dx))
        iy_max = min(N_den, int((ay_pos + r_cut + current_fov) / dx) + 1)
        
        if ix_min >= ix_max or iy_min >= iy_max:
            continue
            
        sub_x = x_den[ix_min:ix_max]
        sub_y = y_den[iy_min:iy_max]
        sub_X, sub_Y = np.meshgrid(sub_x, sub_y)
        
        dist2 = (sub_X - ax_pos)**2 + (sub_Y - ay_pos)**2
        Z_stm_exact[iy_min:iy_max, ix_min:ix_max] += A_i[i] * np.exp(-dist2 / (2 * sigma_atom**2))

    # ------------------------------------------
    # SHARED MOIRÉ EXTRACTION (RECIPROCAL -> REAL SPACE)
    # ------------------------------------------
    g1_A, g1_B, g2_A, g2_B = None, None, None, None
    q1, q2, L1, L2 = None, None, None, None
    
    if len(G1_pts) > 0 and len(G2_pts) > 0:
        g1_A = G1_pts[np.argmax(G1_pts[:, 1])]
        g1_B = G1_pts[np.argmax(G1_pts[:, 0])]
        try:
            g2_A = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_A, axis=1))]
            g2_B = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_B, axis=1))]
            
            q1 = g1_A - g2_A
            q2 = g1_B - g2_B
            Q = np.column_stack([q1, q2])
            
            if np.abs(np.linalg.det(Q)) > 1e-5:
                L_mat = 2 * np.pi * np.linalg.inv(Q.T)
                L1 = L_mat[:, 0]
                L2 = L_mat[:, 1]
        except Exception:
            pass

    # ------------------------------------------
    # PANEL 1: REGISTRY DOMAINS
    # ------------------------------------------
    ax1.set_xlim(-current_fov, current_fov)
    ax1.set_ylim(-current_fov, current_fov)
    
    panel1_subtitle = f"{title_str} | FOV: {zoom_factor}x"
    if L1 is not None and L2 is not None:
        norm_l1 = np.linalg.norm(L1)
        norm_l2 = np.linalg.norm(L2)
        if norm_l2 > 1e-5:
            panel1_subtitle += f" | $L_{{M1}}/L_{{M2}}$: {(norm_l1/norm_l2):.2f}"
            
    ax1.set_title(f"Topology (Registry Map)\n{panel1_subtitle}", color='white', fontsize=13)
    ax1.set_xlabel(r"Distance ($\AA$)", color='white')
    ax1.set_ylabel(r"Distance ($\AA$)", color='white')
    
    show_all = (view_mode == 'Show All Registries')
    show_co_dom = show_all or view_mode in ['Coincident + Hollow', 'Coincident Only']
    show_ho_dom = show_all or view_mode in ['Coincident + Hollow', 'Hollow Only']
    show_br_dom = show_all or view_mode == 'Bridge Only'

    if view_mode == 'Raw Lattices':
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='dodgerblue', alpha=0.5)
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=2, color='crimson', alpha=0.5)
    else:
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='gray', alpha=0.3, marker=',')
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=0.5, color='black', alpha=0.05, marker=',')
        
        if show_co_dom:
            mask_co = score_co > 0.05
            if np.any(mask_co):
                c_co = np.zeros((np.sum(mask_co), 4))
                c_co[:, 0], c_co[:, 1], c_co[:, 2], c_co[:, 3] = 1.0, 0.2, 0.3, score_co[mask_co]
                ax1.scatter(vis_top[mask_co, 0], vis_top[mask_co, 1], s=base_size * score_co[mask_co], c=c_co, edgecolors='none')
                
        if show_ho_dom:
            mask_ho = score_ho > 0.05
            if np.any(mask_ho):
                c_ho = np.zeros((np.sum(mask_ho), 4))
                c_ho[:, 0], c_ho[:, 1], c_ho[:, 2], c_ho[:, 3] = 0.1, 0.6, 1.0, score_ho[mask_ho]
                ax1.scatter(vis_top[mask_ho, 0], vis_top[mask_ho, 1], s=base_size * score_ho[mask_ho], c=c_ho, edgecolors='none')
                
        if show_br_dom:
            mask_br = score_br > 0.05
            if np.any(mask_br):
                c_br = np.zeros((np.sum(mask_br), 4))
                c_br[:, 0], c_br[:, 1], c_br[:, 2], c_br[:, 3] = 0.2, 0.8, 0.2, score_br[mask_br]
                ax1.scatter(vis_top[mask_br, 0], vis_top[mask_br, 1], s=base_size * score_br[mask_br], c=c_br, edgecolors='none')

    if L1 is not None and L2 is not None:
        if np.linalg.norm(L1) < current_fov * 10 and np.linalg.norm(L2) < current_fov * 10:
            cell_x = [0, L1[0], L1[0]+L2[0], L2[0], 0]
            cell_y = [0, L1[1], L1[1]+L2[1], L2[1], 0]
            
            ax1.plot(cell_x, cell_y, color='yellow', linestyle='--', linewidth=2.0, alpha=0.9, zorder=5)
            ax1.annotate("", xy=L1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.5), zorder=6)
            ax1.annotate("", xy=L2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.5), zorder=6)

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
        lr = 0.05 / (1.0 + k_elastic * 10) 
        A_elec = 0.5 * (w2 - w1)**2 
        
        iterations = 120 if is_video_frame else 50
        for i in range(iterations):
            laplacian = ndimage.laplace(Z_map)
            F_elastic = k_elastic * laplacian
            F_vdw = -k_vdw * (Z_map - Z_0)
            F_elec = - A_elec / (Z_map**2)
            Z_map = Z_map + lr * (F_elastic + F_vdw + F_elec)
            Z_map = np.clip(Z_map, 1.5, 5.0) 

    final_zmin, final_zmax = np.min(Z_map), np.max(Z_map)

    # ------------------------------------------
    # PANEL 2: MIDDLE PANEL ROUTING
    # ------------------------------------------
    if mid_panel_mode == 'Geometry (Kinematic Density)':
        vmin = np.percentile(Z_stm_exact, den_contrast)
        vmax = np.percentile(Z_stm_exact, 100 - den_contrast)
        
        im2 = ax2.imshow(Z_stm_exact, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Registry-Modulated STM Topography\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='white', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.45, pad=0.04, anchor=(0.0, 0.0))
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Tunneling Density (a.u.)', color='white')
        
    elif mid_panel_mode == 'Local Doping (Δn)':
        Z_meters = Z_map * 1e-10
        epsilon_0, e_charge = 8.854e-12, 1.602e-19
        delta_W = w2 - w1 if (w2 - w1) != 0 else 1e-6
        C_geom = epsilon_0 / Z_meters
        C_total = (C_geom * layer2_Cq) / (C_geom + layer2_Cq)
        delta_n = (C_total * delta_W) / e_charge / 1e4 
        vmin = np.percentile(delta_n, den_contrast)
        vmax = np.percentile(delta_n, 100 - den_contrast)
        
        im2 = ax2.imshow(delta_n, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Local Doping in Layer 2: $\Delta n$ (cm$^{{-2}}$)\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='#ffcc00', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.45, pad=0.04, anchor=(0.0, 0.0))
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Carrier Density $\Delta n$ (cm$^{-2}$)', color='white')

    elif mid_panel_mode == 'e-ph Coupling (g)':
        g_map = eph_g0 * np.exp(-(Z_map - user_zmin) / eph_decay)
        vmin = np.percentile(g_map, den_contrast)
        vmax = np.percentile(g_map, 100 - den_contrast)
        im2 = ax2.imshow(g_map, extent=[-current_fov, current_fov, -current_fov, current_fov], origin='lower', cmap=den_cmap, vmin=vmin, vmax=vmax)
        ax2.set_title(f"Evanescent e-ph Coupling: $g(\mathbf{{r}})$\nRelaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]", color='#00ffcc', fontsize=13)
        cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.45, pad=0.04, anchor=(0.0, 0.0))
        cbar2.ax.tick_params(colors='white')
        cbar2.set_label('Coupling Strength $g$ (meV)', color='white')
        
    ax2.set_xlim(-current_fov, current_fov)
    ax2.set_ylim(-current_fov, current_fov)
    ax2.set_xlabel(r"Distance ($\AA$)", color='white')
    
    # ------------------------------------------
    # PANEL 3: LEED FFT OR STM FFT ROUTING
    # ------------------------------------------
    lbl1_short = r'SrTiO$_3$' if 'SrTiO' in label1 else (r'FeSe' if 'FeSe' in label1 else (r'Cu(110)' if 'Cu' in label1 else (r'Bi$_2$Se$_3$' if 'Bi' in label1 else 'Graphene')))
    lbl2_short = r'MoS$_2$' if 'MoS' in label2 else 'Rotated'
    
    if panel3_mode == "FFT of STM Topography (Panel 2)":
        Z_stm_centered = Z_stm_exact - np.mean(Z_stm_exact)
        fft_stm = np.fft.fftshift(np.fft.fft2(Z_stm_centered))
        intensity_stm = np.abs(fft_stm)
        
        q_freq_stm = np.fft.fftshift(np.fft.fftfreq(N_den, d=dx)) * 2 * np.pi
        q_min_stm, q_max_stm = q_freq_stm[0], q_freq_stm[-1]
        
        vmax_fft = np.percentile(intensity_stm, 99.8)
        im3 = ax3.imshow(intensity_stm, extent=[q_min_stm, q_max_stm, q_min_stm, q_max_stm], 
                         origin='lower', cmap='afmhot', 
                         norm=Normalize(vmin=0, vmax=vmax_fft))
        
        ax3.scatter(G1_pts[:, 0], G1_pts[:, 1], facecolors='none', edgecolors='cyan', s=80, linewidths=1.0, marker='o', zorder=3, alpha=0.5)
        ax3.scatter(G2_pts[:, 0], G2_pts[:, 1], facecolors='none', edgecolors='red', s=80, linewidths=1.0, marker='s', zorder=3, alpha=0.5)
        
        ax3.set_xlim(-q_max, q_max)
        ax3.set_ylim(-q_max, q_max)
        ax3.set_title(f"FFT Amplitude of STM Topography\nTwist: {theta_deg}" + r"$^\circ$" + f" | q-Zoom: {q_max} Å⁻¹", color='white', fontsize=13)
        ax3.set_xlabel(r"$q_x$ ($\AA^{-1}$)", color='white')
        
        cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.45, pad=0.04, anchor=(0.0, 0.0))
        cbar3.ax.tick_params(colors='white')
        cbar3.set_label('FFT Amplitude (a.u.)', color='white')
        
        legend_elements_3 = [
            mlines.Line2D([0], [0], color='none', marker='o', markeredgecolor='cyan', markersize=8, alpha=0.5, label=f'{lbl1_short} Peaks'),
            mlines.Line2D([0], [0], color='none', marker='s', markeredgecolor='red', markersize=8, alpha=0.5, label=f'{lbl2_short} Peaks')
        ]
        ax3.legend(handles=legend_elements_3, loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=8, framealpha=0.8, ncol=1, labelspacing=0.8)

    else:
        q_min, q_max_fft = q_freq[0], q_freq[-1]
        
        if ('Hex-on-Square' in system_mode or 'Hex-on-Rect' in system_mode) and fft_render_mode == "glass":
            sub_centered = (T_sub_fft - np.mean(T_sub_fft)) * window_2d
            top_centered = (T_top_fft - np.mean(T_top_fft)) * window_2d
            int_sub = np.abs(np.fft.fftshift(np.fft.fft2(sub_centered)))**2
            int_top = np.abs(np.fft.fftshift(np.fft.fft2(top_centered)))**2
            blur_radius = 0.5 + strain_coupling * 4.0
            int_top = ndimage.gaussian_filter(int_top, sigma=blur_radius)
            intensity = int_sub + (int_top * 3.0) + 1e-10
        else:
            T_centered_windowed = (T_fft_engine - np.mean(T_fft_engine)) * window_2d
            intensity = np.abs(np.fft.fftshift(np.fft.fft2(T_centered_windowed)))**2 + 1e-10
            intensity = ndimage.gaussian_filter(intensity, sigma=0.5) 
        
        im3 = ax3.imshow(intensity, extent=[q_min, q_max_fft, q_min, q_max_fft], origin='lower', cmap='viridis', norm=LogNorm(vmin=np.max(intensity)*1e-4, vmax=np.max(intensity)))
        
        if 'Hex-on-Rect' in system_mode:
            BZ1_pts = get_rect_bz(a_cux, a_cuy, 0.0)
        
        ax3.plot(BZ1_pts[:, 0], BZ1_pts[:, 1], color='cyan', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)
        ax3.plot(BZ2_pts[:, 0], BZ2_pts[:, 1], color='red', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)
        ax3.scatter(G1_pts[:, 0], G1_pts[:, 1], facecolors='none', edgecolors='cyan', s=120, linewidths=1.5, marker='o', zorder=3)
        ax3.scatter(G2_pts[:, 0], G2_pts[:, 1], facecolors='none', edgecolors='red', s=120, linewidths=1.5, marker='s', zorder=3)
        
        if ('Hex-on-Square' in system_mode or 'Hex-on-Rect' in system_mode) and "Rigid" not in interfacial_state and strain_coupling > 0:
            G_umklapp = []
            for g1 in G1_pts:
                for g2 in G2_pts:
                    G_umklapp.append(g1 + g2)
                    G_umklapp.append(g1 - g2)
            if G_umklapp:
                G_umklapp = np.array(G_umklapp)
                valid_mask = (np.abs(G_umklapp[:, 0]) < q_max) & (np.abs(G_umklapp[:, 1]) < q_max)
                ax3.scatter(G_umklapp[valid_mask, 0], G_umklapp[valid_mask, 1], color='yellow', s=30, marker='x', alpha=0.7, zorder=4, label='1st Order Umklapp')

        if 'Hex-on-Square' in system_mode and "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            G2_m0 = get_hex_G(a_mos2, -theta_deg)
            G2_m45 = get_hex_G(a_mos2, 90.0 - theta_deg)
            ax3.scatter(G2_m0[:, 0], G2_m0[:, 1], facecolors='none', edgecolors='orange', s=80, linewidths=1.0, marker='D', zorder=3, alpha=0.8, label='MoS$_2$ 0° Replica')
            ax3.scatter(G2_m45[:, 0], G2_m45[:, 1], facecolors='none', edgecolors='magenta', s=80, linewidths=1.0, marker='D', zorder=3, alpha=0.8, label='MoS$_2$ 45° Replica')
            
            phi1 = np.radians(theta_deg)
            phi2 = np.radians(theta_deg + 30.0)
            c1, s1 = np.cos(2*phi1), np.sin(2*phi1)
            c2, s2 = np.cos(2*phi2), np.sin(2*phi2)
            
            G1_m1 = np.array([[g[0]*c1 + g[1]*s1, g[0]*s1 - g[1]*c1] for g in G1_pts])
            G1_m2 = np.array([[g[0]*c2 + g[1]*s2, g[0]*s2 - g[1]*c2] for g in G1_pts])
            
            ax3.scatter(G1_m1[:, 0], G1_m1[:, 1], facecolors='none', edgecolors='lime', s=80, linewidths=1.0, marker='H', zorder=3, alpha=0.8, label=f'FeSe {theta_deg}° Replica')
            ax3.scatter(G1_m2[:, 0], G1_m2[:, 1], facecolors='none', edgecolors='green', s=80, linewidths=1.0, marker='H', zorder=3, alpha=0.8, label=f'FeSe {theta_deg+30}° Replica')

        elif 'Hex-on-Rect' in system_mode and "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            G2_m0 = get_hex_G(a_mos2, -theta_deg)
            G2_m90 = get_hex_G(a_mos2, 180.0 - theta_deg)
            ax3.scatter(G2_m0[:, 0], G2_m0[:, 1], facecolors='none', edgecolors='orange', s=80, linewidths=1.0, marker='D', zorder=3, alpha=0.8, label='MoS$_2$ 0° Replica')
            ax3.scatter(G2_m90[:, 0], G2_m90[:, 1], facecolors='none', edgecolors='magenta', s=80, linewidths=1.0, marker='D', zorder=3, alpha=0.8, label='MoS$_2$ 90° Replica')
            
            phi1 = np.radians(theta_deg)
            phi2 = np.radians(theta_deg + 30.0)
            c1, s1 = np.cos(2*phi1), np.sin(2*phi1)
            c2, s2 = np.cos(2*phi2), np.sin(2*phi2)
            
            G1_m1 = np.array([[g[0]*c1 + g[1]*s1, g[0]*s1 - g[1]*c1] for g in G1_pts])
            G1_m2 = np.array([[g[0]*c2 + g[1]*s2, g[0]*s2 - g[1]*c2] for g in G1_pts])
            
            ax3.scatter(G1_m1[:, 0], G1_m1[:, 1], facecolors='none', edgecolors='lime', s=80, linewidths=1.0, marker='H', zorder=3, alpha=0.8, label=f'Cu {theta_deg}° Replica')
            ax3.scatter(G1_m2[:, 0], G1_m2[:, 1], facecolors='none', edgecolors='green', s=80, linewidths=1.0, marker='H', zorder=3, alpha=0.8, label=f'Cu {theta_deg+30}° Replica')
            
        ax3.set_xlim(-q_max, q_max)
        ax3.set_ylim(-q_max, q_max)
        ax3.set_title(f"Scattering (Simulated LEED)\nTwist: {theta_deg}" + r"$^\circ$" + f" | q-Zoom: {q_max} Å⁻¹", color='white', fontsize=13)
        ax3.set_xlabel(r"$q_x$ ($\AA^{-1}$)", color='white')
        cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.45, pad=0.04, anchor=(0.0, 0.0))
        cbar3.ax.tick_params(colors='white')
        cbar3.set_label('Scattering Intensity (a.u.)', color='white')

        legend_elements_3 = [
            mlines.Line2D([0], [0], color='none', marker='o', markeredgecolor='cyan', markersize=8, label=f'{lbl1_short} Peaks'),
            mlines.Line2D([0], [0], color='none', marker='s', markeredgecolor='red', markersize=8, label=f'{lbl2_short} Peaks'),
            mlines.Line2D([0], [0], color='cyan', linestyle=':', lw=1.5, label=f'{lbl1_short} 1st BZ'),
            mlines.Line2D([0], [0], color='red', linestyle=':', lw=1.5, label=f'{lbl2_short} 1st BZ'),
            mlines.Line2D([0], [0], color='cyan', linestyle='-', lw=1.5, label=r'Recip. Vec. $\mathbf{g}_1$'),
            mlines.Line2D([0], [0], color='red', linestyle='-', lw=1.5, label=r'Recip. Vec. $\mathbf{g}_2$'),
            mlines.Line2D([0], [0], color='yellow', linestyle='--', lw=1.5, label=r'Moiré Vecs. $\mathbf{q}_{M1}, \mathbf{q}_{M2}$')
        ]
        
        if ('Hex-on-Square' in system_mode or 'Hex-on-Rect' in system_mode) and "Rigid" not in interfacial_state and strain_coupling > 0:
            legend_elements_3.append(mlines.Line2D([0], [0], color='none', marker='x', markeredgecolor='yellow', markersize=8, label='1st Order Umklapp'))
            
        if 'Hex-on-Square' in system_mode and "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            legend_elements_3.extend([
                mlines.Line2D([0], [0], color='none', marker='D', markeredgecolor='orange', markersize=8, label='MoS$_2$ 0° Replica'),
                mlines.Line2D([0], [0], color='none', marker='D', markeredgecolor='magenta', markersize=8, label='MoS$_2$ 45° Replica'),
                mlines.Line2D([0], [0], color='none', marker='H', markeredgecolor='lime', markersize=8, label=f'FeSe {theta_deg}° Replica'),
                mlines.Line2D([0], [0], color='none', marker='H', markeredgecolor='green', markersize=8, label=f'FeSe {theta_deg+30}° Replica')
            ])
            
        if 'Hex-on-Rect' in system_mode and "Dodecagonal Quasicrystal" in interfacial_state and strain_coupling > 0:
            legend_elements_3.extend([
                mlines.Line2D([0], [0], color='none', marker='D', markeredgecolor='orange', markersize=8, label='MoS$_2$ 0° Replica'),
                mlines.Line2D([0], [0], color='none', marker='D', markeredgecolor='magenta', markersize=8, label='MoS$_2$ 90° Replica'),
                mlines.Line2D([0], [0], color='none', marker='H', markeredgecolor='lime', markersize=8, label=f'Cu {theta_deg}° Replica'),
                mlines.Line2D([0], [0], color='none', marker='H', markeredgecolor='green', markersize=8, label=f'Cu {theta_deg+30}° Replica')
            ])

        ax3.legend(handles=legend_elements_3, loc='upper left', bbox_to_anchor=(1.05, 1.0), fontsize=8, framealpha=0.8, ncol=1, labelspacing=0.8)

    # ------------------------------------------
    # PANEL 4: EXTENDED FERMI SURFACE MAP (FeSe ONLY)
    # ------------------------------------------
    if show_fs_panel and ax4 is not None:
        ax4.set_xlim(-k_max, k_max)
        ax4.set_ylim(-k_max, k_max)
        ax4.set_title(f"Fermi Surface Extended BZ (Mutual Band Folding)\n{interfacial_state.split(':')[0]}", color='white', fontsize=15)
        ax4.set_xlabel(r"$k_x$ ($\AA^{-1}$)", color='white')
        ax4.set_ylabel(r"$k_y$ ($\AA^{-1}$)", color='white')

        ax4.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax4.axvline(0, color='gray', lw=0.5, alpha=0.5)

        r_fese = 0.175
        r_mos2 = 0.10
        
        q_fese = 2 * np.pi / a_fese
        G1_all, G1_shell1, G1_shell2 = [], [], []
        for n in range(-6, 7):
            for m in range(-6, 7):
                G = np.array([n * q_fese, m * q_fese])
                G1_all.append(G)
                d = np.linalg.norm(G)
                if 0.1 < d < q_fese * 1.1: G1_shell1.append(G)
                elif q_fese * 1.1 < d < q_fese * 1.5: G1_shell2.append(G)
                
        q_mos2 = 4 * np.pi / (np.sqrt(3) * a_mos2)
        th_rad = np.radians(theta_deg)
        R_m = np.array([[np.cos(th_rad), -np.sin(th_rad)], [np.sin(th_rad), np.cos(th_rad)]])
        g2_A = np.array([0, q_mos2]).dot(R_m.T)
        g2_B = np.array([q_mos2*np.sqrt(3)/2, -q_mos2*0.5]).dot(R_m.T)
        
        G2_all, G2_shell1, G2_shell2 = [], [], []
        for n in range(-6, 7):
            for m in range(-6, 7):
                G = n*g2_A + m*g2_B
                G2_all.append(G)
                d = np.linalg.norm(G)
                if 0.1 < d < q_mos2 * 1.1: G2_shell1.append(G)
                elif q_mos2 * 1.1 < d < q_mos2 * 1.8: G2_shell2.append(G)
                
        M_bases = [np.array([q_fese/2, q_fese/2]), np.array([-q_fese/2, q_fese/2]), np.array([-q_fese/2, -q_fese/2]), np.array([q_fese/2, -q_fese/2])]
        all_M_pts = [mb + G for G in G1_all for mb in M_bases]
        all_M_pts = np.array(all_M_pts)
        _, idx = np.unique(np.round(all_M_pts, 4), axis=0, return_index=True)
        M_pts = all_M_pts[idx]
        
        K_mag = q_mos2 / np.sqrt(3)
        K_angles = np.radians(np.arange(0, 360, 60))
        K_bases = [np.array([K_mag * np.cos(a), K_mag * np.sin(a)]).dot(R_m.T) for a in K_angles]
        all_K_pts = [kb + G for G in G2_all for kb in K_bases]
        all_K_pts = np.array(all_K_pts)
        _, idx = np.unique(np.round(all_K_pts, 4), axis=0, return_index=True)
        K_pts = all_K_pts[idx]
        
        BZ_sq_base = get_square_bz(a_fese, 0.0)
        BZ_hex_base = get_hex_bz(a_mos2, theta_deg)
        
        def plot_bzs(ax, G_list, BZ_base, color, alpha):
            for G in G_list:
                if np.abs(G[0]) > k_max + 1 or np.abs(G[1]) > k_max + 1: continue
                shifted = BZ_base + G
                ax.plot(shifted[:,0], shifted[:,1], color=color, ls='-', lw=1.5, alpha=max(alpha, 0.4))
                
        plot_bzs(ax4, G1_all, BZ_sq_base, 'cyan', 0.5)
        plot_bzs(ax4, G2_all, BZ_hex_base, 'red', 0.5)

        def plot_fs(ax, centers, r, col, lw, ls, alpha, lbl=None):
            for idx, pt in enumerate(centers):
                if abs(pt[0]) > k_max + r or abs(pt[1]) > k_max + r: continue
                l = lbl if idx == 0 else None
                ax.add_patch(plt.Circle((pt[0], pt[1]), r, color=col, fill=False, lw=lw, ls=ls, alpha=alpha, label=l))

        plot_fs(ax4, M_pts, r_fese, 'cyan', 2.5, '-', 1.0, 'Primary FeSe (M-pts)')
        plot_fs(ax4, K_pts, r_mos2, 'red', 2.5, '-', 1.0, 'Primary MoS$_2$ (K-pts)')

        legend_elements_4 = [
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=10, lw=2.5, label='Primary FeSe FS'),
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=10, lw=2.5, label='Primary MoS$_2$ FS')
        ]

        if strain_coupling > 0 and "Rigid" not in interfacial_state:
            is_glass = "Glass" in interfacial_state
            
            alpha1 = 0.6 if is_glass else 0.8
            alpha2 = 0.3 if is_glass else 0.5
            lw_mod1 = 3.0 if is_glass else 1.5
            lw_mod2 = 3.0 if is_glass else 1.0
            
            M_1st = [m + g for m in M_pts for g in G2_shell1]
            K_1st = [k + g for k in K_pts for g in G1_shell1]
            plot_fs(ax4, M_1st, r_fese, 'cyan', lw_mod1, '--', alpha1 * strain_coupling)
            plot_fs(ax4, K_1st, r_mos2, 'red', lw_mod1, '--', alpha1 * strain_coupling)
            
            M_2nd = [m + g for m in M_pts for g in G2_shell2]
            K_2nd = [k + g for k in K_pts for g in G1_shell2]
            plot_fs(ax4, M_2nd, r_fese, 'cyan', lw_mod2, ':', alpha2 * strain_coupling)
            plot_fs(ax4, K_2nd, r_mos2, 'red', lw_mod2, ':', alpha2 * strain_coupling)

            legend_elements_4.extend([
                mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=10, lw=lw_mod1, ls='--', alpha=alpha1, label='FeSe Folded (1st Order)'),
                mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=10, lw=lw_mod1, ls='--', alpha=alpha1, label='MoS$_2$ Folded (1st Order)'),
                mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=8, lw=lw_mod2, ls=':', alpha=alpha2, label='FeSe Folded (2nd Order)'),
                mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=8, lw=lw_mod2, ls=':', alpha=alpha2, label='MoS$_2$ Folded (2nd Order)')
            ])

            if "Quasicrystal" in interfacial_state:
                K_m0 = np.array([[k[0], -k[1]] for k in K_pts])
                K_m45 = np.array([[k[1], k[0]] for k in K_pts])
                
                plot_fs(ax4, K_m0, r_mos2 * 1.30, 'orange', 2.5, '--', 0.9)
                plot_fs(ax4, K_m45, r_mos2 * 1.08, 'magenta', 2.0, '--', 0.9)

                legend_elements_4.extend([
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='orange', markersize=10, lw=2.5, ls='--', alpha=0.9, label='MoS$_2$ 0° Replica'),
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='magenta', markersize=10, lw=2.0, ls='--', alpha=0.9, label='MoS$_2$ 45° Replica')
                ])
                
                phi1 = np.radians(theta_deg)
                phi2 = np.radians(theta_deg + 30.0)
                c1, s1 = np.cos(2*phi1), np.sin(2*phi1)
                c2, s2 = np.cos(2*phi2), np.sin(2*phi2)
                
                M_m1 = np.array([[m[0]*c1 + m[1]*s1, m[0]*s1 - m[1]*c1] for m in M_pts])
                M_m2 = np.array([[m[0]*c2 + m[1]*s2, m[0]*s2 - m[1]*c2] for m in M_pts])
                
                plot_fs(ax4, M_m1, r_fese * 1.08, 'lime', 2.0, '--', 0.9)
                plot_fs(ax4, M_m2, r_fese * 1.08, 'green', 2.0, '--', 0.9)
                
                legend_elements_4.extend([
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='lime', markersize=10, lw=2.0, ls='--', alpha=0.9, label=f'FeSe {theta_deg}° Replica'),
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='green', markersize=10, lw=2.0, ls='--', alpha=0.9, label=f'FeSe {theta_deg+30}° Replica')
                ])

        ax4.legend(handles=legend_elements_4, loc='upper right', bbox_to_anchor=(1.03, 1.02), fontsize=10, framealpha=0.9)
        
    return fig

# ==========================================
# 4. STREAMLIT UI CONTROLS
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    system_mode = st.selectbox("System:", ['MoS₂/SrTiO₃ (Hex-on-Square)', 'MoS₂/FeSe (Hex-on-Square)', 'MoS₂/Cu(110) (Hex-on-Rect)', 'MoS₂/Bi₂Se₃ (Hex-on-Hex)', 'MoS₂/Graphene (Hex-on-Hex)', 'MATBG (Hex-on-Hex)'])
    view_mode = st.selectbox("Topology View:", ['Show All Registries', 'Coincident + Hollow', 'Coincident Only', 'Hollow Only', 'Bridge Only', 'Raw Lattices'])
    boundary_mode = st.selectbox("Domain Boundaries:", ["None", "Microscopic (Atomic)", "Mesoscopic (Envelope)"])

with col2:
    mid_panel_mode = st.radio("Middle Panel Metric:", ["Geometry (Kinematic Density)", "Local Doping (Δn)", "e-ph Coupling (g)"], horizontal=True)
    den_cmap = st.selectbox("Panel 2 Color:", ['magma', 'viridis', 'plasma', 'cividis', 'gray', 'bone', 'coolwarm', 'afmhot'])
    panel3_mode = st.radio("Panel 3 Mode:", ["Scattering (Simulated LEED)", "FFT of STM Topography (Panel 2)"])

with col3:
    max_theta = 60.0 if 'Hex-on-Hex' in system_mode else 90.0
    theta_deg = st.slider("Twist Angle (deg):", 0.0, max_theta, 0.0, 0.1)
    zoom_factor = st.slider("FOV Zoom (x):", 1.0, 5.0, 1.0, 0.5)
    
    kcol3a, kcol3b = st.columns(2)
    with kcol3a:
        q_max = st.slider("q-Zoom (Å⁻¹):", 1.0, 8.0, 4.0, 0.5)
    with kcol3b:
        k_max = st.slider("k-Zoom (Panel 4) (Å⁻¹):", 2.0, 15.0, 6.0, 0.5)
    den_contrast = st.slider("Contrast Clip (%):", 0.0, 20.0, 0.0, 1.0)

# --- EXPANDER FOR ADVANCED PHYSICS PARAMETERS ---
with st.expander("⚙️ Advanced Physics Parameters (Interfacial Mechanics & e-ph Coupling)", expanded=True):
    
    st.markdown("**1. Interfacial Phase Engineering**")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        interfacial_state = st.selectbox("Interfacial Phase State:", [
            "1. Rigid vdW Gap (Non-interacting)", 
            "2. Misfit Dislocation Glass (Experimental Blobs)", 
            "3. Dodecagonal Quasicrystal (Theoretical CDW)"
        ])
    with pcol2:
        strain_coupling = st.slider("Interfacial Coupling Strength", 0.0, 1.0, 0.4, 0.1, help="Scales domain mosaicity (Mode 2) or resonant reflection (Mode 3).")
        
    st.markdown("---")
    st.markdown("**2. Interfacial Mechanics & Doping Model**")
    
    relax_mode = st.selectbox("Mechanical Relaxation Model:", [
        "Rigid Lattices (No Relaxation)", 
        "Fast Proxy (Algebraic Shift)", 
        "Continuum Mechanics (PDE Solver)"
    ])
    
    intrinsic_z = {
        'MoS₂/SrTiO₃ (Hex-on-Square)': (3.1, 3.6),
        'MoS₂/FeSe (Hex-on-Square)': (3.2, 3.6),
        'MoS₂/Cu(110) (Hex-on-Rect)': (2.8, 3.4),
        'MoS₂/Bi₂Se₃ (Hex-on-Hex)': (3.2, 3.6),
        'MoS₂/Graphene (Hex-on-Hex)': (3.3, 3.6),
        'MATBG (Hex-on-Hex)': (3.35, 3.6)
    }
    base_zmin, base_zmax = intrinsic_z[system_mode]

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        w1 = st.number_input("Layer 1 Work Function (eV)", value=4.2, step=0.1)
    with kcol2:
        w2 = st.number_input("Layer 2 Work Function (eV)", value=4.5, step=0.1)

    with kcol3:
        user_zmin = st.number_input("Base Unrelaxed Min Gap (Å)", value=float(base_zmin), step=0.1)
    with kcol4:
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
    st.markdown("**3. Local Electron-Phonon Coupling Model**")
    ecol1, ecol2, ecol3, ecol4 = st.columns(4)
    with ecol1:
        eph_g0 = st.number_input("Base Coupling at min gap (meV)", value=80.0, step=5.0)
    with ecol2:
        eph_decay = st.number_input("Evanescent Decay Length $\lambda$ (Å)", value=0.5, step=0.1)

# Render the single unified plot with a locking spinner container
dashboard_placeholder = st.empty()

with st.spinner("Re-calculating physics models and rendering panels... Please wait."):
    with dashboard_placeholder.container():
        fig = create_unified_plot(None, cached_data, system_mode, theta_deg, zoom_factor, q_max, k_max, view_mode, boundary_mode, mid_panel_mode, panel3_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay, interfacial_state, strain_coupling, is_video_frame=False)
        st.pyplot(fig)

# --- VIDEO GENERATOR (CINEMATIC TOOLS) ---
st.markdown("---")
st.markdown("### 🎥 Cinematic Tools")

max_t_int = int(max_theta)
if st.button(f"Generate Twist Angle Scan Video (0° to {max_t_int}°)"):
    st.session_state.is_rendering_video = True
    st.session_state.video_sys_name = system_mode.replace("/", "_").split(" ")[0]
    
    vid_progress = st.progress(0, text=f"Rendering frame 1 of {max_t_int + 1}...")
    
    frames = []
    video_fig = Figure(figsize=(24, 18) if 'FeSe' in system_mode else (21, 6.5), dpi=100) 
    FigureCanvasAgg(video_fig)
    
    for ang in range(max_t_int + 1):
        vid_progress.progress(int((ang / max_t_int) * 100), text=f"Rendering frame {ang + 1} of {max_t_int + 1} (Twist: {ang}°)...")
        
        fig_frame = create_unified_plot(video_fig, cached_data, system_mode, float(ang), zoom_factor, q_max, k_max, view_mode, boundary_mode, mid_panel_mode, panel3_mode, den_cmap, den_contrast, relax_mode, w1, w2, user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay, interfacial_state, strain_coupling, is_video_frame=True)
        
        fig_frame.canvas.draw()
        img_rgba = np.asarray(fig_frame.canvas.buffer_rgba())
        
        img = img_rgba[:, :, :3].copy() 
        frames.append(img)
        
    vid_progress.progress(100, text="Encoding MP4 Video...")
    imageio.mimsave("moire_twist_scan.mp4", frames, fps=4, macro_block_size=None)
    vid_progress.empty()
    st.session_state.is_rendering_video = False
    
    st.success("Video Generated Successfully!")
    
    with open("moire_twist_scan.mp4", "rb") as file:
        st.session_state.video_bytes = file.read()

if "video_bytes" in st.session_state:
    st.video(st.session_state.video_bytes, autoplay=True, loop=True)
    dl_name = st.session_state.get('video_sys_name', 'System')
    st.download_button(
        label="💾 Save Video to Computer",
        data=st.session_state.video_bytes,
        file_name=f"twist_scan_{dl_name}.mp4",
        mime="video/mp4"
    )
