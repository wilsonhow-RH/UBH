"""
UHV-bonded Heterostructure Physics Dashboard
============================================

Interactive exploration of the topology, geometry, scattering, local doping and
many-body interactions of 2D UHV-bonded heterostructures.

Structure of this file
----------------------
  1. Constants and the SYSTEMS registry (single source of truth per material system)
  2. Cached lattice / FFT grid generators
  3. Geometry + reciprocal-space helpers
  4. Interface-field builder (real-space densities, registry scores, G/BZ sets)
  5. Master plotting routine (Panels 1-4)
  6. Streamlit UI (only executed when run as a script)

Adding a new material system only requires a new entry in SYSTEMS.
"""

import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg
import scipy.ndimage as ndimage
import imageio

APP_VERSION = "v. Sep 30, 2026"

# ==========================================
# 1. CONSTANTS & SYSTEM REGISTRY
# ==========================================

# --- Bulk / surface lattice parameters (Angstrom) ---
A_STO = 3.905                     # SrTiO3 cubic lattice constant
A_STO_110 = A_STO * np.sqrt(2)    # SrTiO3(110) in-plane period along [1-10] = 5.5225
A_FESE = 3.905                    # FeSe (1ML on STO) in-plane
A_MOS2 = 3.15                     # 1ML MoS2 (Mo sublattice)
A_BISE = 4.14                     # Bi2Se3 (0001)
A_GRAPHENE = 2.46                 # graphene
A_CU_FCC = 3.61                   # Cu fcc lattice constant
A_CU_NN = A_CU_FCC / np.sqrt(2)   # Cu nearest-neighbour row spacing = 2.5527

# --- Real-space / reciprocal-space grid sizes ---
BASE_GRID = 40.0                  # half-FOV at zoom = 1x (Angstrom)
ZOOM_MAX = 5.0
FOV_MAX = BASE_GRID * ZOOM_MAX    # 200 A
SUB_EXTENT = FOV_MAX              # substrate lattice generated out to the max FOV
TOP_EXTENT = FOV_MAX * 1.5        # overlayer generated wider (it gets rotated)

N_DEN = 512                       # real-space density / STM topography grid
N_FFT = 512                       # kinematic-scattering grid
L_FFT = 400.0                     # side length of the scattering grid (Angstrom)

# Glassy displacement field (misfit-dislocation mode)
GLASS_N_WAVES = 40
GLASS_K0 = 2 * np.pi / 120.0
GLASS_SEED = 42

# STM topography model
SIGMA_ATOM = 1.0                  # apparent atomic radius (Angstrom)
XI_STACK = 0.6                    # registry-contrast decay length (Angstrom)
A_MODULATION = 1.1                # coincident-site brightness enhancement


# --- Material-system registry -------------------------------------------------
#
# geometry:
#   "ortho" -> hexagonal overlayer on a square / rectangular surface mesh
#              (square is simply the ax == ay special case)
#   "hex"   -> hexagonal overlayer on a hexagonal surface mesh
#
# Axis convention for the rectangular meshes: x is taken along the cubic [001]
# in-plane direction of the substrate surface, y along [1-10].
#   Cu(110):     [001] = a0        = 3.6100 A   |  [1-10] = a0/sqrt(2) = 2.5527 A
#   SrTiO3(110): [001] = a0        = 3.9050 A   |  [1-10] = a0*sqrt(2) = 5.5225 A
#   Cu(100):     square mesh, a = a0/sqrt(2)    = 2.5527 A
#
# mirrors: mirror lines of the substrate mesh used to build the dodecagonal
#   quasicrystal replicas (square: 0 deg and 45 deg; rectangular: 0 deg and 90 deg).
#
# z0: default (unrelaxed min, unrelaxed max) interfacial gap in Angstrom. These
#   are editable defaults in the UI, not fixed physical constants.

SYSTEMS = {
    "MoS₂/SrTiO₃(100) (Hex-on-Square)": dict(
        geometry="ortho", sub_ax=A_STO, sub_ay=A_STO, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on SrTiO$_3$(100)",
        label1=r"Layer 1 (SrTiO$_3$(100))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"SrTiO$_3$(100)", top_short=r"MoS$_2$",
        mirrors=(0.0, 45.0), z0=(3.1, 3.6), max_theta=90.0,
        fs_panel=False, cq=0.01,
    ),
    "MoS₂/SrTiO₃(110) (Hex-on-Rect)": dict(
        geometry="ortho", sub_ax=A_STO, sub_ay=A_STO_110, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on SrTiO$_3$(110)",
        label1=r"Layer 1 (SrTiO$_3$(110))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"SrTiO$_3$(110)", top_short=r"MoS$_2$",
        mirrors=(0.0, 90.0), z0=(3.1, 3.6), max_theta=90.0,
        fs_panel=False, cq=0.01,
    ),
    "MoS₂/FeSe (Hex-on-Square)": dict(
        geometry="ortho", sub_ax=A_FESE, sub_ay=A_FESE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on 1ML FeSe",
        label1=r"Layer 1 (FeSe)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"FeSe", top_short=r"MoS$_2$",
        mirrors=(0.0, 45.0), z0=(3.2, 3.6), max_theta=90.0,
        fs_panel=True, cq=0.01,
    ),
    "MoS₂/Cu(100) (Hex-on-Square)": dict(
        geometry="ortho", sub_ax=A_CU_NN, sub_ay=A_CU_NN, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Cu(100)",
        label1=r"Layer 1 (Cu(100))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Cu(100)", top_short=r"MoS$_2$",
        mirrors=(0.0, 45.0), z0=(2.8, 3.4), max_theta=90.0,
        fs_panel=False, cq=0.01,
    ),
    "MoS₂/Cu(110) (Hex-on-Rect)": dict(
        geometry="ortho", sub_ax=A_CU_FCC, sub_ay=A_CU_NN, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Cu(110)",
        label1=r"Layer 1 (Cu(110))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Cu(110)", top_short=r"MoS$_2$",
        mirrors=(0.0, 90.0), z0=(2.8, 3.4), max_theta=90.0,
        fs_panel=False, cq=0.01,
    ),
    "MoS₂/Bi₂Se₃ (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_BISE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on 6QL Bi$_2$Se$_3$",
        label1=r"Layer 1 (Bi$_2$Se$_3$)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Bi$_2$Se$_3$", top_short=r"MoS$_2$",
        mirrors=(), z0=(3.2, 3.6), max_theta=60.0,
        fs_panel=False, cq=0.01,
    ),
    "MoS₂/Graphene (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_GRAPHENE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Graphene",
        label1=r"Layer 1 (Graphene)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Graphene", top_short=r"MoS$_2$",
        mirrors=(), z0=(3.3, 3.6), max_theta=60.0,
        fs_panel=False, cq=0.01,
    ),
    "MATBG (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_GRAPHENE, top_a=A_GRAPHENE,
        title="Magic-Angle Twisted Bilayer Graphene",
        label1="Layer 1 (Graphene)", label2="Layer 2 (Rotated)",
        sub_short=r"Graphene", top_short="Rotated",
        mirrors=(), z0=(3.35, 3.6), max_theta=60.0,
        fs_panel=False, cq=0.002,
    ),
}


def supports_interfacial_state(cfg):
    """Glass / quasicrystal interfacial modes are defined for hex-on-ortho only."""
    return cfg["geometry"] == "ortho"


# ==========================================
# 2. CACHED GRID GENERATORS
# ==========================================
@st.cache_data(show_spinner=False)
def ortho_lattice(ax_, ay_, extent):
    """Rectangular (or square) point lattice, centred exactly on the origin."""
    nx = int(np.floor(extent / ax_))
    ny = int(np.floor(extent / ay_))
    x = np.arange(-nx, nx + 1) * ax_
    y = np.arange(-ny, ny + 1) * ay_
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


@st.cache_data(show_spinner=False)
def hex_lattice(a, extent):
    """Triangular point lattice (primitive vectors at 60 deg), centred on the origin."""
    v1 = np.array([a, 0.0])
    v2 = np.array([a * 0.5, a * np.sqrt(3) / 2])
    n = int(np.ceil(2 * extent / a)) + 2
    idx = np.arange(-n, n + 1)
    nn, mm = np.meshgrid(idx, idx)
    pts = nn.ravel()[:, None] * v1 + mm.ravel()[:, None] * v2
    keep = (np.abs(pts[:, 0]) <= extent) & (np.abs(pts[:, 1]) <= extent)
    return pts[keep]


@st.cache_data(show_spinner=False)
def hex_basis(a):
    """Column-vector basis matrix V and its inverse for a triangular lattice."""
    V = np.array([[a, a * 0.5], [0.0, a * np.sqrt(3) / 2]])
    return V, np.linalg.inv(V)


@st.cache_resource(show_spinner=False)
def fft_grids():
    """Fixed real-space grid and matching q-axis used for the kinematic LEED panel."""
    x = np.linspace(-L_FFT / 2, L_FFT / 2, N_FFT)
    X, Y = np.meshgrid(x, x)
    q = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=(L_FFT / N_FFT))) * 2 * np.pi
    w1 = np.hanning(N_FFT)
    return X, Y, q, w1[:, None] * w1[None, :]


@st.cache_resource(show_spinner=False)
def stm_window(n):
    w1 = np.hanning(n)
    return w1[:, None] * w1[None, :]


# ==========================================
# 3. GEOMETRY & RECIPROCAL-SPACE HELPERS
# ==========================================
def min_hex_dist(frac_pts, V_mat):
    """Distance to the nearest of the four surrounding triangular-lattice nodes."""
    f_floor = np.floor(frac_pts)
    d00 = np.linalg.norm((frac_pts - f_floor).dot(V_mat.T), axis=1)
    d10 = np.linalg.norm((frac_pts - (f_floor + np.array([1, 0]))).dot(V_mat.T), axis=1)
    d01 = np.linalg.norm((frac_pts - (f_floor + np.array([0, 1]))).dot(V_mat.T), axis=1)
    d11 = np.linalg.norm((frac_pts - (f_floor + np.array([1, 1]))).dot(V_mat.T), axis=1)
    return np.minimum(np.minimum(d00, d10), np.minimum(d01, d11))


def hex_registry_distances(vis_top, V_sub, invV_sub):
    """Coincident / hollow / bridge distances for a hexagonal substrate mesh."""
    f = vis_top.dot(invV_sub.T)
    dist_co = min_hex_dist(f, V_sub)
    dist_ho = np.minimum(
        min_hex_dist(f - np.array([1 / 3, 1 / 3]), V_sub),
        min_hex_dist(f - np.array([2 / 3, 2 / 3]), V_sub),
    )
    dist_br = np.minimum(
        np.minimum(
            min_hex_dist(f - np.array([0.5, 0.0]), V_sub),
            min_hex_dist(f - np.array([0.0, 0.5]), V_sub),
        ),
        min_hex_dist(f - np.array([0.5, -0.5]), V_sub),
    )
    return dist_co, dist_ho, dist_br


def ortho_registry_distances(vis_top, ax_, ay_):
    """Coincident / hollow / bridge distances for a rectangular substrate mesh."""
    nx = np.round(vis_top[:, 0] / ax_) * ax_
    ny = np.round(vis_top[:, 1] / ay_) * ay_
    hx = np.floor(vis_top[:, 0] / ax_) * ax_ + ax_ / 2
    hy = np.floor(vis_top[:, 1] / ay_) * ay_ + ay_ / 2

    dist_co = np.hypot(vis_top[:, 0] - nx, vis_top[:, 1] - ny)
    dist_ho = np.hypot(vis_top[:, 0] - hx, vis_top[:, 1] - hy)
    dist_br = np.minimum(
        np.hypot(vis_top[:, 0] - hx, vis_top[:, 1] - ny),
        np.hypot(vis_top[:, 0] - nx, vis_top[:, 1] - hy),
    )
    return dist_co, dist_ho, dist_br


def get_ortho_density(ax_, ay_, X, Y):
    return 2.0 + np.cos(2 * np.pi / ax_ * X) + np.cos(2 * np.pi / ay_ * Y)


def get_hex_density(a, X, Y, theta_deg):
    th = np.radians(theta_deg)
    Xr = X * np.cos(th) + Y * np.sin(th)
    Yr = -X * np.sin(th) + Y * np.cos(th)
    q = 4 * np.pi / (np.sqrt(3) * a)
    return (3.0 + np.cos(q * Yr)
            + np.cos(q * (np.sqrt(3) / 2 * Xr - 0.5 * Yr))
            + np.cos(q * (-np.sqrt(3) / 2 * Xr - 0.5 * Yr)))


def get_reflected_coords(X, Y, phi_deg):
    """Reflect coordinates across a line through the origin at angle phi."""
    phi = np.radians(phi_deg)
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    return X * c + Y * s, X * s - Y * c


def rotation_matrix(theta_deg):
    th = np.radians(theta_deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def get_ortho_G(ax_, ay_):
    qx, qy = 2 * np.pi / ax_, 2 * np.pi / ay_
    return np.array([[qx, 0], [-qx, 0], [0, qy], [0, -qy]])


def get_hex_G(a, theta_deg):
    q = 4 * np.pi / (np.sqrt(3) * a)
    base = np.array([[0, q], [0, -q],
                     [q * np.sqrt(3) / 2, -q * 0.5], [-q * np.sqrt(3) / 2, q * 0.5],
                     [-q * np.sqrt(3) / 2, -q * 0.5], [q * np.sqrt(3) / 2, q * 0.5]])
    return base.dot(rotation_matrix(theta_deg).T)


def get_ortho_bz(ax_, ay_, theta_deg=0.0):
    qx, qy = 2 * np.pi / ax_, 2 * np.pi / ay_
    base = np.array([[qx / 2, qy / 2], [-qx / 2, qy / 2],
                     [-qx / 2, -qy / 2], [qx / 2, -qy / 2], [qx / 2, qy / 2]])
    return base.dot(rotation_matrix(theta_deg).T)


def get_hex_bz(a, theta_deg=0.0):
    q = 4 * np.pi / (np.sqrt(3) * a)
    R_bz = q / np.sqrt(3)
    ang = np.radians([0, 60, 120, 180, 240, 300, 360])
    base = np.column_stack([R_bz * np.cos(ang), R_bz * np.sin(ang)])
    return base.dot(rotation_matrix(theta_deg).T)


# --- Misfit-dislocation ("glassy") displacement field -------------------------
@st.cache_resource(show_spinner=False)
def _glass_waves():
    rng = np.random.default_rng(GLASS_SEED)
    ks = rng.normal(GLASS_K0, GLASS_K0 * 0.2, GLASS_N_WAVES)
    thetas = rng.uniform(0, 2 * np.pi, GLASS_N_WAVES)
    phases = rng.uniform(0, 2 * np.pi, GLASS_N_WAVES)
    return ks * np.cos(thetas), ks * np.sin(thetas), phases


def _glass_raw(X, Y):
    kx, ky, ph = _glass_waves()
    Ux = np.zeros_like(X, dtype=float)
    Uy = np.zeros_like(Y, dtype=float)
    for i in range(GLASS_N_WAVES):
        wave = np.sin(X * kx[i] + Y * ky[i] + ph[i])
        Ux += -ky[i] * wave
        Uy += kx[i] * wave
    return Ux, Uy


@st.cache_resource(show_spinner=False)
def _glass_norm():
    """One fixed normalisation constant, shared by grid- and point-sampled calls.

    Normalising each call by its own sample maximum (as the original code did)
    made the displacement amplitude depend on the sampling set and on the zoom
    level, so the atom overlay and the density map were displaced by different
    amounts. A single reference constant removes that inconsistency.
    """
    g = np.linspace(-FOV_MAX, FOV_MAX, 256)
    X, Y = np.meshgrid(g, g)
    Ux, Uy = _glass_raw(X, Y)
    return float(np.max(np.hypot(Ux, Uy))) + 1e-10


def glass_displacement(X, Y, coupling):
    """Divergence-free random displacement field, amplitude ~ 1.5 * coupling (A)."""
    if coupling <= 0:
        return np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)
    Ux, Uy = _glass_raw(X, Y)
    amp = coupling * 1.5 / _glass_norm()
    return Ux * amp, Uy * amp


# ==========================================
# 4. INTERFACE FIELD BUILDER
# ==========================================
def build_interface(cfg, theta_deg, current_fov, X_den, Y_den,
                    interfacial_state, strain_coupling):
    """Assemble every system-dependent quantity used by Panels 1-3.

    Returns a dict with the visible lattices, the registry scores, the real-space
    density used by the relaxation solver, the reciprocal-space engine used by the
    LEED panel, and the G-vector / Brillouin-zone overlays.
    """
    X_fft, Y_fft, _q, _w = fft_grids()
    R = rotation_matrix(theta_deg)
    top_a = cfg["top_a"]

    glass = (supports_interfacial_state(cfg)
             and "Misfit Dislocation Glass" in interfacial_state
             and strain_coupling > 0)
    qc = (supports_interfacial_state(cfg)
          and "Dodecagonal Quasicrystal" in interfacial_state
          and strain_coupling > 0)

    # ---- visible lattices -------------------------------------------------
    if cfg["geometry"] == "ortho":
        ax_, ay_ = cfg["sub_ax"], cfg["sub_ay"]
        pts_sub = ortho_lattice(ax_, ay_, SUB_EXTENT)
        decay_L = 0.25 * ax_
        T_sub_den = get_ortho_density(ax_, ay_, X_den, Y_den)
        T_sub_fft = get_ortho_density(ax_, ay_, X_fft, Y_fft)
        G1_pts = get_ortho_G(ax_, ay_)
        BZ1_pts = get_ortho_bz(ax_, ay_, 0.0)
    else:
        sub_a = cfg["sub_a"]
        pts_sub = hex_lattice(sub_a, SUB_EXTENT)
        decay_L = 0.25 * sub_a
        T_sub_den = get_hex_density(sub_a, X_den, Y_den, 0.0)
        T_sub_fft = get_hex_density(sub_a, X_fft, Y_fft, 0.0)
        G1_pts = get_hex_G(sub_a, 0.0)
        BZ1_pts = get_hex_bz(sub_a, 0.0)

    pts_top = hex_lattice(top_a, TOP_EXTENT)

    mask_sub = (np.abs(pts_sub[:, 0]) < current_fov) & (np.abs(pts_sub[:, 1]) < current_fov)
    vis_base = pts_sub[mask_sub]

    lim_top = current_fov * 1.5
    mask_top = (np.abs(pts_top[:, 0]) < lim_top) & (np.abs(pts_top[:, 1]) < lim_top)
    vis_top = pts_top[mask_top].dot(R.T)          # fancy indexing -> private copy

    G2_pts = get_hex_G(top_a, theta_deg)
    BZ2_pts = get_hex_bz(top_a, theta_deg)

    T_top_den = get_hex_density(top_a, X_den, Y_den, theta_deg)
    T_top_fft = get_hex_density(top_a, X_fft, Y_fft, theta_deg)

    # ---- interfacial phase state -----------------------------------------
    if glass:
        Ux_den, Uy_den = glass_displacement(X_den, Y_den, strain_coupling)
        Ux_pts, Uy_pts = glass_displacement(vis_top[:, 0], vis_top[:, 1], strain_coupling)

        # Evaluating the density at (X - U) shifts the pattern by +U, so the
        # discrete atoms must also move by +U. The original code moved them by
        # -U, which displaced the atom overlay opposite to the density map.
        T_top_den = get_hex_density(top_a, X_den - Ux_den, Y_den - Uy_den, theta_deg)
        vis_top = vis_top + np.column_stack([Ux_pts, Uy_pts])

        T_total = T_sub_den * T_top_den
        T_fft_engine = None            # Panel 3 blurs the two layers separately

    elif qc:
        weight = strain_coupling * 0.4
        mirrors = cfg["mirrors"]

        # Overlayer replicas: reflections across the substrate mirror lines
        T_top_qc = T_top_den.copy()
        T_top_fft_qc = T_top_fft.copy()
        for phi in mirrors:
            Xr, Yr = get_reflected_coords(X_den, Y_den, phi)
            T_top_qc += weight * get_hex_density(top_a, Xr, Yr, theta_deg)
            Xf, Yf = get_reflected_coords(X_fft, Y_fft, phi)
            T_top_fft_qc += weight * 6.0 * get_hex_density(top_a, Xf, Yf, theta_deg)

        # Substrate replicas: reflections across the overlayer mirror lines
        T_sub_qc = T_sub_den.copy()
        T_sub_fft_qc = T_sub_fft.copy()
        for phi in (theta_deg, theta_deg + 30.0):
            Xr, Yr = get_reflected_coords(X_den, Y_den, phi)
            Xf, Yf = get_reflected_coords(X_fft, Y_fft, phi)
            if cfg["geometry"] == "ortho":
                T_sub_qc += weight * get_ortho_density(cfg["sub_ax"], cfg["sub_ay"], Xr, Yr)
                T_sub_fft_qc += weight * 6.0 * get_ortho_density(cfg["sub_ax"], cfg["sub_ay"], Xf, Yf)
            else:
                T_sub_qc += weight * get_hex_density(cfg["sub_a"], Xr, Yr, 0.0)
                T_sub_fft_qc += weight * 6.0 * get_hex_density(cfg["sub_a"], Xf, Yf, 0.0)

        T_total = T_sub_qc * T_top_qc
        T_sub_fft, T_top_fft = T_sub_fft_qc, T_top_fft_qc
        T_fft_engine = T_sub_fft * T_top_fft

    else:
        T_total = T_sub_den * T_top_den
        T_fft_engine = T_sub_fft + T_top_fft

    # ---- registry scores ---------------------------------------------------
    if cfg["geometry"] == "ortho":
        dist_co, dist_ho, dist_br = ortho_registry_distances(vis_top, cfg["sub_ax"], cfg["sub_ay"])
        ho_scale = 1.0
    else:
        V_sub, invV_sub = hex_basis(cfg["sub_a"])
        dist_co, dist_ho, dist_br = hex_registry_distances(vis_top, V_sub, invV_sub)
        ho_scale = 1.3

    score_co = np.exp(-(dist_co / decay_L) ** 2)
    score_ho = np.exp(-(dist_ho / (decay_L * ho_scale)) ** 2)
    score_br = np.exp(-(dist_br / (decay_L * 0.8)) ** 2)

    return dict(
        vis_base=vis_base, vis_top=vis_top,
        dist_co=dist_co, score_co=score_co, score_ho=score_ho, score_br=score_br,
        T_total=T_total, T_fft_engine=T_fft_engine,
        T_sub_fft=T_sub_fft, T_top_fft=T_top_fft,
        G1_pts=G1_pts, G2_pts=G2_pts, BZ1_pts=BZ1_pts, BZ2_pts=BZ2_pts,
        glass=glass, qc=qc,
    )


def moire_vectors(G1_pts, G2_pts):
    """First-order moire reciprocal vectors and the matching real-space cell.

    Returns (g1_A, g1_B, g2_A, g2_B, L1, L2); L1/L2 are None if the moire cell is
    degenerate (commensurate or near-singular).
    """
    g1_A = G1_pts[np.argmax(G1_pts[:, 1])]
    g1_B = G1_pts[np.argmax(G1_pts[:, 0])]
    g2_A = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_A, axis=1))]
    g2_B = G2_pts[np.argmin(np.linalg.norm(G2_pts - g1_B, axis=1))]

    Q = np.column_stack([g1_A - g2_A, g1_B - g2_B])
    L1 = L2 = None
    if np.abs(np.linalg.det(Q)) > 1e-5:
        L_mat = 2 * np.pi * np.linalg.inv(Q.T)
        L1, L2 = L_mat[:, 0], L_mat[:, 1]
    return g1_A, g1_B, g2_A, g2_B, L1, L2


def stm_topography(vis_top, dist_co, current_fov, x_den, y_den):
    """Sum of registry-weighted Gaussian atomic protrusions on the density grid."""
    A_i = 1.0 + A_MODULATION * np.exp(-(dist_co ** 2) / (2 * XI_STACK ** 2))
    r_cut = 3.5 * SIGMA_ATOM
    dx = (2 * current_fov) / N_DEN
    Z = np.zeros((N_DEN, N_DEN))

    inside = (np.abs(vis_top[:, 0]) <= current_fov + r_cut) & \
             (np.abs(vis_top[:, 1]) <= current_fov + r_cut)
    for ax_pos, ay_pos, amp in zip(vis_top[inside, 0], vis_top[inside, 1], A_i[inside]):
        ix0 = max(0, int((ax_pos - r_cut + current_fov) / dx))
        ix1 = min(N_DEN, int((ax_pos + r_cut + current_fov) / dx) + 1)
        iy0 = max(0, int((ay_pos - r_cut + current_fov) / dx))
        iy1 = min(N_DEN, int((ay_pos + r_cut + current_fov) / dx) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        sub_X, sub_Y = np.meshgrid(x_den[ix0:ix1], y_den[iy0:iy1])
        dist2 = (sub_X - ax_pos) ** 2 + (sub_Y - ay_pos) ** 2
        Z[iy0:iy1, ix0:ix1] += amp * np.exp(-dist2 / (2 * SIGMA_ATOM ** 2))
    return Z


def relax_gap(T_total, relax_mode, w1, w2, user_zmin, user_zmax,
              k_elastic, k_vdw, iterations):
    """Map the kinematic density onto an interfacial gap z(r), then relax it."""
    span = np.max(T_total) - np.min(T_total)
    T_norm = (T_total - np.min(T_total)) / (span + 1e-10)
    Z_0 = user_zmin + T_norm * (user_zmax - user_zmin)

    if relax_mode == "Rigid Lattices (No Relaxation)":
        return Z_0.copy()

    if relax_mode == "Fast Proxy (Algebraic Shift)":
        compression = 0.2 * (w2 - w1) ** 2
        return np.clip(Z_0 - compression, a_min=2.0, a_max=None)

    # Continuum mechanics: gradient descent on elastic + vdW + image-charge terms
    Z_map = Z_0.copy()
    lr = 0.05 / (1.0 + k_elastic * 10)
    A_elec = 0.5 * (w2 - w1) ** 2
    for _ in range(iterations):
        F_elastic = k_elastic * ndimage.laplace(Z_map)
        F_vdw = -k_vdw * (Z_map - Z_0)
        F_elec = -A_elec / (Z_map ** 2)
        Z_map = np.clip(Z_map + lr * (F_elastic + F_vdw + F_elec), 1.5, 5.0)
    return Z_map


def safe_limits(data, clip_pct):
    """Percentile limits that never collapse to a zero-width colour range."""
    vmin = float(np.percentile(data, clip_pct))
    vmax = float(np.percentile(data, 100 - clip_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return vmin, vmax


def figure_size(show_fs_panel):
    return (24, 18) if show_fs_panel else (22, 6.8)


# ==========================================
# 5. MASTER UNIFIED PLOTTING FUNCTION
# ==========================================
def create_unified_plot(fig, system_mode, theta_deg, zoom_factor, q_max, k_max,
                        view_mode, mid_panel_mode, panel3_mode, den_cmap,
                        den_contrast, fft_scale, relax_mode, w1, w2,
                        user_zmin, user_zmax, k_elastic, k_vdw, eph_g0, eph_decay,
                        interfacial_state, strain_coupling, is_video_frame=False):
    cfg = SYSTEMS[system_mode]
    show_fs_panel = cfg["fs_panel"]
    X_fft, Y_fft, q_freq, window_2d = fft_grids()

    if fig is None:
        fig = Figure(figsize=figure_size(show_fs_panel), dpi=100)
        FigureCanvasAgg(fig)
    else:
        fig.clf()

    fig.patch.set_facecolor('#1a1a1a')
    if is_video_frame:
        fig.text(0.02, 0.96, f"Twist Angle: {theta_deg:.1f}°", color='#ffcc00',
                 fontsize=14, fontweight='bold', va='top', ha='left')

    if show_fs_panel:
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.8], wspace=0.22, hspace=0.32,
                              left=0.04, right=0.86, bottom=0.05, top=0.96)
        ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
        ax4 = fig.add_subplot(gs[1, :])
        axes = [ax1, ax2, ax3, ax4]
    else:
        gs = fig.add_gridspec(1, 3, wspace=0.22, left=0.04, right=0.86,
                              bottom=0.10, top=0.90)
        ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
        ax4 = None
        axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')

    current_fov = BASE_GRID * zoom_factor
    base_size = max(5, 50 / (zoom_factor ** 0.5))

    x_den = np.linspace(-current_fov, current_fov, N_DEN)
    y_den = np.linspace(-current_fov, current_fov, N_DEN)
    X_den, Y_den = np.meshgrid(x_den, y_den)

    fields = build_interface(cfg, theta_deg, current_fov, X_den, Y_den,
                             interfacial_state, strain_coupling)
    vis_base, vis_top = fields["vis_base"], fields["vis_top"]
    G1_pts, G2_pts = fields["G1_pts"], fields["G2_pts"]
    BZ1_pts, BZ2_pts = fields["BZ1_pts"], fields["BZ2_pts"]

    Z_stm_exact = stm_topography(vis_top, fields["dist_co"], current_fov, x_den, y_den)
    g1_A, g1_B, gm_A, gm_B, L1, L2 = moire_vectors(G1_pts, G2_pts)

    # ------------------------------------------
    # PANEL 1: REGISTRY DOMAINS
    # ------------------------------------------
    ax1.set_xlim(-current_fov, current_fov)
    ax1.set_ylim(-current_fov, current_fov)

    panel1_subtitle = f"{cfg['title']} | FOV: {zoom_factor}x"
    if L1 is not None:
        norm_l1, norm_l2 = np.linalg.norm(L1), np.linalg.norm(L2)
        if norm_l2 > 1e-5:
            panel1_subtitle += f" | $L_{{M1}}/L_{{M2}}$: {(norm_l1 / norm_l2):.2f}"

    ax1.set_title(f"Topology (Registry Map)\n{panel1_subtitle}", color='white', fontsize=13)
    ax1.set_xlabel(r"Distance ($\AA$)", color='white')
    ax1.set_ylabel(r"Distance ($\AA$)", color='white')

    show_all = (view_mode == 'Show All Registries')
    layers = []
    if show_all or view_mode in ('Coincident + Hollow', 'Coincident Only'):
        layers.append((fields["score_co"], (1.0, 0.2, 0.3)))
    if show_all or view_mode in ('Coincident + Hollow', 'Hollow Only'):
        layers.append((fields["score_ho"], (0.1, 0.6, 1.0)))
    if show_all or view_mode == 'Bridge Only':
        layers.append((fields["score_br"], (0.2, 0.8, 0.2)))

    if view_mode == 'Raw Lattices':
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='dodgerblue', alpha=0.5)
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=2, color='crimson', alpha=0.5)
    else:
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='gray', alpha=0.3, marker=',')
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=0.5, color='black', alpha=0.05, marker=',')
        for score, rgb in layers:
            mask = score > 0.05
            if not np.any(mask):
                continue
            colors = np.zeros((int(np.sum(mask)), 4))
            colors[:, 0], colors[:, 1], colors[:, 2] = rgb
            colors[:, 3] = score[mask]
            ax1.scatter(vis_top[mask, 0], vis_top[mask, 1],
                        s=base_size * score[mask], c=colors, edgecolors='none')

    if L1 is not None and np.linalg.norm(L1) < current_fov * 10 \
            and np.linalg.norm(L2) < current_fov * 10:
        cell_x = [0, L1[0], L1[0] + L2[0], L2[0], 0]
        cell_y = [0, L1[1], L1[1] + L2[1], L2[1], 0]
        ax1.plot(cell_x, cell_y, color='yellow', linestyle='--', linewidth=2.0,
                 alpha=0.9, zorder=5)
        for L in (L1, L2):
            ax1.annotate("", xy=L, xytext=(0, 0), zorder=6,
                         arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.5))

    # Invisible colorbar so Panel 1 keeps the same axes width as Panels 2 and 3
    sm1 = cm.ScalarMappable(cmap=mcolors.ListedColormap([(0, 0, 0, 0)]), norm=Normalize(0, 1))
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax1, shrink=0.36, pad=0.02, aspect=18, anchor=(0.0, 0.0))
    cbar1.ax.set_visible(False)

    # ------------------------------------------
    # SHARED Z-MAP: THE PHYSICS SOLVER ENGINE
    # ------------------------------------------
    Z_map = relax_gap(fields["T_total"], relax_mode, w1, w2, user_zmin, user_zmax,
                      k_elastic, k_vdw, iterations=120 if is_video_frame else 50)
    final_zmin, final_zmax = float(np.min(Z_map)), float(np.max(Z_map))
    gap_note = f"Relaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]"

    # ------------------------------------------
    # PANEL 2: MIDDLE PANEL ROUTING
    # ------------------------------------------
    if mid_panel_mode == 'Geometry (Kinematic Density)':
        data2, cmap_lbl = Z_stm_exact, 'Tunneling Density (a.u.)'
        title2, title_col = f"Registry-Modulated STM Topography\n{gap_note}", 'white'
    elif mid_panel_mode == 'Local Doping (Δn)':
        epsilon_0, e_charge = 8.854e-12, 1.602e-19
        delta_W = (w2 - w1) if (w2 - w1) != 0 else 1e-6
        C_geom = epsilon_0 / (Z_map * 1e-10)
        C_total = (C_geom * cfg["cq"]) / (C_geom + cfg["cq"])
        data2 = (C_total * delta_W) / e_charge / 1e4        # cm^-2
        cmap_lbl = r'Carrier Density $\Delta n$ (cm$^{-2}$)'
        title2 = "Local Doping in Layer 2: $\\Delta n$ (cm$^{-2}$)\n" + gap_note
        title_col = '#ffcc00'
    else:  # e-ph Coupling (g)
        data2 = eph_g0 * np.exp(-(Z_map - user_zmin) / eph_decay)
        cmap_lbl = r'Coupling Strength $g$ (meV)'
        title2 = "Evanescent e-ph Coupling: $g(\\mathbf{r})$\n" + gap_note
        title_col = '#00ffcc'

    vmin2, vmax2 = safe_limits(data2, den_contrast)
    im2 = ax2.imshow(data2, extent=[-current_fov, current_fov, -current_fov, current_fov],
                     origin='lower', cmap=den_cmap, vmin=vmin2, vmax=vmax2)
    ax2.set_title(title2, color=title_col, fontsize=13)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.36, pad=0.02, aspect=18, anchor=(0.0, 0.0))
    cbar2.ax.tick_params(colors='white', labelsize=8)
    cbar2.set_label(cmap_lbl, color='white', fontsize=9)

    ax2.set_xlim(-current_fov, current_fov)
    ax2.set_ylim(-current_fov, current_fov)
    ax2.set_xlabel(r"Distance ($\AA$)", color='white')

    # ------------------------------------------
    # PANEL 3: LEED FFT OR STM FFT ROUTING
    # ------------------------------------------
    lbl1_short, lbl2_short = cfg["sub_short"], cfg["top_short"]

    if panel3_mode == "FFT of STM Topography (Panel 2)":
        Z_win = (Z_stm_exact - np.mean(Z_stm_exact)) * stm_window(N_DEN)
        intensity_3 = np.abs(np.fft.fftshift(np.fft.fft2(Z_win)))

        q_stm = np.fft.fftshift(np.fft.fftfreq(N_DEN, d=(2 * current_fov) / N_DEN)) * 2 * np.pi
        q_lo, q_hi = q_stm[0], q_stm[-1]
        vmax_fft = max(np.max(intensity_3) * (fft_scale / 100.0), 1e-12)

        im3 = ax3.imshow(intensity_3, extent=[q_lo, q_hi, q_lo, q_hi], origin='lower',
                         cmap='afmhot', norm=Normalize(vmin=0, vmax=vmax_fft))
        title_3 = (f"FFT Amplitude of STM Topography\nTwist: {theta_deg}"
                   + r"$^\circ$" + f" | q-Zoom: {q_max} Å⁻¹")
        cbar_lbl = 'FFT Amplitude (a.u.)'
    else:
        q_lo, q_hi = q_freq[0], q_freq[-1]
        if fields["glass"]:
            sub_c = (fields["T_sub_fft"] - np.mean(fields["T_sub_fft"])) * window_2d
            top_c = (fields["T_top_fft"] - np.mean(fields["T_top_fft"])) * window_2d
            int_sub = np.abs(np.fft.fftshift(np.fft.fft2(sub_c))) ** 2
            int_top = np.abs(np.fft.fftshift(np.fft.fft2(top_c))) ** 2
            int_top = ndimage.gaussian_filter(int_top, sigma=0.5 + strain_coupling * 4.0)
            intensity_3 = int_sub + (int_top * 3.0) + 1e-10
        else:
            engine = fields["T_fft_engine"]
            centered = (engine - np.mean(engine)) * window_2d
            intensity_3 = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2 + 1e-10
            intensity_3 = ndimage.gaussian_filter(intensity_3, sigma=0.5)

        im3 = ax3.imshow(intensity_3, extent=[q_lo, q_hi, q_lo, q_hi], origin='lower',
                         cmap='viridis',
                         norm=LogNorm(vmin=np.max(intensity_3) * 1e-4, vmax=np.max(intensity_3)))
        title_3 = (f"Scattering (Simulated LEED)\nTwist: {theta_deg}"
                   + r"$^\circ$" + f" | q-Zoom: {q_max} Å⁻¹")
        cbar_lbl = 'Scattering Intensity (a.u.)'

    # --- overlays -----------------------------------------------------------
    ax3.plot(BZ1_pts[:, 0], BZ1_pts[:, 1], color='cyan', linestyle=':', lw=1.5, alpha=0.8, zorder=2)
    ax3.plot(BZ2_pts[:, 0], BZ2_pts[:, 1], color='red', linestyle=':', lw=1.5, alpha=0.8, zorder=2)
    ax3.scatter(G1_pts[:, 0], G1_pts[:, 1], facecolors='none', edgecolors='cyan',
                s=120, linewidths=1.5, marker='o', zorder=3)
    ax3.scatter(G2_pts[:, 0], G2_pts[:, 1], facecolors='none', edgecolors='red',
                s=120, linewidths=1.5, marker='s', zorder=3)

    for i, (v1, v2) in enumerate([(g1_A, gm_A), (g1_B, gm_B)]):
        ax3.annotate("", xy=v1, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="cyan", lw=1.5))
        ax3.annotate("", xy=v2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
        if np.linalg.norm(v2 - v1) > 1e-5:
            ax3.annotate("", xy=v2, xytext=v1,
                         arrowprops=dict(arrowstyle="-|>", color="yellow", lw=1.5, ls="--"))
            q_label = r"$\mathbf{q}_{M1}$" if i == 0 else r"$\mathbf{q}_{M2}$"
            ax3.annotate(q_label, xy=(v1 + v2) / 2, xytext=(4, 4),
                         textcoords="offset points", color="yellow", fontsize=10)

    legend_elements_3 = [
        mlines.Line2D([0], [0], color='none', marker='o', markeredgecolor='cyan', markersize=8, label=f'{lbl1_short} Peaks'),
        mlines.Line2D([0], [0], color='none', marker='s', markeredgecolor='red', markersize=8, label=f'{lbl2_short} Peaks'),
        mlines.Line2D([0], [0], color='cyan', linestyle=':', lw=1.5, label=f'{lbl1_short} 1st BZ'),
        mlines.Line2D([0], [0], color='red', linestyle=':', lw=1.5, label=f'{lbl2_short} 1st BZ'),
        mlines.Line2D([0], [0], color='cyan', linestyle='-', lw=1.5, label=r'Recip. Vec. $\mathbf{g}_1$'),
        mlines.Line2D([0], [0], color='red', linestyle='-', lw=1.5, label=r'Recip. Vec. $\mathbf{g}_2$'),
        mlines.Line2D([0], [0], color='yellow', linestyle='--', lw=1.5, label=r'Moiré Vecs. $\mathbf{q}_{M1}, \mathbf{q}_{M2}$'),
    ]

    if supports_interfacial_state(cfg) and "Rigid" not in interfacial_state and strain_coupling > 0:
        G_umklapp = np.concatenate([
            (G1_pts[:, None, :] + G2_pts[None, :, :]).reshape(-1, 2),
            (G1_pts[:, None, :] - G2_pts[None, :, :]).reshape(-1, 2),
        ])
        keep = (np.abs(G_umklapp[:, 0]) < q_max) & (np.abs(G_umklapp[:, 1]) < q_max)
        if np.any(keep):
            ax3.scatter(G_umklapp[keep, 0], G_umklapp[keep, 1], color='yellow', s=30,
                        marker='x', alpha=0.7, zorder=4)
            legend_elements_3.append(
                mlines.Line2D([0], [0], color='none', marker='x', markeredgecolor='yellow',
                              markersize=8, label='1st Order Umklapp'))

    if fields["qc"]:
        replica_styles = [('orange', 'D'), ('magenta', 'D')]
        for (phi, (col, mk)) in zip(cfg["mirrors"], replica_styles):
            # Reflecting the overlayer across a line at phi maps its orientation
            # theta -> 2*phi - theta.
            G_rep = get_hex_G(cfg["top_a"], 2 * phi - theta_deg)
            ax3.scatter(G_rep[:, 0], G_rep[:, 1], facecolors='none', edgecolors=col,
                        s=80, linewidths=1.0, marker=mk, zorder=3, alpha=0.8)
            legend_elements_3.append(
                mlines.Line2D([0], [0], color='none', marker=mk, markeredgecolor=col,
                              markersize=8, label=f'{lbl2_short} {phi:g}° Replica'))

        for (phi, col) in ((theta_deg, 'lime'), (theta_deg + 30.0, 'green')):
            c, s = np.cos(2 * np.radians(phi)), np.sin(2 * np.radians(phi))
            G_rep = np.column_stack([G1_pts[:, 0] * c + G1_pts[:, 1] * s,
                                     G1_pts[:, 0] * s - G1_pts[:, 1] * c])
            ax3.scatter(G_rep[:, 0], G_rep[:, 1], facecolors='none', edgecolors=col,
                        s=80, linewidths=1.0, marker='H', zorder=3, alpha=0.8)
            legend_elements_3.append(
                mlines.Line2D([0], [0], color='none', marker='H', markeredgecolor=col,
                              markersize=8, label=f'{lbl1_short} {phi:g}° Replica'))

    ax3.set_xlim(-q_max, q_max)
    ax3.set_ylim(-q_max, q_max)
    ax3.set_title(title_3, color='white', fontsize=13)
    ax3.set_xlabel(r"$q_x$ ($\AA^{-1}$)", color='white')

    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.36, pad=0.02, aspect=18, anchor=(0.0, 0.0))
    cbar3.ax.tick_params(colors='white', labelsize=8)
    cbar3.set_label(cbar_lbl, color='white', fontsize=9)

    ax3.legend(handles=legend_elements_3, loc='upper left', bbox_to_anchor=(1.04, 1.02),
               fontsize=8, framealpha=0.85, ncol=1, labelspacing=0.5,
               handletextpad=0.4, borderpad=0.4)

    # ------------------------------------------
    # PANEL 4: EXTENDED FERMI SURFACE MAP (FeSe ONLY)
    # ------------------------------------------
    if show_fs_panel and ax4 is not None:
        draw_fermi_surface_panel(ax4, cfg, theta_deg, k_max,
                                 interfacial_state, strain_coupling)

    return fig


def draw_fermi_surface_panel(ax4, cfg, theta_deg, k_max, interfacial_state, strain_coupling):
    """Mutual band folding of the FeSe M-pockets and the MoS2 K-pockets."""
    ax4.set_xlim(-k_max, k_max)
    ax4.set_ylim(-k_max, k_max)
    ax4.set_title("Fermi Surface Extended BZ (Mutual Band Folding)\n"
                  + interfacial_state.split(':')[0], color='white', fontsize=15)
    ax4.set_xlabel(r"$k_x$ ($\AA^{-1}$)", color='white')
    ax4.set_ylabel(r"$k_y$ ($\AA^{-1}$)", color='white')
    ax4.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax4.axvline(0, color='gray', lw=0.5, alpha=0.5)

    a_sub, a_top = cfg["sub_ax"], cfg["top_a"]
    r_sub, r_top = 0.175, 0.10

    q_sub = 2 * np.pi / a_sub
    G1_all, G1_shell1, G1_shell2 = [], [], []
    for n in range(-6, 7):
        for m in range(-6, 7):
            G = np.array([n * q_sub, m * q_sub])
            G1_all.append(G)
            d = np.linalg.norm(G)
            if 0.1 < d < q_sub * 1.1:
                G1_shell1.append(G)
            elif q_sub * 1.1 < d < q_sub * 1.5:
                G1_shell2.append(G)

    q_top = 4 * np.pi / (np.sqrt(3) * a_top)
    R_m = rotation_matrix(theta_deg)
    gA = np.array([0, q_top]).dot(R_m.T)
    gB = np.array([q_top * np.sqrt(3) / 2, -q_top * 0.5]).dot(R_m.T)

    G2_all, G2_shell1, G2_shell2 = [], [], []
    for n in range(-6, 7):
        for m in range(-6, 7):
            G = n * gA + m * gB
            G2_all.append(G)
            d = np.linalg.norm(G)
            if 0.1 < d < q_top * 1.1:
                G2_shell1.append(G)
            elif q_top * 1.1 < d < q_top * 1.8:
                G2_shell2.append(G)

    def unique_pockets(bases, G_list):
        pts = np.array([b + G for G in G_list for b in bases])
        _, idx = np.unique(np.round(pts, 4), axis=0, return_index=True)
        return pts[idx]

    M_bases = [np.array([q_sub / 2, q_sub / 2]), np.array([-q_sub / 2, q_sub / 2]),
               np.array([-q_sub / 2, -q_sub / 2]), np.array([q_sub / 2, -q_sub / 2])]
    M_pts = unique_pockets(M_bases, G1_all)

    K_mag = q_top / np.sqrt(3)
    K_bases = [np.array([K_mag * np.cos(a), K_mag * np.sin(a)]).dot(R_m.T)
               for a in np.radians(np.arange(0, 360, 60))]
    K_pts = unique_pockets(K_bases, G2_all)

    BZ_sub_base = get_ortho_bz(a_sub, a_sub, 0.0)
    BZ_top_base = get_hex_bz(a_top, theta_deg)

    def plot_bzs(G_list, BZ_base, color):
        for G in G_list:
            if abs(G[0]) > k_max + 1 or abs(G[1]) > k_max + 1:
                continue
            shifted = BZ_base + G
            ax4.plot(shifted[:, 0], shifted[:, 1], color=color, ls='-', lw=1.5, alpha=0.5)

    plot_bzs(G1_all, BZ_sub_base, 'cyan')
    plot_bzs(G2_all, BZ_top_base, 'red')

    def plot_fs(centers, r, col, lw, ls, alpha):
        for pt in centers:
            if abs(pt[0]) > k_max + r or abs(pt[1]) > k_max + r:
                continue
            ax4.add_patch(Circle((pt[0], pt[1]), r, color=col, fill=False,
                                 lw=lw, ls=ls, alpha=alpha))

    plot_fs(M_pts, r_sub, 'cyan', 2.5, '-', 1.0)
    plot_fs(K_pts, r_top, 'red', 2.5, '-', 1.0)

    legend_elements_4 = [
        mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=10, lw=2.5, label=f'Primary {cfg["sub_short"]} FS'),
        mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=10, lw=2.5, label=f'Primary {cfg["top_short"]} FS'),
    ]

    if strain_coupling > 0 and "Rigid" not in interfacial_state:
        is_glass = "Glass" in interfacial_state
        alpha1 = 0.6 if is_glass else 0.8
        alpha2 = 0.3 if is_glass else 0.5
        lw1 = 3.0 if is_glass else 1.5
        lw2 = 3.0 if is_glass else 1.0

        plot_fs([m + g for m in M_pts for g in G2_shell1], r_sub, 'cyan', lw1, '--', alpha1 * strain_coupling)
        plot_fs([k + g for k in K_pts for g in G1_shell1], r_top, 'red', lw1, '--', alpha1 * strain_coupling)
        plot_fs([m + g for m in M_pts for g in G2_shell2], r_sub, 'cyan', lw2, ':', alpha2 * strain_coupling)
        plot_fs([k + g for k in K_pts for g in G1_shell2], r_top, 'red', lw2, ':', alpha2 * strain_coupling)

        legend_elements_4.extend([
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=10, lw=lw1, ls='--', alpha=alpha1, label=f'{cfg["sub_short"]} Folded (1st Order)'),
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=10, lw=lw1, ls='--', alpha=alpha1, label=f'{cfg["top_short"]} Folded (1st Order)'),
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='cyan', markersize=8, lw=lw2, ls=':', alpha=alpha2, label=f'{cfg["sub_short"]} Folded (2nd Order)'),
            mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=8, lw=lw2, ls=':', alpha=alpha2, label=f'{cfg["top_short"]} Folded (2nd Order)'),
        ])

        if "Quasicrystal" in interfacial_state:
            mirrors = cfg["mirrors"]
            styles = [('orange', r_top * 1.30, 2.5), ('magenta', r_top * 1.08, 2.0)]
            for (phi, (col, rr, lw)) in zip(mirrors, styles):
                c, s = np.cos(2 * np.radians(phi)), np.sin(2 * np.radians(phi))
                K_rep = np.column_stack([K_pts[:, 0] * c + K_pts[:, 1] * s,
                                         K_pts[:, 0] * s - K_pts[:, 1] * c])
                plot_fs(K_rep, rr, col, lw, '--', 0.9)
                legend_elements_4.append(
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor=col,
                                  markersize=10, lw=lw, ls='--', alpha=0.9,
                                  label=f'{cfg["top_short"]} {phi:g}° Replica'))

            for (phi, col) in ((theta_deg, 'lime'), (theta_deg + 30.0, 'green')):
                c, s = np.cos(2 * np.radians(phi)), np.sin(2 * np.radians(phi))
                M_rep = np.column_stack([M_pts[:, 0] * c + M_pts[:, 1] * s,
                                         M_pts[:, 0] * s - M_pts[:, 1] * c])
                plot_fs(M_rep, r_sub * 1.08, col, 2.0, '--', 0.9)
                legend_elements_4.append(
                    mlines.Line2D([0], [0], marker='o', color='none', markeredgecolor=col,
                                  markersize=10, lw=2.0, ls='--', alpha=0.9,
                                  label=f'{cfg["sub_short"]} {phi:g}° Replica'))

    ax4.legend(handles=legend_elements_4, loc='upper right', bbox_to_anchor=(1.03, 1.02),
               fontsize=10, framealpha=0.9)


# ==========================================
# 6. STREAMLIT UI
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == "physics2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Password to access the UHV Dashboard:",
                      type="password", on_change=password_entered, key="password")
        return False
    if not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Password to access the UHV Dashboard:",
                      type="password", on_change=password_entered, key="password")
        st.error("Incorrect Password")
        return False
    return True


def render_video(system_mode, max_theta, plot_kwargs):
    max_t_int = int(max_theta)
    progress = st.progress(0, text=f"Rendering frame 1 of {max_t_int + 1}...")

    frames = []
    video_fig = Figure(figsize=figure_size(SYSTEMS[system_mode]["fs_panel"]), dpi=100)
    FigureCanvasAgg(video_fig)

    for ang in range(max_t_int + 1):
        progress.progress(int((ang / max_t_int) * 100),
                          text=f"Rendering frame {ang + 1} of {max_t_int + 1} (Twist: {ang}°)...")
        frame = create_unified_plot(video_fig, system_mode, float(ang),
                                    is_video_frame=True, **plot_kwargs)
        frame.canvas.draw()
        frames.append(np.asarray(frame.canvas.buffer_rgba())[:, :, :3].copy())

    progress.progress(100, text="Encoding MP4 Video...")
    out_path = Path(tempfile.gettempdir()) / "moire_twist_scan.mp4"
    imageio.mimsave(out_path, frames, fps=4, macro_block_size=None)
    progress.empty()
    return out_path.read_bytes()


def main():
    st.set_page_config(page_title="UHV-bonded Heterostructure Physics Dashboard",
                       layout="wide")
    st.markdown(
        "# UHV-bonded Heterostructure Physics Dashboard "
        f"<span style='font-size: 20px; font-weight: normal; color: #888888;'>{APP_VERSION}</span>",
        unsafe_allow_html=True)
    st.markdown("Explore the topology, geometry, scattering, local doping level and "
                "many-body interactions of 2D UHV-bonded heterostructures.")

    if not check_password():
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        system_mode = st.selectbox("System:", list(SYSTEMS.keys()))
        view_mode = st.selectbox("Topology View:",
                                 ['Show All Registries', 'Coincident + Hollow',
                                  'Coincident Only', 'Hollow Only', 'Bridge Only',
                                  'Raw Lattices'])

    cfg = SYSTEMS[system_mode]

    with col2:
        mid_panel_mode = st.radio("Middle Panel Metric:",
                                  ["Geometry (Kinematic Density)", "Local Doping (Δn)",
                                   "e-ph Coupling (g)"], horizontal=True)
        den_cmap = st.selectbox("Panel 2 Color:",
                                ['magma', 'viridis', 'plasma', 'cividis', 'gray',
                                 'bone', 'coolwarm', 'afmhot'])
        panel3_mode = st.radio("Panel 3 Mode:",
                               ["Scattering (Simulated LEED)",
                                "FFT of STM Topography (Panel 2)"])
        fft_scale = st.slider("FFT Intensity Scale (% Max):", 0.1, 100.0, 10.0, 0.5,
                              help="Lower value enhances weak Moiré FFT spots")

    with col3:
        max_theta = cfg["max_theta"]
        theta_deg = st.slider("Twist Angle (deg):", 0.0, max_theta, 0.0, 0.1)
        zoom_factor = st.slider("FOV Zoom (x):", 1.0, ZOOM_MAX, 1.0, 0.5)

        kcol3a, kcol3b = st.columns(2)
        with kcol3a:
            q_max = st.slider("q-Zoom (Å⁻¹):", 1.0, 8.0, 4.0, 0.5)
        with kcol3b:
            k_max = st.slider("k-Zoom (Panel 4) (Å⁻¹):", 2.0, 15.0, 6.0, 0.5)
        den_contrast = st.slider("Contrast Clip (%):", 0.0, 20.0, 0.0, 1.0)

    with st.expander("⚙️ Advanced Physics Parameters (Interfacial Mechanics & e-ph Coupling)",
                     expanded=True):

        st.markdown("**1. Interfacial Phase Engineering**")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            interfacial_state = st.selectbox("Interfacial Phase State:", [
                "1. Rigid vdW Gap (Non-interacting)",
                "2. Misfit Dislocation Glass (Experimental Blobs)",
                "3. Dodecagonal Quasicrystal (Theoretical CDW)",
            ])
        with pcol2:
            strain_coupling = st.slider("Interfacial Coupling Strength", 0.0, 1.0, 0.4, 0.1,
                                        help="Scales domain mosaicity (Mode 2) or resonant reflection (Mode 3).")
        if not supports_interfacial_state(cfg):
            st.caption("ℹ️ Modes 2 and 3 are only implemented for hex-on-square and "
                       "hex-on-rectangular systems; this system is rendered as a rigid vdW gap.")

        st.markdown("---")
        st.markdown("**2. Interfacial Mechanics & Doping Model**")

        relax_mode = st.selectbox("Mechanical Relaxation Model:", [
            "Rigid Lattices (No Relaxation)",
            "Fast Proxy (Algebraic Shift)",
            "Continuum Mechanics (PDE Solver)",
        ])

        base_zmin, base_zmax = cfg["z0"]

        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1:
            w1 = st.number_input("Layer 1 Work Function (eV)", value=4.2, step=0.1)
        with kcol2:
            w2 = st.number_input("Layer 2 Work Function (eV)", value=4.5, step=0.1)
        with kcol3:
            # Keys are system-scoped so the per-system defaults actually refresh
            # when the material system is changed.
            user_zmin = st.number_input(r"Base Unrelaxed Min Gap (Å)",
                                        value=float(base_zmin), step=0.1,
                                        key=f"zmin::{system_mode}")
        with kcol4:
            user_zmax = st.number_input(r"Base Unrelaxed Max Gap (Å)",
                                        value=float(base_zmax), step=0.1,
                                        key=f"zmax::{system_mode}")

        k_elastic, k_vdw = 0.0, 0.0
        if relax_mode == "Continuum Mechanics (PDE Solver)":
            st.markdown("*Continuum Tuning Parameters (Determines spatial smoothness vs structural pinning):*")
            scol1, scol2 = st.columns(2)
            with scol1:
                k_elastic = st.slider("Elastic Bending Rigidity (κ)", 0.0, 2.0, 0.5, 0.1)
            with scol2:
                k_vdw = st.slider(r"vdW Spring Stiffness ($k_{vdW}$)", 0.1, 5.0, 1.0, 0.1)

        st.markdown("---")
        st.markdown("**3. Local Electron-Phonon Coupling Model**")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            eph_g0 = st.number_input("Base Coupling at min gap (meV)", value=80.0, step=5.0)
        with ecol2:
            eph_decay = st.number_input(r"Evanescent Decay Length $\lambda$ (Å)",
                                        value=0.5, step=0.1)

    if user_zmax < user_zmin:
        st.warning("Max gap is below min gap — swapping the two for this render.")
        user_zmin, user_zmax = user_zmax, user_zmin
    if eph_decay <= 0:
        st.warning("Evanescent decay length must be positive; using 0.1 Å.")
        eph_decay = 0.1

    plot_kwargs = dict(
        zoom_factor=zoom_factor, q_max=q_max, k_max=k_max, view_mode=view_mode,
        mid_panel_mode=mid_panel_mode, panel3_mode=panel3_mode, den_cmap=den_cmap,
        den_contrast=den_contrast, fft_scale=fft_scale, relax_mode=relax_mode,
        w1=w1, w2=w2, user_zmin=user_zmin, user_zmax=user_zmax,
        k_elastic=k_elastic, k_vdw=k_vdw, eph_g0=eph_g0, eph_decay=eph_decay,
        interfacial_state=interfacial_state, strain_coupling=strain_coupling,
    )

    dashboard_placeholder = st.empty()
    with st.spinner("Re-calculating physics models and rendering panels... Please wait."):
        with dashboard_placeholder.container():
            fig = create_unified_plot(None, system_mode, theta_deg, **plot_kwargs)
            st.pyplot(fig)

    # --- VIDEO GENERATOR (CINEMATIC TOOLS) ---
    st.markdown("---")
    st.markdown("### 🎥 Cinematic Tools")

    max_t_int = int(cfg["max_theta"])
    if st.button(f"Generate Twist Angle Scan Video (0° to {max_t_int}°)"):
        st.session_state.video_sys_name = system_mode.replace("/", "_").split(" ")[0]
        st.session_state.video_bytes = render_video(system_mode, cfg["max_theta"], plot_kwargs)
        st.success("Video Generated Successfully!")

    if "video_bytes" in st.session_state:
        st.video(st.session_state.video_bytes, autoplay=True, loop=True)
        dl_name = st.session_state.get('video_sys_name', 'System')
        st.download_button(
            label="💾 Save Video to Computer",
            data=st.session_state.video_bytes,
            file_name=f"twist_scan_{dl_name}.mp4",
            mime="video/mp4",
        )


if __name__ == "__main__":
    main()
