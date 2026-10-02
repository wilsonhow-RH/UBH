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
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
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
N_FFT = 1024                      # kinematic-scattering grid
#   Nyquist q = pi*N_FFT/L_FFT = 8.04 A^-1, covering the whole q-Zoom slider
#   range; at N_FFT = 512 everything beyond 4.02 A^-1 was blank.
L_FFT = 400.0                     # side length of the scattering grid (Angstrom)

# Quantum capacitance of the overlayer, Cq = e^2 g_s g_v m* / (2 pi hbar^2)  [F/m^2].
# Monolayer MoS2 conduction band: g_s = 2, g_v = 2 (K valleys), m* ~ 0.45-0.48 m_e
# -> Cq ~ 0.60 F/m^2, consistent with the ~70 uF/cm^2 quoted for n-type 1L MoS2.
# Including the higher Q valleys (g_v = 6, m* ~ 0.6) would raise this to ~2.4 F/m^2,
# which only matters once E_F reaches the Q-valley edge (~0.1-0.25 eV above the CBM).
CQ_MOS2 = 0.60
CQ_GRAPHENE_DIRAC = 0.002   # vanishing DOS near the Dirac point; a placeholder scale

# Glassy displacement field (misfit-dislocation mode)
GLASS_N_WAVES = 40
GLASS_MOSAIC_FALLBACK = 120.0     # domain spacing used when no moire cell exists
GLASS_SEED = 42

# STM topography model
SIGMA_ATOM = 1.0                  # apparent atomic radius (Angstrom)
# (XI_STACK removed: the apparent-height contrast now uses the same registry
#  decay widths as Panel 1, which previously used 0.98 A while Panel 2 used a
#  separate hardcoded 0.6 A, so the two panels disagreed on the domain size.)
A_MODULATION = 1.1                # coincident-site brightness enhancement
# Relative apparent-height weights of the three stacking registries. The model
# previously used the coincident site alone, so Panel 2 showed a two-level
# (coincident / not) contrast while Panel 1 resolved three registries. The
# ordering below -- top site highest, bridge intermediate, hollow as the
# reference -- is a modelling choice, not a derived result; set bridge and
# hollow to 0 to recover the original behaviour.
REGISTRY_WEIGHTS_DEFAULT = (1.0, 0.4, 0.0)   # (coincident, bridge, hollow)


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
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MoS₂/SrTiO₃(110) (Hex-on-Rect)": dict(
        geometry="ortho", sub_ax=A_STO, sub_ay=A_STO_110, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on SrTiO$_3$(110)",
        label1=r"Layer 1 (SrTiO$_3$(110))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"SrTiO$_3$(110)", top_short=r"MoS$_2$",
        mirrors=(0.0, 90.0), z0=(3.1, 3.6), max_theta=90.0,
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MoS₂/FeSe (Hex-on-Square)": dict(
        geometry="ortho", sub_ax=A_FESE, sub_ay=A_FESE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on 1ML FeSe",
        label1=r"Layer 1 (FeSe)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"FeSe", top_short=r"MoS$_2$",
        mirrors=(0.0, 45.0), z0=(3.2, 3.6), max_theta=90.0,
        fs_panel=True, cq=CQ_MOS2,
    ),
    "MoS₂/Cu(100) (Hex-on-Square)": dict(
        geometry="ortho", sub_ax=A_CU_NN, sub_ay=A_CU_NN, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Cu(100)",
        label1=r"Layer 1 (Cu(100))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Cu(100)", top_short=r"MoS$_2$",
        mirrors=(0.0, 45.0), z0=(2.8, 3.4), max_theta=90.0,
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MoS₂/Cu(110) (Hex-on-Rect)": dict(
        geometry="ortho", sub_ax=A_CU_FCC, sub_ay=A_CU_NN, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Cu(110)",
        label1=r"Layer 1 (Cu(110))", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Cu(110)", top_short=r"MoS$_2$",
        mirrors=(0.0, 90.0), z0=(2.8, 3.4), max_theta=90.0,
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MoS₂/Bi₂Se₃ (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_BISE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on 6QL Bi$_2$Se$_3$",
        label1=r"Layer 1 (Bi$_2$Se$_3$)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Bi$_2$Se$_3$", top_short=r"MoS$_2$",
        mirrors=(), z0=(3.2, 3.6), max_theta=60.0,
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MoS₂/Graphene (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_GRAPHENE, top_a=A_MOS2,
        title=r"1ML MoS$_2$ on Graphene",
        label1=r"Layer 1 (Graphene)", label2=r"Layer 2 (MoS$_2$)",
        sub_short=r"Graphene", top_short=r"MoS$_2$",
        mirrors=(), z0=(3.3, 3.6), max_theta=60.0,
        fs_panel=False, cq=CQ_MOS2,
    ),
    "MATBG (Hex-on-Hex)": dict(
        geometry="hex", sub_a=A_GRAPHENE, top_a=A_GRAPHENE,
        title="Magic-Angle Twisted Bilayer Graphene",
        label1="Layer 1 (Graphene)", label2="Layer 2 (Rotated)",
        sub_short=r"Graphene", top_short="Rotated",
        mirrors=(), z0=(3.35, 3.6), max_theta=60.0,
        fs_panel=False, cq=CQ_GRAPHENE_DIRAC,
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


def primitive_pair(G_pts):
    """Two independent primitive reciprocal vectors out of a first-shell G star.

    Rows 0 and 2 are independent for both get_ortho_G (qx,0 / 0,qy) and
    get_hex_G (0,q / q*sqrt(3)/2,-q/2).
    """
    return G_pts[0], G_pts[2]


def reflect_points(pts, phi_deg):
    """Reflect an (N, 2) array across a line through the origin at angle phi."""
    c, s = np.cos(2 * np.radians(phi_deg)), np.sin(2 * np.radians(phi_deg))
    return np.column_stack([pts[:, 0] * c + pts[:, 1] * s,
                            pts[:, 0] * s - pts[:, 1] * c])


def umklapp_lattice(G1_pts, G2_pts, order, q_limit):
    """Integer combinations n*b1 + m*b2 + p*c1 + r*c2 of the two reciprocal bases.

    The STM topography is the overlayer lattice multiplied by a registry envelope
    with the moire periodicity, so its Fourier support is this whole module, not
    just the first-order difference set G1 +/- G2.
    """
    q, _ = umklapp_indexed(G1_pts, G2_pts, order, q_limit)
    return q


def umklapp_indexed(G1_pts, G2_pts, order, q_limit):
    """Same lattice, but also returning the (n, m, p, r) index of every point.

    Returns (positions (N,2), indices (N,4)). Duplicates in position are NOT
    removed here — classification needs to see every index vector that lands on
    a given spot in order to decide whether the assignment is unique.
    """
    b1, b2 = primitive_pair(G1_pts)
    c1, c2 = primitive_pair(G2_pts)
    rng = np.arange(-order, order + 1)
    n, m, p, r = (a.ravel() for a in np.meshgrid(rng, rng, rng, rng, indexing='ij'))
    keep = (np.abs(n) + np.abs(m) + np.abs(p) + np.abs(r)) <= order
    n, m, p, r = n[keep], m[keep], p[keep], r[keep]
    P = (n[:, None] * b1 + m[:, None] * b2 + p[:, None] * c1 + r[:, None] * c2)
    inside = (np.abs(P[:, 0]) <= q_limit) & (np.abs(P[:, 1]) <= q_limit)
    return P[inside], np.column_stack([n, m, p, r])[inside]


# Symmetry origin of a diffraction spot, in the order used for tie-breaking.
UMKLAPP_CLASSES = ('substrate', 'overlayer', 'replica', 'mixed')


def classify_umklapp(G1_pts, top_stars, order, q_limit, tol):
    """Label every predicted spot by the symmetry operation that generates it.

    Each candidate is an index vector (s; n, m, p, r): sublattice s of the
    overlayer (s = 0 primary, s > 0 a mirror replica), n,m on the substrate
    reciprocal basis, p,r on that sublattice's basis. Its class is

        substrate  p = r = 0            pure substrate Bragg / harmonic
        overlayer  n = m = 0, s = 0     pure overlayer Bragg / harmonic
        replica    n = m = 0, s > 0     pure mirror-replica Bragg / harmonic
        mixed      both nonzero         genuine interlayer umklapp

    Because the module generated by two incommensurate lattices is dense, a
    position alone does not determine its index vector: at high enough order
    some combination lands arbitrarily close to any q. The assignment is
    therefore made canonical by (i) a resolution tolerance `tol`, which should
    be the reciprocal-space peak width 2*pi/L of the analysis window, and
    (ii) a minimal-order rule: among all index vectors falling inside one
    tolerance ball, the one of lowest total order |n|+|m|+|p|+|r| wins, ties
    broken toward the primary sublattice and toward fewer substrate indices.
    A spot whose lowest-order candidates disagree on class is flagged
    `ambiguous`: it cannot be assigned from its position alone, only from its
    intensity or from how it transforms under the mirror operations.

    Returns (positions (M,2), class indices (M,), total orders (M,),
             ambiguous flags (M,)).
    """
    cand_q, cand_cls, cand_ord, cand_rank = [], [], [], []
    for s_idx, star in enumerate(top_stars):
        P, I = umklapp_indexed(G1_pts, star, order, q_limit)
        if len(P) == 0:
            continue
        n_sub = np.abs(I[:, 0]) + np.abs(I[:, 1])
        n_top = np.abs(I[:, 2]) + np.abs(I[:, 3])
        total = n_sub + n_top
        alive = total > 0
        cls = np.where(
            n_top == 0, 0,
            np.where(n_sub == 0, 1 if s_idx == 0 else 2, 3))
        cand_q.append(P[alive])
        cand_cls.append(cls[alive])
        cand_ord.append(total[alive])
        # tie-break key: lowest order, then primary sublattice, then fewer substrate indices
        cand_rank.append(np.column_stack([total[alive],
                                          np.full(alive.sum(), s_idx),
                                          n_sub[alive]]))
    if not cand_q:
        empty = np.zeros((0, 2))
        return empty, np.zeros(0, int), np.zeros(0, int), np.zeros(0, bool)

    Q = np.vstack(cand_q)
    C = np.concatenate(cand_cls)
    O = np.concatenate(cand_ord)
    K = np.vstack(cand_rank)

    # group candidates that fall within one tolerance ball, lowest order first
    srt = np.lexsort((K[:, 2], K[:, 1], K[:, 0]))
    Q, C, O, K = Q[srt], C[srt], O[srt], K[srt]

    reps, cls_out, ord_out, amb_out = [], [], [], []
    taken = np.zeros(len(Q), bool)
    for i in range(len(Q)):
        if taken[i]:
            continue
        near = (~taken) & (np.linalg.norm(Q - Q[i], axis=1) <= tol)
        taken |= near
        grp_cls, grp_ord = C[near], O[near]
        best = grp_ord.min()
        amb_out.append(bool(len(np.unique(grp_cls[grp_ord == best])) > 1))
        reps.append(Q[i])
        cls_out.append(int(C[i]))
        ord_out.append(int(O[i]))
    return (np.array(reps), np.array(cls_out, int),
            np.array(ord_out, int), np.array(amb_out, bool))


def orbit_classify(q_pts, intensity_of, mirrors, tol_q, tol_I=0.35):
    """Index-free classification of diffraction spots by point-group orbit.

    Unlike `classify_umklapp`, which needs the index vectors and therefore only
    works on a model, this asks a question that can be put to measured data:
    for each spot q and each mirror line of the substrate, is sigma(q) also a
    spot, and of comparable intensity?

    The physical content is that the mirror replicas of the overlayer are, by
    construction, the images of the primary Bragg set under the substrate
    mirrors, so replica-origin spots pair up with primary-origin spots under
    sigma. Mixed umklapp spots map onto other mixed spots of the same order
    instead, and spots belonging to the common point group map onto themselves.

    Returns an integer label per spot:
        2  invariant   -- sigma(q) is q itself (q lies on a mirror line)
        1  paired      -- sigma(q) is a different spot of comparable intensity
        0  unpaired    -- sigma(q) has no partner: the spot breaks that mirror

    `intensity_of` is a callable mapping an (N,2) array of q to intensities, so
    the same routine works on a simulated map or on an experimental FFT.
    """
    q_pts = np.asarray(q_pts, dtype=float).reshape(-1, 2)
    if len(q_pts) == 0 or not mirrors:
        return np.zeros(len(q_pts), int)
    I_self = np.asarray(intensity_of(q_pts), dtype=float)
    I_ref = np.max(I_self) if np.max(I_self) > 0 else 1.0

    labels = np.zeros(len(q_pts), int)
    for phi in mirrors:
        img = reflect_points(q_pts, phi)
        on_line = np.linalg.norm(img - q_pts, axis=1) <= tol_q
        I_img = np.asarray(intensity_of(img), dtype=float)
        # comparable brightness, measured on the log scale so weak spots are
        # compared fairly against weak spots
        ratio = (I_img + 1e-30) / (I_self + 1e-30)
        comparable = (np.abs(np.log10(ratio)) < -np.log10(tol_I)) & (I_img > 1e-3 * I_ref)
        labels = np.maximum(labels, np.where(on_line, 2, np.where(comparable, 1, 0)))
    return labels


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
def _glass_waves(k0):
    rng = np.random.default_rng(GLASS_SEED)
    ks = rng.normal(k0, k0 * 0.2, GLASS_N_WAVES)
    thetas = rng.uniform(0, 2 * np.pi, GLASS_N_WAVES)
    phases = rng.uniform(0, 2 * np.pi, GLASS_N_WAVES)
    return ks * np.cos(thetas), ks * np.sin(thetas), phases


def _glass_raw(X, Y, k0, dilatational):
    """Random displacement field mixing a solenoidal and a curl-free component.

        u_sol = sum (-k_y, k_x) sin(k.r + phi)   divergence free: no area change
        u_dil = sum ( k_x, k_y) sin(k.r + phi)   curl free: pure local dilatation

    The original model used u_sol alone, so div(u) vanished identically and the
    field carried no misfit strain at all -- but a misfit-dislocation network is
    precisely a partition of the interface into locally dilated or compressed
    commensurate domains. `dilatational` is the weight of the curl-free part:
    0 reproduces the old behaviour, 1 gives a purely dilatational field.
    """
    kx, ky, ph = _glass_waves(k0)
    f = float(np.clip(dilatational, 0.0, 1.0))
    Ux = np.zeros_like(X, dtype=float)
    Uy = np.zeros_like(Y, dtype=float)
    for i in range(GLASS_N_WAVES):
        wave = np.sin(X * kx[i] + Y * ky[i] + ph[i])
        Ux += (-ky[i] * (1.0 - f) + kx[i] * f) * wave
        Uy += (kx[i] * (1.0 - f) + ky[i] * f) * wave
    return Ux, Uy


@st.cache_resource(show_spinner=False)
def _glass_norm(k0, dilatational):
    """One fixed normalisation constant, shared by grid- and point-sampled calls.

    Normalising each call by the maximum over its own sample set made the
    amplitude depend on the sampling and on the zoom level, so the atom overlay
    and the density map were displaced by different amounts.
    """
    g = np.linspace(-FOV_MAX, FOV_MAX, 256)
    X, Y = np.meshgrid(g, g)
    Ux, Uy = _glass_raw(X, Y, k0, dilatational)
    return float(np.max(np.hypot(Ux, Uy))) + 1e-10


def glass_displacement(X, Y, coupling, mosaic_L=GLASS_MOSAIC_FALLBACK,
                       dilatational=0.5):
    """Random displacement field, peak amplitude ~ 1.5 * coupling (Angstrom).

    mosaic_L is the dominant domain-wall spacing. It was hardcoded at 120 A
    regardless of twist angle, mismatch or material; the caller now passes the
    moire period, which is what actually sets the domain size in a relaxed
    incommensurate contact.
    """
    if coupling <= 0:
        return np.zeros_like(X, dtype=float), np.zeros_like(Y, dtype=float)
    k0 = 2 * np.pi / max(float(mosaic_L), 5.0)
    Ux, Uy = _glass_raw(X, Y, k0, dilatational)
    amp = coupling * 1.5 / _glass_norm(k0, dilatational)
    return Ux * amp, Uy * amp


# ==========================================
# 4. INTERFACE FIELD BUILDER
# ==========================================
SCATTER_MODELS = ("Kinematic (single scattering)", "Include double diffraction")


def combine_layers(T_sub_fft, T_top_fft, scatter_model):
    """Build the field whose transform Panel 3 squares.

    Summing the two layer densities is single kinematic scattering: each layer
    diffracts independently and the pattern is the union of the two Bragg sets.
    Multiplying them represents an electron diffracted by one layer and again by
    the other, which is what generates the umklapp satellites at n*G1 + m*G2.

    Real LEED is multiple-scattering dominated and double-diffraction spots are
    routine for any incommensurate overlayer, so this is now its own control.
    Previously the product was switched on only by selecting an interfacial phase
    state, which made an unrelated setting gate whether double diffraction
    existed at all.
    """
    if scatter_model == SCATTER_MODELS[0]:
        return T_sub_fft + T_top_fft
    return T_sub_fft * T_top_fft


def build_interface(cfg, theta_deg, current_fov, X_den, Y_den,
                    interfacial_state, strain_coupling,
                    scatter_model='Include double diffraction',
                    mosaic_L=0.0, dilatational=0.5,
                    registry_weights=REGISTRY_WEIGHTS_DEFAULT):
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
    # Domain spacing defaults to the moire period rather than a hardcoded 120 A.
    if mosaic_L and mosaic_L > 0:
        glass_L = float(mosaic_L)
    else:
        *_, L1m, L2m = moire_vectors(G1_pts, G2_pts)
        glass_L = (GLASS_MOSAIC_FALLBACK if L1m is None else
                   float(np.clip(0.5 * (np.linalg.norm(L1m) + np.linalg.norm(L2m)),
                                 10.0, 1000.0)))

    if glass:
        Ux_den, Uy_den = glass_displacement(X_den, Y_den, strain_coupling,
                                            glass_L, dilatational)
        Ux_pts, Uy_pts = glass_displacement(vis_top[:, 0], vis_top[:, 1],
                                            strain_coupling, glass_L, dilatational)

        # Evaluating the density at (X - U) shifts the pattern by +U, so the
        # discrete atoms must also move by +U. The original code moved them by
        # -U, which displaced the atom overlay opposite to the density map.
        T_top_den = get_hex_density(top_a, X_den - Ux_den, Y_den - Uy_den, theta_deg)
        vis_top = vis_top + np.column_stack([Ux_pts, Uy_pts])

        # Scattering from the ACTUAL displaced lattice. The previous model blurred
        # the intensity of the undisplaced overlayer with a width proportional to
        # the coupling and scaled it by an arbitrary x3, which neither suppresses
        # Bragg peaks by the correct G-dependent Debye-Waller factor nor conserves
        # the weight it removes. Transforming the displaced density does both
        # automatically: exp(-<(G.u)^2>) falls off with |G|, and the missing weight
        # reappears as diffuse scattering.
        Ux_f, Uy_f = glass_displacement(X_fft, Y_fft, strain_coupling,
                                        glass_L, dilatational)
        T_top_fft = get_hex_density(top_a, X_fft - Ux_f, Y_fft - Uy_f, theta_deg)

        T_total = T_sub_den * T_top_den
        T_fft_engine = combine_layers(T_sub_fft, T_top_fft, scatter_model)

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
        T_fft_engine = combine_layers(T_sub_fft, T_top_fft, scatter_model)

    else:
        T_total = T_sub_den * T_top_den
        T_fft_engine = combine_layers(T_sub_fft, T_top_fft, scatter_model)

    # ---- registry scores ---------------------------------------------------
    if cfg["geometry"] == "ortho":
        def registry(pts):
            return ortho_registry_distances(pts, cfg["sub_ax"], cfg["sub_ay"])
        ho_scale = 1.0
    else:
        V_sub, invV_sub = hex_basis(cfg["sub_a"])

        def registry(pts):
            return hex_registry_distances(pts, V_sub, invV_sub)
        ho_scale = 1.3

    dist_co, dist_ho, dist_br = registry(vis_top)
    decay_widths = (decay_L, decay_L * ho_scale, decay_L * 0.8)

    score_co = np.exp(-(dist_co / decay_widths[0]) ** 2)
    score_ho = np.exp(-(dist_ho / decay_widths[1]) ** 2)
    score_br = np.exp(-(dist_br / decay_widths[2]) ** 2)

    # ---- overlayer sublattices feeding the STM topography ------------------
    # In the quasicrystal state the density engine adds mirror-reflected copies
    # of the overlayer with weight `weight`. The atom-resolved topography now
    # carries the same replicas, so Panel 2 and the FFT of Panel 2 describe the
    # same structure the LEED engine and the replica overlays describe.
    def height_score(pts):
        """Registry-resolved apparent-height modulation, in [0, ~1]."""
        d_co, d_ho, d_br = registry(pts)
        w_co, w_br, w_ho = registry_weights
        return (w_co * np.exp(-(d_co / decay_widths[0]) ** 2)
                + w_br * np.exp(-(d_br / decay_widths[2]) ** 2)
                + w_ho * np.exp(-(d_ho / decay_widths[1]) ** 2))

    top_layers = [(vis_top, height_score(vis_top), 1.0)]
    if qc:
        w_rep = strain_coupling * 0.4
        for phi in cfg["mirrors"]:
            rep = reflect_points(vis_top, phi)
            top_layers.append((rep, height_score(rep), w_rep))

    return dict(
        vis_base=vis_base, vis_top=vis_top, top_layers=top_layers,
        dist_co=dist_co, score_co=score_co, score_ho=score_ho, score_br=score_br,
        decay_widths=decay_widths,
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


def stm_topography(top_layers, current_fov, x_den, y_den):
    """Sum of registry-weighted Gaussian atomic protrusions on the density grid.

    `top_layers` is a list of (positions, registry height score, weight); the
    quasicrystal state supplies mirror-replica sublattices as extra entries.
    """
    r_cut = 3.5 * SIGMA_ATOM
    dx = (2 * current_fov) / N_DEN
    Z = np.zeros((N_DEN, N_DEN))

    for pts, score, weight in top_layers:
        if weight <= 0 or len(pts) == 0:
            continue
        A_i = weight * (1.0 + A_MODULATION * score)
        _accumulate_atoms(Z, pts, A_i, current_fov, r_cut, dx, x_den, y_den)
    return Z


def _accumulate_atoms(Z, pts, A_i, current_fov, r_cut, dx, x_den, y_den):
    inside = (np.abs(pts[:, 0]) <= current_fov + r_cut) & \
             (np.abs(pts[:, 1]) <= current_fov + r_cut)
    for ax_pos, ay_pos, amp in zip(pts[inside, 0], pts[inside, 1], A_i[inside]):
        ix0 = max(0, int((ax_pos - r_cut + current_fov) / dx))
        ix1 = min(N_DEN, int((ax_pos + r_cut + current_fov) / dx) + 1)
        iy0 = max(0, int((ay_pos - r_cut + current_fov) / dx))
        iy1 = min(N_DEN, int((ay_pos + r_cut + current_fov) / dx) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        sub_X, sub_Y = np.meshgrid(x_den[ix0:ix1], y_den[iy0:iy1])
        dist2 = (sub_X - ax_pos) ** 2 + (sub_Y - ay_pos) ** 2
        Z[iy0:iy1, ix0:ix1] += amp * np.exp(-dist2 / (2 * SIGMA_ATOM ** 2))


def relax_gap(T_total, relax_mode, w1, w2, user_zmin, user_zmax,
              smooth_len, k_vdw, dx, iterations=12,
              elastic_operator='Membrane tension (∇²)'):
    """Map the kinematic density onto an interfacial gap z(r), then relax it.

    The continuum model is the steady state of

        0 = kappa grad^2 z - k_vdw (z - z_0) - A_elec / z^2 ,

    with kappa = k_vdw * smooth_len^2. Writing the electrostatic term as a source
    S(z) = A_elec / z^2 and transforming, the linear part inverts exactly:

        z(k) = [ z_0(k) - S(k)/k_vdw ] / ( 1 + smooth_len^2 k^2 )

    i.e. a Lorentzian low-pass of half-power wavevector 1/smooth_len. Only the
    nonlinear source is iterated, and it converges in a few passes because the
    electrostatic pull is a weak perturbation on the vdW template.

    This replaces the previous explicit Euler loop over `ndimage.laplace`, which
    was a PIXEL Laplacian: kappa was effectively scaled by dx^2, so the same
    physical system relaxed differently at different zoom levels, and the step
    size sat close to the explicit stability limit once dx^2 was divided out.
    Here k is in physical A^-1, so the result no longer depends on the field of
    view, the scheme is unconditionally stable, and the linear part is exact
    rather than partially converged.
    """
    span = np.max(T_total) - np.min(T_total)
    T_norm = (T_total - np.min(T_total)) / (span + 1e-10)
    Z_0 = user_zmin + T_norm * (user_zmax - user_zmin)

    if relax_mode == "Rigid Lattices (No Relaxation)":
        return Z_0.copy()

    if relax_mode == "Fast Proxy (Algebraic Shift)":
        # NOTE: a rigid shift, so it adds no spatial structure -- the doping and
        # e-ph maps differ from rigid mode by a constant only.
        compression = 0.2 * (w2 - w1) ** 2
        return np.clip(Z_0 - compression, a_min=2.0, a_max=None)

    # meV/A^3; k_vdw is in meV/A^4, so (A_elec/k_vdw)/z^2 is a length.
    A_elec = K_ELEC * (w2 - w1) ** 2
    z_hi = max(5.0, user_zmax + 1.0)
    if smooth_len <= 0 and A_elec == 0:
        return np.clip(Z_0, 1.5, z_hi)

    # The spectral solve imposes periodic boundaries, which the field of view does
    # not obey, so the frame edge is otherwise smoothed against the opposite edge.
    # Measured effect of reflect-padding by 3 smoothing lengths (3x zoom, theta=17.2):
    # 21.9 mA at l = 1 A, 5.1 mA at l = 3 A, 0.2 mA at l = 50 A -- it matters most at
    # SMALL l, where sharp atomic corrugation mismatches across the wrap, and the
    # default-l error is comparable to the interior signal (sigma ~ 3.3 mA). Cost is
    # ~0.1 s. NOTE: the long-wavelength structure that looks like an edge effect in
    # the rendered map is usually not one -- at twists where the moire period is
    # comparable to the field of view, the frame simply contains ~1 moire cell.
    n = Z_0.shape[0]
    pad = int(np.clip(np.ceil(3.0 * smooth_len / dx), 8, n // 2))
    Z_0p = np.pad(Z_0, pad, mode='reflect')
    m = Z_0p.shape[0]

    k1 = 2 * np.pi * np.fft.fftfreq(m, d=dx)
    K2 = k1[:, None] ** 2 + k1[None, :] ** 2
    if elastic_operator.startswith("Bending"):
        # True bending rigidity: the force is -kappa grad^4 z, so the response is
        # 1/(1 + l^4 k^4). This is the physically correct operator for a monolayer;
        # the tension form 1/(1 + l^2 k^2) is kept because it is what the original
        # +kappa grad^2 z term corresponded to.
        lowpass = 1.0 / (1.0 + (smooth_len ** 4) * K2 ** 2)
    else:
        lowpass = 1.0 / (1.0 + (smooth_len ** 2) * K2)

    Zp = Z_0p.copy()
    for _ in range(int(iterations)):
        src = Z_0p - (A_elec / max(k_vdw, 1e-6)) / (Zp ** 2)
        Zp = np.clip(np.real(np.fft.ifft2(np.fft.fft2(src) * lowpass)), 1.5, z_hi)
    return Zp[pad:pad + n, pad:pad + n]


EPS0 = 8.854e-12          # F/m
E_CHARGE = 1.602e-19      # C

# Electrostatic pressure between two plates held at a fixed potential difference
# is eps0*dV^2/(2 z^2). Expressed as an energy per unit area in meV/A^2 with z in
# Angstrom, U(z) = -K_ELEC * dV^2 / z, so the force per area is -K_ELEC*dV^2/z^2
# with K_ELEC in meV/A. This replaces the previous coefficient 0.5*(w2-w1)^2,
# which had units of eV^2 and was used directly as a force -- an arbitrary scale.
K_ELEC = 62.42 * EPS0 / 2e-10     # = 2.764 meV/A per volt^2


def carrier_density(Z_map, delta_W, Cq, C_sc=0.0):
    """Charge transferred into layer 2, in cm^-2.

    Series chain: geometric gap capacitance eps0/z, the overlayer quantum
    capacitance Cq, and -- optionally -- a substrate space-charge capacitance
    C_sc. The last one matters because SrTiO3 is a semiconductor, not a metal: a
    depletion or accumulation layer forms and sits in series with the gap. C_sc = 0
    here means "omit it" (a metallic substrate), which is what the model assumed
    throughout. A proper treatment needs a Poisson solve in the substrate with a
    field-dependent permittivity; this is a lumped stand-in, not that.
    """
    C_geom = EPS0 / (Z_map * 1e-10)
    inv = 1.0 / C_geom + 1.0 / max(Cq, 1e-9)
    if C_sc and C_sc > 0:
        inv = inv + 1.0 / C_sc
    return (delta_W / inv) / E_CHARGE / 1e4


def safe_limits(data, clip_pct):
    """Percentile limits that never collapse to a zero-width colour range."""
    vmin = float(np.percentile(data, clip_pct))
    vmax = float(np.percentile(data, 100 - clip_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-12:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return vmin, vmax


def registry_legend(layers, boundary_mode, decay_widths, vis_top, current_fov):
    """Panel 1 legend; with boundaries on, annotate FWHM width and areal coverage."""
    fwhm = 2 * np.sqrt(np.log(2))
    strict = (np.abs(vis_top[:, 0]) <= current_fov) & (np.abs(vis_top[:, 1]) <= current_fov)
    n_total = max(1, int(np.sum(strict)))

    handles = []
    for i, name, score, rgb, _edge in layers:
        label = name
        if boundary_mode != "None":
            cov = np.sum(score[strict] >= 0.5) / n_total * 100
            label = f"{name} (W: {decay_widths[i] * fwhm:.1f}Å, Cov: {cov:.1f}%)"
        handles.append(mlines.Line2D([0], [0], marker='o', color='w', lw=0,
                                     markerfacecolor=rgb, markersize=9, label=label))
    return handles


def draw_domain_boundaries(ax, boundary_mode, vis_top, layers, current_fov, top_a):
    """Iso-contours at registry score 0.5, either atom-resolved or coarse-grained."""
    if len(vis_top) < 4 or not layers:
        return

    if boundary_mode == "Microscopic (Atomic)":
        try:
            triang = mtri.Triangulation(vis_top[:, 0], vis_top[:, 1])
        except (ValueError, RuntimeError):
            return
        for _i, _name, score, _rgb, edge in layers:
            if np.nanmax(score) <= 0.5 or np.nanmin(score) >= 0.5:
                continue
            ax.tricontour(triang, score, levels=[0.5], colors=edge,
                          linewidths=1.5, linestyles='solid')
        return

    # Mesoscopic: dilate the atom-sampled score onto a grid, smooth, contour at
    # the midpoint of the smoothed range so the envelope of the domains survives.
    pad_fov = current_fov * 1.2
    N_pad = int(N_DEN * 1.2)
    dx = (pad_fov * 2) / N_pad

    valid = (np.abs(vis_top[:, 0]) < pad_fov) & (np.abs(vis_top[:, 1]) < pad_fov)
    if not np.any(valid):
        return
    vt = vis_top[valid]
    ix = np.clip(np.round((vt[:, 0] + pad_fov) / dx).astype(int), 0, N_pad - 1)
    iy = np.clip(np.round((vt[:, 1] + pad_fov) / dx).astype(int), 0, N_pad - 1)

    radius = max(1, int(np.ceil((top_a * 1.5) / dx)))
    y_fp, x_fp = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    footprint = x_fp ** 2 + y_fp ** 2 <= radius ** 2      # isotropic, preserves symmetry

    pad = np.linspace(-pad_fov, pad_fov, N_pad)
    X_pad, Y_pad = np.meshgrid(pad, pad)

    for _i, _name, score, _rgb, edge in layers:
        grid_z = np.zeros((N_pad, N_pad))
        np.maximum.at(grid_z, (iy, ix), score[valid])
        grid_z = ndimage.maximum_filter(grid_z, footprint=footprint)
        grid_z = ndimage.gaussian_filter(grid_z, sigma=radius)
        z_min, z_max = float(np.min(grid_z)), float(np.max(grid_z))
        if (z_max - z_min) > 1e-5:
            ax.contour(X_pad, Y_pad, grid_z, levels=[z_min + (z_max - z_min) * 0.5],
                       colors=edge, linewidths=1.5, linestyles='solid')


def figure_size(show_fs_panel):
    return (24, 18) if show_fs_panel else (22, 6.8)


# ==========================================
# 5. MASTER UNIFIED PLOTTING FUNCTION
# ==========================================
def create_unified_plot(fig, system_mode, theta_deg, zoom_factor, q_max, k_max,
                        view_mode, boundary_mode, mid_panel_mode, panel3_mode, den_cmap,
                        den_contrast, fft_scale, relax_mode, w1, w2,
                        user_zmin, user_zmax, smooth_len, k_vdw, eph_g0, eph_q0,
                        interfacial_state, strain_coupling, umklapp_order=3,
                        layer2_cq=None, scatter_model='Include double diffraction',
                        mosaic_L=0.0, dilatational=0.5,
                        registry_weights=REGISTRY_WEIGHTS_DEFAULT,
                        substrate_csc=0.0, layer2_n0=0.0,
                        classify_mode='Index-based (model)',
                        elastic_operator='Membrane tension (∇²)',
                        is_video_frame=False):
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
                             interfacial_state, strain_coupling,
                             scatter_model, mosaic_L, dilatational,
                             registry_weights)
    vis_base, vis_top = fields["vis_base"], fields["vis_top"]
    G1_pts, G2_pts = fields["G1_pts"], fields["G2_pts"]
    BZ1_pts, BZ2_pts = fields["BZ1_pts"], fields["BZ2_pts"]

    Z_stm_exact = stm_topography(fields["top_layers"], current_fov, x_den, y_den)
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
    layers = []                                  # (index, name, score, fill rgb, edge colour)
    if show_all or view_mode in ('Coincident + Hollow', 'Coincident Only'):
        layers.append((0, 'Coincident', fields["score_co"], (1.0, 0.2, 0.3), '#ff6666'))
    if show_all or view_mode in ('Coincident + Hollow', 'Hollow Only'):
        layers.append((1, 'Hollow', fields["score_ho"], (0.1, 0.6, 1.0), '#66b3ff'))
    if show_all or view_mode == 'Bridge Only':
        layers.append((2, 'Bridge', fields["score_br"], (0.2, 0.8, 0.2), '#66ff66'))

    if view_mode == 'Raw Lattices':
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='dodgerblue', alpha=0.5)
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=2, color='crimson', alpha=0.5)
    else:
        ax1.scatter(vis_base[:, 0], vis_base[:, 1], s=2, color='gray', alpha=0.3, marker=',')
        ax1.scatter(vis_top[:, 0], vis_top[:, 1], s=0.5, color='black', alpha=0.05, marker=',')
        for _i, _name, score, rgb, _edge in layers:
            mask = score > 0.05
            if not np.any(mask):
                continue
            colors = np.zeros((int(np.sum(mask)), 4))
            colors[:, 0], colors[:, 1], colors[:, 2] = rgb
            colors[:, 3] = score[mask]
            ax1.scatter(vis_top[mask, 0], vis_top[mask, 1],
                        s=base_size * score[mask], c=colors, edgecolors='none')

        if boundary_mode != "None":
            draw_domain_boundaries(ax1, boundary_mode, vis_top, layers,
                                   current_fov, cfg["top_a"])

    if view_mode != 'Raw Lattices':
        ax1.legend(handles=registry_legend(layers, boundary_mode, fields["decay_widths"],
                                           vis_top, current_fov),
                   loc='upper right', fontsize=9, framealpha=0.8)

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
                      smooth_len, k_vdw, dx=(2 * current_fov) / N_DEN,
                      elastic_operator=elastic_operator)
    final_zmin, final_zmax = float(np.min(Z_map)), float(np.max(Z_map))
    Cq_eff = cfg["cq"] if layer2_cq is None else float(layer2_cq)
    delta_n_map = carrier_density(Z_map, (w2 - w1), Cq_eff, substrate_csc)
    gap_note = f"Relaxed Gap: [{final_zmin:.2f} Å, {final_zmax:.2f} Å]"

    # ------------------------------------------
    # PANEL 2: MIDDLE PANEL ROUTING
    # ------------------------------------------
    if mid_panel_mode == 'Geometry (Kinematic Density)':
        data2, cmap_lbl = Z_stm_exact, 'Tunneling Density (a.u.)'
        title2, title_col = f"Registry-Modulated STM Topography\n{gap_note}", 'white'
    elif mid_panel_mode == 'Local Doping (Δn)':
        # The old code substituted dW = 1e-6 when the work functions were equal,
        # only to avoid a degenerate colourbar; safe_limits() handles that now, so
        # equal work functions correctly give no charge transfer at all.
        data2 = delta_n_map
        cmap_lbl = r'Carrier Density $\Delta n$ (cm$^{-2}$)'
        title2 = "Local Doping in Layer 2: $\\Delta n$ (cm$^{-2}$)\n" + gap_note
        title_col = '#ffcc00'
    else:  # e-ph Coupling (g)
        # An interfacial polar phonon of in-plane wavevector q couples to the
        # overlayer as exp(-q z), so the decay length is 1/q0, not a free
        # constant. g0 is referenced to the actual minimum of the relaxed map,
        # so "coupling at the minimum gap" is true after relaxation as well.
        data2 = eph_g0 * np.exp(-eph_q0 * (Z_map - np.min(Z_map)))
        cmap_lbl = r'Coupling Strength $g$ (meV)'
        contrast = 100.0 * (1.0 - np.exp(-eph_q0 * (final_zmax - final_zmin)))
        title2 = (r"Evanescent e-ph Coupling: $g(\mathbf{r})$"
                  + f"\n$q_0$ = {eph_q0:.3f} " + r"$\AA^{-1}$"
                  + f" ($1/q_0$ = {1.0 / eph_q0:.0f} Å) | spatial contrast {contrast:.1f}%")
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
        engine = fields["T_fft_engine"]
        centered = (engine - np.mean(engine)) * window_2d
        intensity_3 = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2 + 1e-10
        intensity_3 = ndimage.gaussian_filter(intensity_3, sigma=0.5)

        im3 = ax3.imshow(intensity_3, extent=[q_lo, q_hi, q_lo, q_hi], origin='lower',
                         cmap='viridis',
                         norm=LogNorm(vmin=np.max(intensity_3) * 1e-4, vmax=np.max(intensity_3)))
        sm_tag = "kinematic" if scatter_model == SCATTER_MODELS[0] else "with double diffraction"
        title_3 = (f"Scattering (Simulated LEED, {sm_tag})\nTwist: {theta_deg}"
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

    # The STM topography is always registry-modulated, so its Fourier transform
    # always carries the full umklapp module. The LEED engine only mixes the two
    # lattices multiplicatively once an interfacial state is switched on.
    stm_fft_mode = (panel3_mode == "FFT of STM Topography (Panel 2)")
    show_umklapp = stm_fft_mode or (supports_interfacial_state(cfg)
                                    and "Rigid" not in interfacial_state
                                    and strain_coupling > 0)
    if show_umklapp and umklapp_order >= 1:
        # Each overlayer sublattice present in the model gets its own satellite
        # set, so in the quasicrystal state the replicas carry satellites too.
        top_stars = [G2_pts]
        if fields["qc"]:
            top_stars += [get_hex_G(cfg["top_a"], 2 * phi - theta_deg) for phi in cfg["mirrors"]]

        # Resolution of the analysis window: spots closer together than this
        # cannot be told apart, so that is the tolerance the classifier uses.
        window_L = (2 * current_fov) if stm_fft_mode else L_FFT
        tol_q = 2 * np.pi / window_L

        Gq, Gcls, Gord, Gamb = classify_umklapp(G1_pts, top_stars, int(umklapp_order),
                                                q_max, tol_q)

        if classify_mode.startswith("Point-group") and len(Gq) and cfg["mirrors"]:
            # Re-label from the intensity map itself rather than from indices, so
            # the same test can be applied to measured data.
            q_axis = np.linspace(q_lo, q_hi, intensity_3.shape[0])

            def intensity_of(pts):
                ix = np.clip(np.searchsorted(q_axis, pts[:, 0]), 0, len(q_axis) - 1)
                iy = np.clip(np.searchsorted(q_axis, pts[:, 1]), 0, len(q_axis) - 1)
                return intensity_3[iy, ix]

            lab = orbit_classify(Gq, intensity_of, cfg["mirrors"], tol_q)
            orbit_styles = {2: ('cyan', 'D', 'on a mirror line (invariant)'),
                            1: ('magenta', '*', 'mirror-paired (replica-type)'),
                            0: ('yellow', 'x', 'mirror-breaking (umklapp-type)')}
            for lv, (col, mk, name) in orbit_styles.items():
                sel = lab == lv
                if not np.any(sel):
                    continue
                ax3.scatter(Gq[sel, 0], Gq[sel, 1], color=col, s=60.0 / Gord[sel],
                            marker=mk, alpha=0.75, zorder=4, linewidths=1.0)
                legend_elements_3.append(
                    mlines.Line2D([0], [0], color='none', marker=mk, markeredgecolor=col,
                                  markerfacecolor='none', markersize=7, label=name))
            Gq = Gq[:0]          # orbit labels drawn instead of the index labels

        if len(Gq):
            styles = {
                0: ('cyan',    '+', r'substrate harmonic $n\mathbf{b}$'),
                1: ('red',     '+', r'overlayer harmonic $p\mathbf{c}$'),
                2: ('magenta', '*', r'replica harmonic $p\mathbf{c}^{(\sigma)}$'),
                3: ('yellow',  'x', r'mixed umklapp $n\mathbf{b}+p\mathbf{c}$'),
            }
            for cls_id, (col, mk, name) in styles.items():
                sel = (Gcls == cls_id) & (~Gamb)
                if not np.any(sel):
                    continue
                # marker area falls with total order, so the hierarchy is visible
                sizes = 60.0 / Gord[sel]
                ax3.scatter(Gq[sel, 0], Gq[sel, 1], color=col, s=sizes, marker=mk,
                            alpha=0.75, zorder=4, linewidths=1.0)
                legend_elements_3.append(
                    mlines.Line2D([0], [0], color='none', marker=mk, markeredgecolor=col,
                                  markerfacecolor='none', markersize=7,
                                  label=f'{name}, $N\\leq{int(umklapp_order)}$'))
            if np.any(Gamb):
                ax3.scatter(Gq[Gamb, 0], Gq[Gamb, 1], facecolors='none', edgecolors='0.7',
                            s=70, marker='o', linewidths=0.8, alpha=0.7, zorder=4)
                legend_elements_3.append(
                    mlines.Line2D([0], [0], color='none', marker='o', markeredgecolor='0.7',
                                  markerfacecolor='none', markersize=7,
                                  label=f'origin ambiguous ({int(np.sum(Gamb))} spots)'))

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

        # Substrate replicas live only in the scattering engine: the substrate
        # lattice is not part of the STM topography, so they cannot appear in its FFT.
        if not stm_fft_mode:
            for (phi, col) in ((theta_deg, 'lime'), (theta_deg + 30.0, 'green')):
                G_rep = reflect_points(G1_pts, phi)
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
        n_total = float(layer2_n0) + float(np.mean(delta_n_map))
        draw_fermi_surface_panel(ax4, cfg, theta_deg, k_max,
                                 interfacial_state, strain_coupling,
                                 n_layer2=n_total)

    return fig


def fermi_k(n_cm2, g_sv=4.0):
    """2D Fermi wavevector from a sheet carrier density: n = g_s g_v k_F^2 / (4 pi)."""
    n_A2 = max(float(n_cm2), 0.0) * 1e-16
    return float(np.sqrt(4.0 * np.pi * n_A2 / g_sv))


def draw_fermi_surface_panel(ax4, cfg, theta_deg, k_max, interfacial_state,
                             strain_coupling, n_layer2=None, fese_ecc=1.8):
    """Mutual band folding of the FeSe M-pockets and the MoS2 K-pockets."""
    ax4.set_xlim(-k_max, k_max)
    ax4.set_ylim(-k_max, k_max)
    title4 = "Fermi Surface Extended BZ (Mutual Band Folding)\n" + interfacial_state.split(':')[0]
    if n_layer2 is not None:
        title4 += (f" | layer 2: n = {n_layer2:.2e} cm$^{{-2}}$, "
                   f"$k_F$ = {fermi_k(n_layer2):.3f} " + r"$\AA^{-1}$")
    ax4.set_title(title4, color='white', fontsize=15)
    ax4.set_xlabel(r"$k_x$ ($\AA^{-1}$)", color='white')
    ax4.set_ylabel(r"$k_y$ ($\AA^{-1}$)", color='white')
    ax4.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax4.axvline(0, color='gray', lw=0.5, alpha=0.5)

    a_sub, a_top = cfg["sub_ax"], cfg["top_a"]
    r_sub = 0.175                      # FeSe M-pocket k_F, ~0.20 A^-1 experimentally
    # The overlayer pocket was also a hardcoded 0.10 A^-1 (n = 3.2e13 cm^-2), with
    # no link to the carrier density the doping panel computes. It is now derived
    # from n_layer2 = background doping + transferred charge.
    r_top = 0.10 if n_layer2 is None else fermi_k(n_layer2)
    if r_top < 1e-3:
        r_top = 1e-3

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

    def plot_fs(centers, r, col, lw, ls, alpha, ecc=1.0):
        """Draw pockets; ecc > 1 makes them ellipses elongated along Gamma-pocket.

        The FeSe M-point electron pockets are elliptical, not circular; the
        overlayer K pockets are taken as isotropic (ecc = 1).
        """
        for pt in centers:
            if abs(pt[0]) > k_max + r * ecc or abs(pt[1]) > k_max + r * ecc:
                continue
            if ecc == 1.0:
                ax4.add_patch(Circle((pt[0], pt[1]), r, color=col, fill=False,
                                     lw=lw, ls=ls, alpha=alpha))
            else:
                ang = np.degrees(np.arctan2(pt[1], pt[0]))
                ax4.add_patch(Ellipse((pt[0], pt[1]), 2 * r * np.sqrt(ecc),
                                      2 * r / np.sqrt(ecc), angle=ang, color=col,
                                      fill=False, lw=lw, ls=ls, alpha=alpha))

    plot_fs(M_pts, r_sub, 'cyan', 2.5, '-', 1.0, fese_ecc)
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

        plot_fs([m + g for m in M_pts for g in G2_shell1], r_sub, 'cyan', lw1, '--', alpha1 * strain_coupling, fese_ecc)
        plot_fs([k + g for k in K_pts for g in G1_shell1], r_top, 'red', lw1, '--', alpha1 * strain_coupling)
        plot_fs([m + g for m in M_pts for g in G2_shell2], r_sub, 'cyan', lw2, ':', alpha2 * strain_coupling, fese_ecc)
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
                plot_fs(M_rep, r_sub * 1.08, col, 2.0, '--', 0.9, fese_ecc)
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
        boundary_mode = st.selectbox("Domain Boundaries:",
                                     ["None", "Microscopic (Atomic)", "Mesoscopic (Envelope)"],
                                     help="Draws iso-contours at registry score 0.5 on Panel 1 "
                                          "and adds FWHM width / areal coverage to its legend.")

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
        scatter_model = st.selectbox("Scattering Model:", list(SCATTER_MODELS), index=1,
                                     help="Summing the two layer densities is single "
                                          "kinematic scattering. Multiplying them adds "
                                          "double diffraction, which is what generates the "
                                          "umklapp satellites and is routine in real LEED. "
                                          "Previously this was switched on as a side effect "
                                          "of selecting an interfacial phase state.")
        classify_mode = st.selectbox("Umklapp Classification:",
                                     ["Index-based (model)", "Point-group orbit (data-like)"],
                                     help="Index-based labels each spot by the (n,m,p,r) that "
                                          "generates it — only possible for a model. The orbit "
                                          "test instead asks whether the mirror image of a spot "
                                          "is also a spot of comparable intensity, which can be "
                                          "applied to a measured FFT.")
        fft_scale = st.slider("FFT Intensity Scale (% Max):", 0.1, 100.0, 10.0, 0.5,
                              help="Lower value enhances weak Moiré FFT spots")
        umklapp_order = st.slider("Umklapp Marker Order (n):", 0, 5, 3, 1,
                                  help="Panel 3 marks every spot at n·G₁ + m·G₂ with "
                                       "|n|+|m| ≤ this order. 0 hides the markers; "
                                       "higher orders explain more of the satellite spots "
                                       "at the cost of clutter.")

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
        if "Misfit Dislocation Glass" in interfacial_state and supports_interfacial_state(cfg):
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                dilatational = st.slider("Dilatational Fraction of u(r)", 0.0, 1.0, 0.5, 0.05,
                                         help="0 = divergence-free (the original model: no local "
                                              "area change, hence no misfit strain). 1 = purely "
                                              "dilatational. A real misfit network has both.")
            with gcol2:
                mosaic_L = st.number_input("Mosaic Wavelength (Å), 0 = auto", 0.0, 1000.0, 0.0, 10.0,
                                           help="Domain-wall spacing. 0 uses the moiré period, "
                                                "which is what actually sets domain size. The old "
                                                "model hardcoded 120 Å for every system and twist.")
        else:
            dilatational, mosaic_L = 0.5, 0.0

        if not supports_interfacial_state(cfg):
            st.caption("ℹ️ Modes 2 and 3 are only implemented for hex-on-square and "
                       "hex-on-rectangular systems; this system is rendered as a rigid vdW gap.")

        st.markdown("---")
        st.markdown("**2. Interfacial Mechanics & Doping Model**")

        relax_mode = st.selectbox("Mechanical Relaxation Model:", [
            "Rigid Lattices (No Relaxation)",
            "Fast Proxy (Algebraic Shift)",
            "Continuum Mechanics (PDE Solver)",
        ], help="Fast Proxy applies a rigid shift only — it adds no spatial structure.")

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

        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol2:
            substrate_csc = st.number_input(r"Substrate Space-Charge $C_{sc}$ (F/m², 0 = off)",
                                            0.0, 10.0, 0.0, 0.01, format="%.3f",
                                            key=f"csc::{system_mode}",
                                            help="SrTiO₃ is a semiconductor, not a metal: a "
                                                 "depletion/accumulation layer sits in series with "
                                                 "the gap. 0 omits it (the original assumption). "
                                                 "This is a lumped stand-in, not a Poisson solve "
                                                 "with field-dependent ε.")
        with qcol3:
            layer2_n0 = st.number_input("Layer 2 Background n₀ (cm⁻²)",
                                        0.0, 1e15, 2.7e13, 1e12, format="%.2e",
                                        key=f"n0::{system_mode}",
                                        help="Pre-existing carrier density in layer 2. Panel 4's "
                                             "pocket radius is now k_F = √(4πn/g_sg_v) with "
                                             "n = n₀ + ⟨Δn⟩, instead of a hardcoded 0.10 Å⁻¹. "
                                             "The default reproduces that old radius; set 0 to "
                                             "see the transferred charge alone.")
        with qcol1:
            layer2_cq = st.number_input(r"Layer 2 Quantum Capacitance $C_q$ (F/m²)",
                                        value=float(cfg["cq"]), step=0.05, format="%.3f",
                                        key=f"cq::{system_mode}",
                                        help="Cq = e²g_s g_v m*/(2πħ²). Monolayer MoS₂ K valleys "
                                             "(g_s=g_v=2, m*≈0.45 mₑ) give 0.60 F/m²; including the Q "
                                             "valleys would give ~2.4 F/m². The geometric capacitance "
                                             "ε₀/z is ≈0.026 F/m², so a realistic Cq leaves the interface "
                                             "geometry-limited and preserves the moiré contrast in Δn.")

        st.markdown("*Apparent-height weights of the three stacking registries (Panel 2):*")
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            w_co = st.slider("Coincident weight", 0.0, 1.0, 1.0, 0.05)
        with rcol2:
            w_br = st.slider("Bridge weight", 0.0, 1.0, 0.4, 0.05)
        with rcol3:
            w_ho = st.slider("Hollow weight", 0.0, 1.0, 0.0, 0.05)
        registry_weights = (w_co, w_br, w_ho)
        st.caption("ℹ️ The topography previously used the coincident site alone, so Panel 2 showed "
                   "a two-level contrast while Panel 1 resolved three registries. The ordering "
                   "above is a modelling choice; set bridge and hollow to 0 for the old behaviour.")

        smooth_len, k_vdw = 0.0, 400.0
        elastic_operator = "Bending rigidity (∇⁴)"
        if relax_mode == "Continuum Mechanics (PDE Solver)":
            st.markdown("*Continuum Tuning Parameters (Determines spatial smoothness vs structural pinning):*")
            scol1, scol2 = st.columns(2)
            with scol1:
                smooth_len = st.slider("Elastic Smoothing Length ℓ (Å)", 0.0, 50.0, 3.0, 0.5,
                                      help="Sets the elastic stiffness as κ = k_vdW·ℓ². The relaxed "
                                           "gap is the template low-passed at q = 1/ℓ, so ℓ should be "
                                           "compared with the moiré period. Replaces the old κ slider, "
                                           "whose units were pixels (ℓ ≈ 0.7 Å, sub-grid).")
            with scol2:
                k_vdw = st.slider(r"vdW Spring Stiffness $k_{vdW}$ (meV/Å⁴)",
                                  10.0, 2000.0, 400.0, 10.0,
                                  help="Restoring force toward the geometric template. Now in "
                                       "physical units, so it is directly comparable with the "
                                       "electrostatic pressure ε₀ΔV²/2z². A vdW well of depth "
                                       "~20 meV/Å² and width ~0.3 Å corresponds to ~400 meV/Å⁴. "
                                       "It is still a Hookean spring to the template, not a vdW "
                                       "potential with its own equilibrium.")
            elastic_operator = st.selectbox("Elastic Operator:",
                                            ["Bending rigidity (∇⁴)", "Membrane tension (∇²)"],
                                            help="∇⁴ is the correct operator for a monolayer "
                                                 "(response 1/(1+ℓ⁴k⁴)). ∇² is what the original "
                                                 "+κ∇²z force actually corresponded to and is kept "
                                                 "for comparison.")

        st.markdown("---")
        st.markdown("**3. Local Electron-Phonon Coupling Model**")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            eph_g0 = st.number_input("Base Coupling at min gap (meV)", value=80.0, step=5.0,
                                     help="Energy scale of the interfacial mode. For FeSe/SrTiO₃ the "
                                          "replica bands sit ~90-100 meV below the main bands.")
        with ecol2:
            eph_q0 = st.number_input(r"Phonon In-Plane Wavevector $q_0$ (Å⁻¹)",
                                     value=0.02, step=0.005, format="%.3f",
                                     help="An interfacial polar phonon couples as exp(−q₀z), so the "
                                          "decay length is 1/q₀. Forward-focused coupling (the FeSe/STO "
                                          "replica-band regime) needs q₀/k_F ~ 0.1, i.e. q₀ ~ 0.02 Å⁻¹ "
                                          "and 1/q₀ ~ 50 Å. Larger q₀ means zone-boundary phonons.")
        st.caption("ℹ️ At small q₀ the coupling is almost uniform across the moiré — that flatness *is* "
                   "forward focusing, not a bug. Note also that g(r) = g₀·exp(−q₀·z(r)) is a local "
                   "approximation, valid only when the moiré period ≫ 1/q₀; in the forward-focused "
                   "limit that condition is violated and the map should be read qualitatively.")

    if user_zmax < user_zmin:
        st.warning("Max gap is below min gap — swapping the two for this render.")
        user_zmin, user_zmax = user_zmax, user_zmin
    if eph_q0 <= 0:
        st.warning("Phonon wavevector must be positive; using 0.005 Å⁻¹.")
        eph_q0 = 0.005

    plot_kwargs = dict(
        zoom_factor=zoom_factor, q_max=q_max, k_max=k_max, view_mode=view_mode,
        boundary_mode=boundary_mode, umklapp_order=umklapp_order,
        mid_panel_mode=mid_panel_mode, panel3_mode=panel3_mode, den_cmap=den_cmap,
        den_contrast=den_contrast, fft_scale=fft_scale, relax_mode=relax_mode,
        w1=w1, w2=w2, user_zmin=user_zmin, user_zmax=user_zmax,
        smooth_len=smooth_len, k_vdw=k_vdw, eph_g0=eph_g0, eph_q0=eph_q0,
        layer2_cq=layer2_cq, scatter_model=scatter_model, mosaic_L=mosaic_L,
        dilatational=dilatational, registry_weights=registry_weights,
        substrate_csc=substrate_csc, layer2_n0=layer2_n0,
        classify_mode=classify_mode, elastic_operator=elastic_operator,
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
