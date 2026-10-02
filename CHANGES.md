# Changes — cleanup pass + two new material systems

## New material systems

| System | Geometry | Surface mesh used |
|---|---|---|
| `MoS₂/SrTiO₃(110) (Hex-on-Rect)` | hex-on-rectangular | a = 3.9050 Å along [001], a√2 = 5.5225 Å along [1̄10] |
| `MoS₂/Cu(100) (Hex-on-Square)` | hex-on-square | a₀/√2 = 2.5527 Å |

Axis convention for both rectangular meshes: **x ∥ [001], y ∥ [1̄10]**. This is the
same convention the existing Cu(110) entry already used (3.61 Å = a₀ along [001],
2.553 Å = a₀/√2 along [1̄10]), so all four cubic-substrate systems are now
consistent with each other.

The existing `MoS₂/SrTiO₃ (Hex-on-Square)` entry was renamed to
`MoS₂/SrTiO₃(100) (Hex-on-Square)` to disambiguate it from the new (110) entry.

Default unrelaxed gaps for the new systems were inherited from their (100)/(110)
counterparts (STO(110) ← STO(100); Cu(100) ← Cu(110)). **These are placeholders,
not literature values** — they are editable in the UI.

## Bugs fixed

1. **Substrate lattice was not centred on the origin.** `np.arange(-max_grid, max_grid, a)`
   put the nearest SrTiO₃ node at x ≈ −1.795 Å (≈ 0.46 a) rather than at 0, while
   the density functions `cos(2πx/a)` and the registry classifier `round(x/a)·a`
   both assume a node *at* the origin. Panel 1's grey substrate overlay was
   therefore drawn offset from the lattice the registry colours were computed
   against. Lattices are now built as `arange(-n, n+1)·a`.

2. **Glassy displacement field moved the atoms and the density in opposite
   directions.** Evaluating the density at `(X − U)` shifts the pattern by `+U`,
   but the atom positions were updated as `p − U`. Panel 2's "Geometry" metric
   (built from the atoms) and its "Local Doping"/"e-ph" metrics (built from the
   density) were consequently displaced oppositely. Atoms now move `+U`.

3. **Glassy field amplitude depended on the sampling set.** The field was
   normalised by the maximum over whatever points it was evaluated at, so the
   grid-sampled and point-sampled versions had different amplitudes, and the
   amplitude also drifted with the FOV zoom. Normalisation is now a single cached
   constant evaluated on a fixed reference grid.

4. **Quasicrystal replica overlays in Panel 3 were always labelled "FeSe"** in the
   hex-on-square branch, even when SrTiO₃ was selected (and would have said "Cu"
   for STO(110)). Labels are now taken from the system registry.

5. **Per-system gap defaults never refreshed.** `st.number_input` widget state is
   keyed by label + position, so switching systems left the previous system's
   min/max gap on screen. The two gap inputs are now system-scoped via `key=`.

6. **`SyntaxWarning: invalid escape sequence`** on Python 3.12+ from `"$\Delta n$"`,
   `"$\lambda$"`, `"$\mathbf{r}$"` in non-raw strings. All converted to raw strings.

7. **Degenerate colour ranges.** Flat inputs (e.g. min gap = max gap, or equal work
   functions) produced `vmin == vmax` in `imshow`. Now guarded by `safe_limits()`.

8. **`a_cuy` refined** from the rounded 2.553 to the exact a₀/√2 = 2.55266 Å, so
   that Cu(100) and Cu(110) derive from one constant. This changes the Cu(110)
   kinematic density by ≲ 8 × 10⁻² out of ≈ 12 (≈ 0.6 %); every other existing
   system is bit-identical to the original (verified numerically).

## Dead / redundant code removed

- **`boundary_mode`** ("Domain Boundaries: None / Microscopic / Mesoscopic") was
  read from the UI and passed through the whole call chain but **never used
  anywhere** in the plotting function. The control was removed. If this was a
  planned feature rather than an abandoned one, it needs re-adding along with an
  implementation.
- The duplicated hex-on-square and hex-on-rect branches (~110 lines, identical
  apart from `ax` vs `ay` and the mirror angles) were merged into one
  "hex-on-orthorhombic" path; square is simply the `ax == ay` case.
  `get_square_density/_G/_bz` are now `get_ortho_*` with equal arguments.
- `T_top_den` / `T_top_fft` were computed, then unconditionally recomputed with
  identical arguments inside each branch.
- `T_fft_engine` was computed in the glass branch and never used (Panel 3 rebuilds
  the two layers separately there). It is now `None` in that branch.
- `st.session_state.is_rendering_video` was set but never read.
- Unused import `matplotlib.tri`; `matplotlib.pyplot` replaced by a direct
  `matplotlib.patches.Circle` import (pyplot's global state is best avoided in a
  Streamlit app).
- `label=` keyword arguments on the Panel 3 / Panel 4 artists were ignored, since
  both legends are built from explicit `handles=` lists.
- `intrinsic_z`, the selectbox list, the `lbl1_short` if/else chain, `max_theta`,
  and the quantum-capacitance override for MATBG were five separate places that
  had to agree on the system name. They are now one `SYSTEMS` registry entry.
- `requirements.txt`: `imageio` was listed twice (plain and with the `ffmpeg`
  extra); the plain entry was dropped.
- The twist-scan video wrote `moire_twist_scan.mp4` into the current working
  directory, which is not reliably writable in a PyInstaller bundle. It now goes
  to the system temp directory.
- Interactive and video figures used different figure sizes (22 × 6.8 vs 21 × 6.5)
  for the same layout; both now call `figure_size()`.
- The script body moved into `main()` under an `if __name__ == "__main__"` guard,
  so the physics layer can be imported and tested without starting Streamlit.

## Known limitations left in place (not changed)

- **Interfacial phase states 2 and 3 are only implemented for hex-on-orthorhombic
  systems.** They silently did nothing for Bi₂Se₃ / Graphene / MATBG. The
  behaviour is unchanged, but the UI now says so explicitly.
- The LEED grid (N = 512 over L = 400 Å) has a Nyquist cut-off at q ≈ 4.02 Å⁻¹,
  while the q-Zoom slider goes to 8 Å⁻¹. Beyond ≈ 4 Å⁻¹ the map is empty even
  though the overlay markers are still drawn. Raising `N_FFT` to 1024 fixes it at
  ~4× the FFT cost.
- In the "rigid" branch the scattering engine is the **sum** of the two layer
  densities (independent kinematic scattering), whereas the quasicrystal branch
  uses their **product** (which generates the umklapp/moiré satellites). This
  asymmetry is inherited from the original code and was left as-is, but it is
  worth deciding deliberately which one you want.
- Panel 2's subtitle reports the relaxed gap range even in "Geometry" mode, where
  the displayed image is the STM topography rather than the gap map.
- `Z_stm_exact` is still built with a per-atom Python loop (~1.6 k atoms at 1× zoom,
  ~40 k at 5×). It is the dominant cost of a rerun and of video rendering.

---

# Update 2 — Panel 3 overlay mismatch, and `boundary_mode` restored

## Why the replica symbols matched nothing in "FFT of STM Topography" mode

Measured on your exact configuration (STO(100), θ = 33°, 3× FOV, quasicrystal
state, coupling 0.4), peak-picking the STM-FFT and testing each overlay set
against it:

```
              MoS₂ G2 (red squares)     6/6   markers on a real spot
              STO  G1 (cyan circles)    4/4
              MoS₂ 0°  replica          0/6      <-- all four replica sets
              MoS₂ 45° replica          0/6           matched nothing
              STO  33° replica          0/4
              STO  63° replica          0/4
              1st-order umklapp G1±G2  20/36   but explained only 10 of 60 spots
```

Two independent causes [supported by the measurements above]:

**(a) The quasicrystal state never reached the topography.** It modified
`T_total` and the FFT engine only; `vis_top` and `dist_co` — the sole inputs to
`stm_topography` — were byte-identical with and without it. Panel 2 and its FFT
were therefore always the plain rigid structure, so replica markers could not
land on anything by construction. In the LEED engine the same markers are
correct: all four sets sit on real peaks, 20/20.

**(b) The satellite spots are a whole lattice, not one difference.** The
topography is the MoS₂ lattice multiplied by a registry envelope with the moiré
period, so its Fourier support is the full module `n·G₁ + m·G₂`, not just
`G₁ ± G₂`. The 1st-order set accounted for 10 of 60 spots; the order-≤5 module
accounts for 56 of 60.

## Fixes

1. **The quasicrystal state now modulates the topography.** In QC mode the
   overlayer enters `stm_topography` as three weighted sublattices — the primary
   one plus a mirror reflection across each substrate mirror line, at the same
   weight (`strain_coupling × 0.4`) the density engine already used. Each replica
   carries its own registry amplitude computed against the substrate. After this,
   both MoS₂ replica marker sets land 6/6 on real STM-FFT spots.
   *This changes Panel 2's appearance in QC mode* (the replicas are now visible in
   the topography) — that is the point, but flag it if it isn't what you wanted;
   reverting is one line in `build_interface`.

2. **Umklapp markers generalised to order n**, with a new "Umklapp Marker Order
   (n)" slider (0–5, default 3). Markers are drawn at every `n·b₁+m·b₂+p·c₁+r·c₂`
   with `|n|+|m|+|p|+|r| ≤ order`. In QC mode the replica sublattices get their
   own satellite sets too (order 3: 64 of 90 spots explained, vs 34 without).
   Set the slider to 0 to hide them.

3. **Overlays are now mode-aware.** Umklapp markers are shown in STM-FFT mode
   unconditionally (the registry envelope is always present, independent of the
   interfacial state), whereas in LEED mode they still appear only once an
   interfacial state is switched on. Substrate replica markers are suppressed in
   STM-FFT mode: the substrate lattice is not part of the topography, so they
   cannot appear there regardless.

**Still unexplained at default settings:** roughly a quarter of the weak spots.
They are higher-order cross terms (order > 3) and satellites of satellites;
raising the slider to 5 catches most of them at the cost of a dense marker field.

## `boundary_mode` restored

You were right that it existed — it is in the early version you sent, and it was
already dead in the version you gave me first (the argument was threaded through
`create_unified_plot` but the body never referenced it). It has been restored
from your early file:

- Selector back in column 1: `None` / `Microscopic (Atomic)` / `Mesoscopic (Envelope)`.
- **Microscopic**: `tricontour` on the Delaunay triangulation of the overlayer
  atoms, iso-line at registry score 0.5.
- **Mesoscopic**: atom scores scattered onto a padded grid, dilated with a
  *circular* footprint of radius 1.5 a (so the envelope keeps the structural
  symmetry), Gaussian-smoothed, contoured at the midpoint of the smoothed range.
- **Panel 1 legend restored** (it had also been dropped), with the FWHM domain
  width and areal coverage annotations that only appear when boundaries are on:
  `Coincident (W: 1.6Å, Cov: 15.1%)`.

Changes made while restoring: `decay_L` and the hollow/bridge scale factors are
now returned from `build_interface` as `decay_widths` so the widths are computed
from one place; `a_mos2` → `cfg["top_a"]` and `N_den` → `N_DEN` for the new
registry; guards added for degenerate cases (fewer than 4 atoms, a score field
entirely above or below 0.5, an empty padded region) that would previously have
raised inside `tricontour`.

## Verification

126 render configurations (8 systems × 3 interfacial states × 3 boundary modes,
plus 6 topology views × 3 boundary modes × 3 umklapp orders on STO(100) at 33°
in STM-FFT mode) all render without error or numerical warning.

---

# Update 3 — umklapp spots classified by symmetry origin

Panel 3's single yellow-× umklapp set is replaced by four classed sets, each spot
labelled by the index vector (s; n,m,p,r) that generates it: cyan `+` substrate
harmonic, red `+` overlayer harmonic, magenta `*` replica harmonic, yellow `×`
mixed interlayer umklapp, grey `○` origin ambiguous. Marker area scales as 1/N so
the order hierarchy is visible. New functions `umklapp_indexed()` and
`classify_umklapp()`; the latter documents the minimal-order canonical rule and the
resolution tolerance that make the assignment well posed for a dense module.

See `PHYSICS_AUDIT.md` Part 1 for the scheme, the measured intensity-vs-class
table that validates it, and the ambiguity measurements. Part 2 of that document
is a module-by-module physics audit with evidence; Part 3 is a prioritized queue of
what is still open.

---

# Update 4 — audit items 1–3 (and 4) implemented

- **Cq** moved to registry constants (MoS₂ 0.60 F/m², MATBG 0.002 F/m²) and exposed
  in the UI. Δn contrast improves 3.3×, absolute level 3.7×.
- **Relaxation solver** replaced with a semi-implicit spectral scheme. The κ slider
  becomes "Elastic Smoothing Length ℓ (Å)". Zoom dependence 27 % → ~10 %, exactly
  converged, unconditionally stable, reflect-padded boundaries. Retires item 4.
- **e-ph coupling** reparametrised from a bare decay length λ to the phonon
  wavevector q₀, with g₀ referenced to the actual relaxed minimum gap.

Measurements and the remaining open items are in `PHYSICS_AUDIT.md` Part 4.

---

# Update 5 — audit items 5–11 and the remaining §2 sub-items

New UI controls: **Scattering Model** (kinematic vs double diffraction),
**Umklapp Classification** (index-based vs point-group orbit), **Dilatational
Fraction** and **Mosaic Wavelength** (glass mode), **Elastic Operator** (∇⁴ vs ∇²),
**Substrate Space-Charge C_sc**, **Layer 2 Background n₀**, and three
**registry apparent-height weights**.

Physics changes: real structure factor for glass scattering (replaces the
intensity blur); dilatational component in the misfit field; mosaic scale tied to
the moiré period; `N_FFT` 512 → 1024; registry-resolved STM contrast sharing
Panel 1's decay widths; Fermi pockets derived from n₀ + ⟨Δn⟩ and drawn elliptical
for FeSe; electrostatic term and `k_vdW` in physical units (meV/Å³, meV/Å⁴);
bending rigidity as the default elastic operator; the `ΔW = 1e-6` fudge actually
removed this time.

Measurements for each, and the eight items that remain genuinely open, are in
`PHYSICS_AUDIT.md` Part 5.
