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
