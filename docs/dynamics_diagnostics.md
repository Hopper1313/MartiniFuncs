# Dynamics diagnostics: what is computed, and why

Code: `MartiniFuncs/statistics/dynamics.py`, plus new outputs of `trajectory.analysis.analyze_run`:
`com_nonaffine_angstrom`, `cell_origin_angstrom`, `orientation_axes`, topology events, and
`analyze_segments`. All quantities use molecular centres of mass (COMs) unless stated otherwise.

Notation:

- **R_m(t)**: COM of molecule m. Because it is tracked across periodic boundaries, d = R(t₀+τ) − R(t₀)
  is a true displacement.
- **⟨…⟩**: an average over molecules of one species and over all time origins t₀.
- **H**: the cell, with lattice vectors as rows.
- **o**: the cell origin (the LAMMPS lo corner).
- **s**: fractional coordinates, r = o + s·H.

---

## 1. Shear-flow-subtracted (non-affine) displacement

**Problem.** Under `fix deform` with `nvt/sllod`, the fluid carries an imposed streaming velocity. Lab-frame
displacements, which is what the default COM tracking (TOR) measures, contain this flow. The x-MSD then
grows faster than linearly (Taylor dispersion) and stops being a diffusion measurement.

**Streaming field.** LAMMPS uses the box-deformation rate for it: u(r) = s·(dH/dt), with s measured from
the box lo corner. For an xy tilt changing at rate dxy/dt = γ̇·L_y this gives u_x = γ̇ (y − y_lo), simple
shear with γ̇ the `erate`.

**Affine displacement.** A point that only rides with the deformation keeps its fractional coordinate s.
Its displacement over one frame is exactly

  Δr_aff = (o_t − o_{t−1}) + s · (H_t − H_{t−1}).

The non-affine step is the TOR step minus Δr_aff. s is evaluated at the midpoint of the step,
s̄ = (s_{t−1} + s_t)/2, using the continuous end point. The error of the midpoint rule is second order in
the frame spacing, and `analyze_run` applies it at every source frame:

  u_t = u_{t−1} + mic(W_t − W_{t−1}, H_t) − Δr_aff.

**Special cases:**
- **Constant cell:** Δr_aff = 0, so the non-affine trajectory equals TOR exactly.
- **NPT with a fixed centre:** the origin moves by −ΔL/2, so Δr_aff = (s − ½)·ΔL, the barostat's dilation
  about the centre. This term is bounded and averages to zero, so subtracting it is optional for NPT.

**Checks:**
- Pure affine motion (shear, or dilation about the centre) gives zero non-affine displacement to 1e-9 Å.
- Shear plus Brownian motion recovers 6Dt, while the lab-frame MSD is more than twice as large.

---

## 2. Caging and sampling diagnostics

The COM MSD of a dense liquid has three regimes:
- **ballistic,** MSD = ⟨|V|²⟩ t², with ⟨|V|²⟩ = 3 k_BT/M (equipartition for a COM);
- **a plateau** while each molecule rattles in the cage formed by its neighbours;
- **diffusive,** MSD → 6Dt once molecules escape their cages.

### 2.1 Is the first lag resolved? (`sampling_diagnostics`)

Free flight over one frame spacing Δt would give MSD(Δt) = ⟨|V|²⟩Δt². The **ballistic ratio** is

  MSD(Δt) / (⟨|V|²⟩ Δt²).

- **About 1:** the first saved lag is still ballistic, so short-time motion is resolved.
- **Much less than 1:** collisions and caging already happened within one frame. The single-reference
  MSD then "jumps" to the plateau at the first saved point, which is the effect you observed.

C(Δt)/C(0) near zero likewise means the velocity memory is shorter than Δt, so VACF and Green–Kubo
integrals cannot be resolved. The crossover time √(MSD(Δt)/⟨|V|²⟩) estimates when free flight ends.

### 2.2 Non-Gaussian parameter (`displacement_moments`)

For a Gaussian displacement distribution in 3D, ⟨|d|⁴⟩ = (5/3)⟨|d|²⟩². This holds for both ballistic and
diffusive motion. The parameter

  α₂(τ) = 3⟨|d|⁴⟩ / (5⟨|d|²⟩²) − 1

is therefore 0 for Gaussian motion and positive when the displacements are heterogeneous: a few
molecules have moved far while most are still caged.

**Example.** If a fraction f of molecules move as a Gaussian and the rest not at all, then
α₂ = 1/f − 1 (the test uses f = ½, giving α₂ = 1). In glassy liquids α₂ peaks at the end of the plateau
and returns to 0 in the diffusive regime.

### 2.3 Self part of the van Hove function (`van_hove_self`)

G_s(r, τ) = ⟨δ(r − |d(τ)|)⟩ / 4πr². It is returned as the radial density p(r) = 4πr²G_s, which integrates
to 1, next to the Gaussian with the same MSD:

  p_G(r) = 4πr² (3/(2π MSD))^{3/2} exp(−3r²/(2 MSD)).

A tail, or a second peak, above p_G at large r is direct evidence of cage-escape (hopping) events.

### 2.4 Correlation of successive displacements (`step_correlation`)

  χ(k) = ⟨d₁·d₂⟩ / ⟨|d₁|²⟩, with d₁ = R(t+k) − R(t) and d₂ = R(t+2k) − R(t+k).

| motion | χ(k) |
|---|---|
| ballistic (persistent) | → +1 |
| free diffusion (independent increments) | 0 |
| caged | negative |

**Why −½ for a caged molecule.** Write R = c + ε, with c a slowly moving cage centre and ε a fast
fluctuation, independent between frames, with ⟨ε²⟩ = σ². Then

  d₁ = Δc₁ + ε₁ − ε₀ and d₂ = Δc₂ + ε₂ − ε₁, so ⟨d₁·d₂⟩ = −σ² and ⟨|d₁|²⟩ = 2σ²,

so χ = −½. This was the signature found in your 0.5 ps NPT data.

---

## 3. Green–Kubo diffusion (`green_kubo_diffusion`)

The MSD and the velocity autocorrelation C(t) = ⟨V(0)·V(t)⟩ are exactly related:

  MSD(t) = 2 ∫₀ᵗ (t − s) C(s) ds,

so d MSD/dt → 2∫₀^∞ C ds and

  D = (1/3) ∫₀^∞ ⟨V(0)·V(t)⟩ dt.

The running integral D(τ) (trapezoidal rule) is returned; its plateau is the estimate. It must agree with
the MSD slope, since both measure the same quantity.

**Caveats:**
- The frame spacing must be much shorter than the VACF decay time; 5 fs works, 0.5 ps does not.
- The long-time tail of C(t) makes the integral noisy. Report D(τ) over a window and its spread, not a
  single point.

**Check:** an exponential VACF, 3σ²e^{−γt}, gives D = σ²/γ to 1e-6.

---

## 4. Finite-size correction (`yeh_hummer_correction`)

A diffusing molecule in a periodic box drags a hydrodynamic flow that also acts on its own periodic images,
and this slows its apparent diffusion. For a cubic box of edge L (Yeh & Hummer, J. Phys. Chem. B 108,
15873, 2004):

  D_∞ = D_PBC + k_BT ξ / (6π η L), with ξ = 2.837297.

ξ is the lattice sum for a cubic lattice of images, and η is the shear viscosity of the same model.
`cubic_box_length` refuses boxes more than 1% anisotropic. The correction scales as 1/L, so for small
boxes it can be comparable to D itself.

---

## 5. NEMD shear viscosity (`nemd_shear_viscosity`)

In steady simple shear, a Newtonian fluid has shear stress σ_xy = η γ̇. LAMMPS prints the pressure
tensor P = −σ, so

  η = −⟨P_xy⟩ / γ̇.

Units: γ̇ is the `fix deform … erate` (1/ps); 1 bar = 10⁵ Pa and 1/ps = 10¹² s⁻¹. The uncertainty is the
standard error of block means.

**Valid only:**
- in the linear-response regime (η independent of γ̇);
- after the stress has reached a steady state.

If the block values keep rising, the stress is still building: the material is responding
**elastically**, with σ ≈ G·γ for strain γ = γ̇t, rather than flowing. In that case only an apparent
viscosity, or a shear modulus, can be stated (see the results below).

---

## 6. Rotational dynamics (`orientation_correlation`, `analyze_run(orientations=True)`)

### 6.1 Axes

For each molecule, the mass-weighted gyration tensor is

  S = Σᵢ mᵢ (rᵢ − R)(rᵢ − R)ᵀ.

Its eigenvectors are the principal axes, also those of the inertia tensor I = tr(S)𝟙 − S:
- the eigenvector with the largest eigenvalue is the **long axis**;
- the one with the smallest is the direction of least extent, which for a planar molecule is the **plane
  normal**.

Eigenvector signs are arbitrary, so `analyze_run` flips each axis when needed to keep it continuous in
time.

### 6.2 Correlation functions

  C_l(τ) = ⟨P_l(u(t₀) · u(t₀+τ))⟩, with P₁(x) = x and P₂(x) = (3x² − 1)/2.

- **P₂:** independent of the sign of u, and the quantity measured by NMR relaxation and depolarized
  light scattering. It is computed by FFT from the tensor q_ab = u_a u_b, because
  ⟨(u·u′)²⟩ = Σ_ab ⟨q_ab q′_ab⟩.
- **P₁:** a plain vector correlation, also by FFT.

### 6.3 Rotational diffusion

For small-step rotational diffusion with coefficient D_r, P_l is an eigenfunction of the rotational
diffusion operator (eigenvalue −l(l+1)D_r), so

  C_l(τ) = exp(−l(l+1) D_r τ), and τ_l = ∫C_l dt gives τ₁ = 1/(2D_r), τ₂ = 1/(6D_r) (ratio 3).

`correlation_time` integrates up to the first zero or a chosen upper limit. A plateau of C_l below 1 that
does not decay means libration (rocking) within a cage without full reorientation.

**Caveat:** when two gyration eigenvalues are nearly equal, that axis is poorly defined and can switch
between the near-equal directions.

**Check:** simulated rotational diffusion reproduces exp(−2D_rτ) and exp(−6D_rτ) to within 0.03.

---

## 7. Reaction-aware tracking (`on_topology_change`, `analyze_segments`)

**Reference topology.** The molecules of the segment's first frame, given by the bond rule
|r_j + S·H − r_i| < 1.3(r_cov,i + r_cov_j).

**Change.** A frame differs from the reference if any bond joins two reference molecules (a merge) or any
molecule splits into several pieces. `trajectory.reactions.topology_change` returns the changed
components; it is generic and not limited to one product size.

**Persistence.** A change counts once it lasts `topology_persistence` consecutive source frames, meaning
every component that changed at its first frame stays inside a changed component. A product that also
grabs a neighbour for a moment still persists; a close contact that separates again does not. The
event's frame is the first frame of that run.

**Segments.** `analyze_segments` ends a segment at the event and starts the next one there, re-deriving
molecules from that frame. Products become new species (grouped by formula), and TOR and lag statistics
never cross a boundary, so every fixed-membership statistic stays valid within each segment.
`on_topology_change="record"` lists events without splitting.

The 1.3 bond rule infers connectivity from distances. A long-lived close contact can look like a bond, so
inspect events (atom IDs, formulas) before interpreting them chemically.

---

## 8. Potentials of mean force (`pmf_from_rdf`, `pmf_1d`, `pmf_2d`)

For a pair distance, g(r) is the probability density relative to an ideal gas, so the reversible work to
bring the pair from infinite separation to r is

  W(r) = −k_BT ln g(r).

For any sampled coordinate x, W(x) = −k_BT ln[P(x)/J(x)] + const, where J is the density that x would have
without interactions:
- J ∝ sin θ for an angle between two directions in 3D;
- J ∝ r² for a distance;
- J = 1 otherwise.

`pmf_2d` combines a distance with an angle. `load_pair_geometries` reads the forming distances and angles
from an export's `index.jsonl`.

**Caveat:** pair geometries exist only inside the 3.25 Å screen, so their PMF is conditional on being a
candidate pair. For the full distance dependence, use the species COM RDF.

---

## 9. Results on your NVT shear run (2.5 GPa, 400 K; first 20 ps, saved every 5 fs)

| | C₅H₆ (CPD) | C₁₁H₉NO₂ |
|---|---|---|
| ballistic ratio, 5 fs saving | 0.999 (resolved) | 0.999 |
| ballistic ratio, if saved every 0.5 ps | 0.075 (not resolved) | 0.12 |
| free-flight crossover | 0.14 ps | 0.18 ps |
| first VACF zero (backscattering off the cage) | 0.085 ps | 0.11 ps |
| MSD plateau (0.3–10 ps) | 0.24–0.30 Å² | 0.13–0.16 Å² |
| α₂ maximum (0.5–10 ps) | about 0.2 | about 0.16 |
| χ(k) at 0.005 / 0.1 / 0.5 ps | +0.99 / −0.04 / −0.51 | +0.99 / +0.21 / −0.55 |
| D from MSD slope, 5–10 ps | ~7e-8 cm²/s | ~4e-8 cm²/s |
| P₂ of plane normal at 19 ps | 0.87 | 0.96 |
| P₂ of long axis at 19 ps | 0.34 (see caveat in §6) | 0.99 |

- **Lab-frame vs non-affine MSD:** indistinguishable over 20 ps at γ̇ = 10⁻⁴ ps⁻¹, as expected.
- **Green–Kubo D(τ):** fluctuates between about 10⁻⁷ and 10⁻⁶ cm²/s and has not converged in 20 ps. Over
  this window it neither confirms nor contradicts the MSD slope.
- **NEMD stress (full 1.5 ns log):** −⟨P_xy⟩/γ̇ = 0.94 ± 0.13 Pa·s (1.25 ± 0.09 in the second half), but
  the block values rise steadily, from 0.06 to 1.57 Pa·s. The stress is still building with strain (total
  strain γ̇t = 0.15). The response is therefore elastic, with an apparent shear modulus of order
  1570 bar / 0.15 ≈ 1 GPa, not a steady viscous flow. This is solid- or glass-like.
- **Finite size:** with η ≈ 1 Pa·s the Yeh–Hummer correction for this 22.7 Å box is about 4e-9 cm²/s.

**Interpretation.** At 2.5 GPa and 400 K both species are caged:
- motion is ballistic up to about 0.15 ps;
- a velocity backscatter at about 0.1 ps;
- a displacement plateau of a few tenths of Å²;
- heterogeneous displacements (α₂ > 0) and very slow escape;
- molecules that barely reorient, and a shear stress that grows elastically.

So the "jump" in the single-reference MSD at 0.5 ps saving is sub-frame cage rattling. Diffusion
coefficients from runs this short (or at this pressure) are upper bounds on a nearly arrested system, not
converged liquid values.
