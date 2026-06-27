# CFD Project: Laminar & Turbulent Channel Flow — Pressure Drop and Friction Factor in OpenFOAM

A complete, ready-to-run OpenFOAM project for a 2nd-year mechanical engineering
student. It validates CFD results against the fluid mechanics theory you've
already studied (Hagen–Poiseuille flow, Darcy friction factor, Moody chart).

---

## 0. Why "channel flow" and not a round pipe?

You picked "pipe / channel flow". A *true* circular pipe needs either a full
3D mesh or a 2D **axisymmetric wedge** mesh — both are trickier to set up
correctly and easier to get subtly wrong (wedge angle, axis treatment, etc.).

So this project uses **flow between two infinite parallel plates** (a 2D
channel) instead. The physics is the same family of problem you studied in
Fluid Mechanics — viscous shear, pressure-driven flow, friction factor,
laminar-to-turbulent transition — and the geometry is simple enough that you
can be confident every mesh and boundary condition is correct. The maths
below is for this exact channel geometry, not a circular pipe (the
correlations differ slightly — see the table).

**Stretch goal (optional, after this works):** redo the laminar case as a 2D
axisymmetric wedge mesh of a real circular pipe and compare. That's a
natural "Part 2" if you want to extend this into a bigger project.

---

## 1. The geometry

We simulate **half** the channel, using a symmetry plane at the centreline
(this is standard practice — it halves the mesh size with zero loss of
accuracy, since the flow is symmetric about the centreline anyway).

```
                    wall  (no-slip)
        ───────────────────────────────
        |                              |  h = 0.01 m (half-gap)
inlet → |     -----> flow direction    | → outlet
        |                              |
        ───────────────────────────────
                 symmetry (centreline)

        |<----------- L = 3 m -------->|
```

- Full channel gap: H = 2h = 0.02 m
- **Hydraulic diameter**: Dh = 2H = 0.04 m (derivation: Dh = 4A/P, and for an
  infinitely wide channel A = H·(width), P = 2·(width), so Dh = 2H)
- Fluid: air, ν = 1.5×10⁻⁵ m²/s
- Two cases, same geometry, different inlet velocity:

| Case | Re_Dh | Inlet U (m/s) | Regime |
|---|---|---|---|
| `laminarChannel`   | 500   | 0.1875 | Laminar |
| `turbulentChannel` | 20000 | 7.5    | Turbulent (k-ω SST) |

---

## 2. The theory (so you understand *why*, not just *how*)

### 2.1 Laminar case — exact solution

For fully developed laminar flow between parallel plates (gap H), the
classic result (derivable from the Navier–Stokes equations — you may have
seen this exact derivation in your Fluid Mechanics course) is:

```
-dp/dx = 12 μ U / H²
```

Combining this with the Darcy–Weisbach definition of friction factor,
f = (-dp/dx)·Dh / (½ρU²), and substituting Dh = 2H, gives the clean result:

```
f = 96 / Re_Dh        (exact, for parallel-plate channel flow)
```

(Compare this to f = 64/Re for a circular pipe — same idea, different
geometry constant, because the velocity profile and wall shear distribution
differ.)

### 2.2 Turbulent case — empirical correlation

There's no exact closed-form solution once the flow is turbulent. We use the
**Blasius correlation** (valid for smooth walls, Re < ~10⁵), which you may
have already seen on the Moody chart as the smooth-pipe curve:

```
f = 0.316 · Re_Dh^-0.25
```

### 2.3 Entrance length

Both correlations assume **fully developed flow** — the velocity profile has
stopped changing with x. Near the inlet, the profile is still developing.
Rough entrance-length estimates:

- Laminar:   Le/Dh ≈ 0.05–0.06 × Re_Dh  →  for Re=500, Le ≈ 1.0–1.2 m
- Turbulent: Le/Dh ≈ 10–60 (much shorter, turbulence mixes the profile fast)

This is *why* the channel is 3 m long and *why* the post-processing scripts
only fit the pressure profile over x = 1.5–3.0 m — to stay safely inside the
fully developed region for both cases.

---

## 3. Running it

You said OpenFOAM is already installed and working — good. From inside
each case folder:

```bash
cd laminarChannel
blockMesh          # builds the mesh from system/blockMeshDict
checkMesh          # sanity check - look for "Mesh OK" and no negative volumes
simpleFoam         # runs the steady-state incompressible solver
```

Watch the residuals print to the terminal. The case is set to stop either at
the iteration limit or once residuals drop below the `residualControl`
values in `system/fvSolution` (1e-4) — whichever comes first.

Do the same for `turbulentChannel` (it will take a bit longer — more
iterations, more cells).

### Quick visual check

```bash
paraFoam     # opens ParaView with the case loaded
```

Look at the U field — you should see a parabola-ish profile (laminar) or a
flatter, fuller profile (turbulent) that develops from a uniform inlet.

### Mesh quality check for the turbulent case (important!)

```bash
postProcess -func yPlus -latestTime
```

This prints the y+ value at the wall. For the `kqRWallFunction` /
`omegaWallFunction` / `nutkWallFunction` set used here, you want the
near-wall cell's y+ to land roughly in the **30–300** range. If it's way
outside that, edit the `simpleGrading` line in
`turbulentChannel/system/blockMeshDict` (make the near-wall cells thinner or
thicker), then `blockMesh` and re-run. This kind of mesh-sensitivity check
is itself a legitimate, gradeable part of a CFD project — write down what
you tried and why.

---

## 4. Post-processing — turning the run into results

See `postProcessing_scripts/README_scripts.md`. In short, from inside a case
folder:

```bash
python3 ../postProcessing_scripts/friction_factor.py --regime laminar
python3 ../postProcessing_scripts/velocity_profile.py --regime laminar
```

(swap `laminar` for `turbulent` in the other case folder)

This gives you:
1. A computed Darcy friction factor from the CFD pressure drop, compared
   numerically against the theoretical/empirical value (§2).
2. A plot of the CFD velocity profile against the analytical parabola (or a
   log-law sketch for the turbulent case), saved as a PNG.

**Expect a few percent difference between CFD and theory** — that's normal
and is itself worth discussing in your report (mesh resolution, numerical
scheme, finite domain). A huge difference (>15–20%) usually means the case
hasn't converged, the mesh is too coarse, or you're sampling too close to
the inlet (still developing flow).

---

## 5. Suggested report structure

1. **Objective** — validate OpenFOAM against classical fluid mechanics
   theory for channel flow, laminar and turbulent.
2. **Theory** — the derivation in §2 (rewrite in your own words — this
   shows you understand it, not just copied it).
3. **Methodology** — geometry, mesh (include a screenshot from ParaView or
   `checkMesh` output), boundary conditions, solver settings, why you chose
   the mesh grading you did.
4. **Mesh/convergence check** — show the y+ check for the turbulent case,
   and ideally one mesh-refinement comparison (re-run with double the cells
   and show the friction factor barely changes — this is called a "grid
   convergence study" and is genuinely good practice to mention even at
   2nd-year level).
5. **Results** — friction factor: CFD vs. theory, % difference, for both
   regimes. Velocity profile plots for both regimes.
6. **Discussion** — why does the turbulent profile look "fuller" than the
   laminar parabola? Why is the turbulent friction factor lower or higher
   than you might naively expect? What would change with a rougher wall
   (not modelled here)?
7. **Conclusion**

---

## 6. If you want to go further (optional extensions)

- **Reynolds number sweep**: re-run the laminar case at 3–4 different Re
  and plot CFD friction factor vs. Re_Dh on a log-log Moody-style chart
  alongside the f=96/Re line.
- **Mesh convergence study**: double the cell counts in `blockMeshDict` and
  show the friction factor changes by less than ~1–2%.
- **Axisymmetric pipe**: extend to a real circular pipe with a wedge mesh
  and compare f=64/Re instead of f=96/Re — this directly answers "what
  about an actual pipe?" if an examiner asks.

Good luck with it — this is a genuinely solid 2nd/3rd-year CFD project, and
the validation-against-theory angle (rather than just "I ran a simulation")
is exactly what makes it look rigorous.
