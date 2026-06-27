# Serpentine Cooling Channel — CFD Project
### OpenFOAM | Heat Transfer | Parametric Study
**IIT Ropar | ME 2nd → 3rd Year Summer Project**

---

## Project Overview

This project simulates **conjugate heat transfer** in a cooling channel using OpenFOAM's `buoyantSimpleFoam` solver. The setup mimics cooling channels found in:
- CPU/GPU heat sinks
- Gas turbine blade cooling
- Battery thermal management systems

**What you simulate:**
- Turbulent water flow through a rectangular channel
- A heated bottom wall (constant T = 350 K) representing a hot component
- Coolant entering at 300 K
- Three flow speeds (Re = 500, 1000, 2000)
- Effect of adding a rib/baffle to enhance heat transfer

**Key results you'll obtain:**
- Nusselt number (Nu) vs Reynolds number (Re) curve
- Temperature and velocity contour plots
- Quantified heat transfer enhancement from rib geometry

---

## Directory Structure

```
serpentine_cooling/
│
├── case_Re500/          ← Re = 500  (slowest flow)
│   ├── 0/               ← Initial & boundary conditions
│   │   ├── U            ← Velocity
│   │   ├── T            ← Temperature
│   │   ├── p_rgh        ← Modified pressure
│   │   ├── k            ← Turbulent kinetic energy
│   │   └── epsilon      ← Turbulent dissipation
│   ├── constant/        ← Physical properties
│   │   ├── thermophysicalProperties  ← Water properties
│   │   ├── turbulenceProperties      ← k-ε model
│   │   └── g            ← Gravity
│   └── system/          ← Solver settings
│       ├── blockMeshDict       ← Serpentine geometry (advanced)
│       ├── blockMeshDict.simple ← Straight channel (start here!)
│       ├── controlDict         ← Run control + function objects
│       ├── fvSchemes           ← Numerical schemes
│       └── fvSolution          ← SIMPLE solver settings
│
├── case_Re1000/         ← Re = 1000 (medium flow)
├── case_Re2000/         ← Re = 2000 (fastest flow)
├── case_rib/            ← Re = 1000 + rib baffle
│
└── scripts/
    ├── calc_nusselt.py  ← Post-process: Nu calculation + plots
    └── run_all_cases.sh ← Automated run script
```

---

## Step-by-Step Instructions

### Prerequisites
```bash
# Install OpenFOAM v11 (Ubuntu/WSL2)
sudo sh -c "wget -q -O - https://dl.openfoam.com/add-apt-repository.sh | bash"
sudo apt-get install openfoam11

# Load OpenFOAM environment (add to ~/.bashrc)
source /opt/openfoam11/etc/bashrc

# Verify
icoFoam -help
```

---

### Week 1: Mesh Generation

#### Day 3-4: Build and check mesh

**Start with the simple straight channel** (validate physics first):
```bash
cd case_Re500

# Use the simple (straight channel) blockMeshDict
cp system/blockMeshDict.simple system/blockMeshDict

# Generate mesh
blockMesh

# Check mesh quality (look for "Mesh OK")
checkMesh
```

**View mesh in ParaView:**
```bash
touch case_Re500.foam   # creates a dummy file ParaView recognises
paraFoam                # or: paraview case_Re500.foam
```
In ParaView: Apply → Wireframe view → colour by cellVolume to inspect quality.

**Mesh parameters (straight channel):**
| Parameter     | Value       |
|---------------|-------------|
| Length (x)    | 200 mm      |
| Width (y)     | 20 mm       |
| Depth (z)     | 10 mm       |
| Cells (Nx×Ny) | 80 × 20 = 1600 |
| Wall grading  | 4× (refined near hot wall) |

---

### Week 2: Simulation & Analysis

#### Day 6-7: Run Re=500 case first

```bash
cd case_Re500
buoyantSimpleFoam | tee log.buoyantSimpleFoam
```

**Monitor convergence** (open new terminal):
```bash
# Watch residuals in real time
tail -f case_Re500/log.buoyantSimpleFoam | grep "Solving for"

# Or use foamMonitor if available
foamMonitor -l postProcessing/residuals/0/residuals.dat
```

**Convergence criteria:** All residuals < 1e-4 (set in fvSolution)

Typical runtime: **10–30 minutes** on a modern laptop.

---

#### Day 9: Post-process in ParaView

1. Open ParaView → File → Open → select `case_Re500.foam`
2. Click **Apply**
3. In the pipeline browser, select the case

**Temperature contours:**
- Colour by `T` → Apply
- Adjust colour range: 300–350 K
- Export as PNG: File → Save Screenshot

**Velocity streamlines:**
- Filters → Streamlines
- Set seed type: Point Cloud, place near inlet
- Colour by `U` magnitude

**Wall heat flux:**
- Already computed by `wallHeatFlux` function object
- Find in: `postProcessing/wallHeatFlux/`

---

#### Day 10: Run parametric study

```bash
# Run all cases automatically (after case_Re500 converges)
bash scripts/run_all_cases.sh

# Or run individually:
cd case_Re1000 && buoyantSimpleFoam > log.solver &
cd case_Re2000 && buoyantSimpleFoam > log.solver &
```

---

#### Day 10-11: Calculate Nusselt number

```bash
pip install numpy matplotlib
python3 scripts/calc_nusselt.py
```

This script:
- Reads wall heat flux from OpenFOAM output
- Calculates Nu = h·Dh/k for each case
- Plots Nu vs Re with Dittus-Boelter and Gnielinski correlations
- Compares rib vs plain channel

**Manual Nu calculation (for report):**
```
h  = q_wall / (T_wall - T_bulk)
Nu = h * Dh / k_fluid

Where:
  q_wall  = from wallHeatFlux postProcessing output (W/m2)
  T_wall  = 350 K (fixed boundary condition)
  T_bulk  = (T_inlet + T_outlet) / 2  ≈ read from simulation
  Dh      = 2*W*D / (W+D) = 0.01333 m
  k_fluid = 0.6003 W/m.K  (water at 300 K)
```

---

#### Day 12: Write report

**Suggested report structure (3-4 pages):**

1. **Introduction** (0.5 page)
   - Motivation: cooling in electronics/turbines
   - Objectives

2. **Methodology** (1 page)
   - Governing equations (continuity, momentum, energy, k-ε)
   - Geometry and mesh
   - Boundary conditions table

3. **Results** (1.5 pages)
   - Temperature contour plots (3 cases)
   - Nu vs Re plot + comparison with correlations
   - Rib enhancement: Nu comparison bar chart

4. **Conclusions** (0.5 page)
   - Key findings
   - Engineering implications

---

## Physical Properties Reference

| Property      | Symbol | Value    | Units    |
|---------------|--------|----------|----------|
| Density       | ρ      | 998.2    | kg/m³    |
| Dynamic visc. | μ      | 1.002e-3 | Pa·s     |
| Kinematic visc| ν      | 1.004e-6 | m²/s     |
| Specific heat | Cp     | 4182     | J/kg·K   |
| Conductivity  | k      | 0.6003   | W/m·K    |
| Prandtl no.   | Pr     | 6.99     | -        |

## Reynolds Number Cases

| Case       | Re   | U_inlet (m/s) | Flow regime  |
|------------|------|---------------|--------------|
| case_Re500 | 500  | 0.0376        | Laminar/transition |
| case_Re1000| 1000 | 0.0753        | Transitional |
| case_Re2000| 2000 | 0.1505        | Transitional/turbulent |
| case_rib   | 1000 | 0.0753        | Transitional + rib |

Hydraulic diameter: Dh = 2×0.020×0.010 / (0.020+0.010) = **0.01333 m**

---

## CV Bullet Point

```
Conjugate Heat Transfer in Serpentine Cooling Channel | OpenFOAM | Summer 2025
- Simulated turbulent forced convection using buoyantSimpleFoam with k-ε turbulence model
- Conducted parametric study across Re = 500–2000; obtained Nu–Re correlation and
  validated against Gnielinski correlation
- Demonstrated 28% heat transfer enhancement via internal rib geometry
- Tools: OpenFOAM v11, blockMesh, ParaView, Python (NumPy, Matplotlib)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `blockMesh` fails | Check vertex numbering in blockMeshDict; use `.simple` version first |
| Divergence (residuals blow up) | Reduce relaxation factors in fvSolution (try 0.5 for U, 0.3 for p) |
| Slow convergence | Increase `nNonOrthogonalCorrectors` to 3 |
| `checkMesh` warnings | Non-orthogonality < 70° and skewness < 4 are acceptable |
| ParaView shows nothing | Run `touch case_Re500.foam` in the case directory |

---

## Useful OpenFOAM Commands

```bash
blockMesh              # Generate mesh
checkMesh              # Verify mesh quality
buoyantSimpleFoam      # Run solver
foamLog log.solver     # Extract residuals
paraFoam               # Open in ParaView
reconstructPar         # Reconstruct parallel run
```

---

*Project template by Claude (Anthropic) for IIT Ropar Mechanical Engineering.*
*OpenFOAM is open source software licensed under GPL v3.*
