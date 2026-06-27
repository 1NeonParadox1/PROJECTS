# Serpentine Cooling Channel — CFD Project
### OpenFOAM | Heat Transfer | Parametric Study

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
