# CFD Channel Flow Analysis using OpenFOAM

A Computational Fluid Dynamics (CFD) project that simulates **pressure-driven laminar and turbulent channel flow** using **OpenFOAM**. The project validates numerical results against classical fluid mechanics correlations by comparing pressure drop, velocity profiles, and Darcy friction factor.

## Project Objectives

- Simulate incompressible channel flow in OpenFOAM
- Analyze both laminar and turbulent flow regimes
- Compare CFD results with analytical and empirical correlations
- Compute Darcy friction factor from pressure drop
- Generate velocity profile plots for validation

---

## Features

- Laminar Channel Flow Simulation
- Turbulent Channel Flow Simulation (k-ω SST Model)
- Structured Hexahedral Mesh using `blockMesh`
- Automated Post-processing Scripts
- Friction Factor Calculation
- Velocity Profile Visualization
- Validation Against Theoretical Results

---

## Project Structure

```
CFD_ChannelFlow_Project/
│
├── laminarChannel/
│   ├── 0/
│   ├── constant/
│   └── system/
│
├── turbulentChannel/
│   ├── 0/
│   ├── constant/
│   └── system/
│
├── postProcessing_scripts/
│   ├── friction_factor.py
│   ├── velocity_profile.py
│   └── README_scripts.md
│
└── README.md
```

---

## Physics

### Laminar Flow
- Reynolds Number ≈ **500**
- Analytical validation using

\[
f = \frac{96}{Re}
\]

### Turbulent Flow
- Reynolds Number ≈ **20,000**
- k-ω SST turbulence model
- Validation using the Blasius correlation

\[
f = 0.316 Re^{-0.25}
\]

---

## Software Used

- OpenFOAM
- ParaView
- Python
- NumPy
- Matplotlib

---

## Running the Simulation

### Generate Mesh

```bash
blockMesh
```

### Check Mesh

```bash
checkMesh
```

### Run Solver

```bash
simpleFoam
```

### Visualize Results

```bash
paraFoam
```

---

## Post Processing

Calculate friction factor:

```bash
python3 ../postProcessing_scripts/friction_factor.py --regime laminar
```

Generate velocity profile:

```bash
python3 ../postProcessing_scripts/velocity_profile.py --regime laminar
```

Replace `laminar` with `turbulent` for the turbulent case.

---

## Results

The project compares CFD predictions with theoretical correlations through:

- Pressure Drop
- Darcy Friction Factor
- Velocity Distribution
- Fully Developed Flow Behavior

Expected outputs include:

- Friction factor comparison
- Velocity profile plots
- Pressure gradient validation

---

## Learning Outcomes

This project demonstrates:

- CFD simulation workflow in OpenFOAM
- Mesh generation and validation
- Boundary condition implementation
- Turbulence modeling using k-ω SST
- CFD result validation with fluid mechanics theory
- Python-based post-processing

---

## Future Improvements

- Mesh independence study
- Reynolds number sweep
- Circular pipe flow simulation
- Comparison with experimental data
- Additional turbulence models (k-ε, LES)

---

## Author

**Ayush Raj**

B.Tech Mechanical Engineering  
Indian Institute of Technology (IIT) Ropar

---

## License

This project is intended for educational and learning purposes.