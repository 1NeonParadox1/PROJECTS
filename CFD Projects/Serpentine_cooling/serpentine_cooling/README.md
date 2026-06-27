# Serpentine Cooling Channel CFD Analysis using OpenFOAM

A Computational Fluid Dynamics (CFD) project that investigates **heat transfer enhancement in a serpentine cooling channel** using **OpenFOAM**. The project performs a parametric study over multiple Reynolds numbers and evaluates the effect of adding a rib/baffle on cooling performance through conjugate heat transfer simulations.

## Project Objectives

- Simulate coolant flow through a serpentine cooling channel
- Analyze heat transfer under different Reynolds numbers
- Evaluate the influence of rib/baffle geometry on thermal performance
- Calculate Nusselt number for each operating condition
- Visualize temperature and velocity distributions

---

## Features

- Conjugate Heat Transfer Simulation
- Multiple Reynolds Number Cases (500, 1000, 2000)
- Rib/Baffle Heat Transfer Enhancement Study
- Structured Mesh Generation using `blockMesh`
- Automated Post-processing Scripts
- Nusselt Number Calculation
- Temperature and Velocity Contour Visualization

---

## Project Structure

```text
serpentine_cooling/
│
├── case_Re500/
├── case_Re1000/
├── case_Re2000/
├── case_rib/
│
├── scripts/
│   ├── calc_nusselt.py
│   ├── paraview_export.py
│   └── run_all_cases.sh
│
├── report/
│   └── report.tex
│
└── README.md
```

---

## Physics

### Flow Conditions

- Coolant: Water
- Inlet Temperature: **300 K**
- Heated Wall Temperature: **350 K**
- Reynolds Numbers:
  - Re = 500
  - Re = 1000
  - Re = 2000

### Simulation

- Incompressible turbulent flow
- Heat transfer using `buoyantSimpleFoam`
- Turbulence model (k-ε)
- Comparison between smooth and ribbed channel configurations

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
buoyantSimpleFoam
```

### Visualize Results

```bash
paraFoam
```

---

## Post Processing

Calculate the Nusselt number:

```bash
python3 scripts/calc_nusselt.py
```

Export visualization data:

```bash
python3 scripts/paraview_export.py
```

Run all simulation cases automatically:

```bash
bash scripts/run_all_cases.sh
```

---

## Results

The project evaluates cooling performance using:

- Temperature Distribution
- Velocity Profiles
- Pressure Drop
- Heat Transfer Coefficient
- Average Nusselt Number
- Comparison between Smooth and Ribbed Channels

Expected outputs include:

- Temperature contour plots
- Velocity contour plots
- Nusselt Number vs Reynolds Number graph
- Heat transfer enhancement due to rib geometry

---

## Applications

This simulation is representative of cooling systems used in:

- CPU & GPU Heat Sinks
- Gas Turbine Blade Cooling
- Battery Thermal Management Systems
- Compact Heat Exchangers
- Electronic Device Cooling

---

## Learning Outcomes

This project demonstrates:

- CFD simulation using OpenFOAM
- Heat transfer analysis
- Mesh generation and validation
- Turbulence modeling
- Parametric CFD studies
- Python-based post-processing
- Thermal performance evaluation using the Nusselt number

---

## Future Improvements

- Mesh independence study
- Optimization of rib geometry
- Additional turbulence models (k-ω SST, LES)
- Transient heat transfer simulations
- Experimental validation
- Higher Reynolds number investigations

---

## Author

**Ayush Raj**

B.Tech Mechanical Engineering  
Indian Institute of Technology (IIT) Ropar

---

## License

This project is intended for educational and learning purposes.