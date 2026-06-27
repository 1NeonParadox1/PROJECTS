# Post-processing scripts

Run these **from inside the case folder** (e.g. `laminarChannel/`) after
`simpleFoam` has converged.

```bash
# 1. Friction factor vs theory
python3 ../postProcessing_scripts/friction_factor.py --regime laminar
python3 ../postProcessing_scripts/friction_factor.py --regime turbulent

# 2. Velocity profile plot (saved as velocity_profile.png)
python3 ../postProcessing_scripts/velocity_profile.py --regime laminar
python3 ../postProcessing_scripts/velocity_profile.py --regime turbulent
```

Requires: `numpy`, `matplotlib` (`pip install numpy matplotlib` if missing).
