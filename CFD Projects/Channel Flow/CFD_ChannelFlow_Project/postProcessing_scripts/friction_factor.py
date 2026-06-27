"""
friction_factor.py
-------------------
Computes the Darcy friction factor from an OpenFOAM channel-flow run and
compares it against the analytical / empirical correlation.

USAGE (run from inside the case folder, e.g. laminarChannel/, AFTER running
simpleFoam and after the case has converged):

    python3 ../postProcessing_scripts/friction_factor.py --regime laminar
    python3 ../postProcessing_scripts/friction_factor.py --regime turbulent

It reads:
    postProcessing/centrelineSample/<latestTime>/centreline_p.xy

which has columns:  x   p

and fits a straight line to p(x) in the FULLY DEVELOPED region
(we use the back half of the domain, x in [1.5, 3.0] m, to avoid the
entrance-length region) to get dp/dx, then computes:

    f = (-dp/dx) * Dh / (0.5 * rho * Umean^2)        [Darcy friction factor]

and compares it to:
    laminar:    f_theory = 96 / Re_Dh   (exact, parallel-plate Poiseuille flow)
    turbulent:  f_theory = 0.316 * Re_Dh^-0.25   (Blasius correlation, Re < 1e5)
"""

import argparse
import glob
import os
import numpy as np


def find_latest_centreline_file():
    candidates = glob.glob("postProcessing/centrelineSample/*/centreline_p.xy")
    if not candidates:
        raise FileNotFoundError(
            "Could not find postProcessing/centrelineSample/*/centreline_p.xy\n"
            "Make sure you have run simpleFoam in this case folder first."
        )
    # pick the folder with the largest numeric timestep
    def time_of(path):
        try:
            return float(path.split(os.sep)[-2])
        except ValueError:
            return -1
    candidates.sort(key=time_of)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["laminar", "turbulent"], required=True)
    parser.add_argument("--rho", type=float, default=1.2, help="density [kg/m3], default air")
    parser.add_argument("--nu", type=float, default=1.5e-5, help="kinematic viscosity [m2/s]")
    parser.add_argument("--Dh", type=float, default=0.04, help="hydraulic diameter [m] = 2*H")
    parser.add_argument("--U", type=float, default=None,
                         help="mean inlet velocity [m/s]. If omitted, uses the "
                              "default for the chosen regime (0.1875 laminar, 7.5 turbulent).")
    parser.add_argument("--fit_xmin", type=float, default=1.5,
                         help="start of the fully-developed region used for the linear fit [m]")
    parser.add_argument("--fit_xmax", type=float, default=3.0,
                         help="end of the fully-developed region used for the linear fit [m]")
    args = parser.parse_args()

    U = args.U
    if U is None:
        U = 0.1875 if args.regime == "laminar" else 7.5

    path = find_latest_centreline_file()
    print(f"Reading: {path}")
    data = np.loadtxt(path)
    x, p = data[:, 0], data[:, 1]

    mask = (x >= args.fit_xmin) & (x <= args.fit_xmax)
    if mask.sum() < 5:
        raise RuntimeError("Too few points in the fit region - widen --fit_xmin/--fit_xmax")

    # linear fit: p = m*x + c  ->  dp/dx = m
    m, c = np.polyfit(x[mask], p[mask], 1)
    dpdx = m

    Re = U * args.Dh / args.nu
    f_cfd = (-dpdx) * args.Dh / (0.5 * args.rho * U**2)

    if args.regime == "laminar":
        f_theory = 96.0 / Re
        theory_label = "f = 96/Re_Dh  (exact parallel-plate Poiseuille flow)"
    else:
        f_theory = 0.316 * Re ** -0.25
        theory_label = "f = 0.316*Re_Dh^-0.25  (Blasius correlation, valid Re<~1e5)"

    error_pct = 100.0 * (f_cfd - f_theory) / f_theory

    print("\n--- Friction factor results --------------------------------")
    print(f"Regime              : {args.regime}")
    print(f"Mean velocity U     : {U:.4f} m/s")
    print(f"Hydraulic diameter  : {args.Dh:.4f} m")
    print(f"Re_Dh               : {Re:.1f}")
    print(f"dp/dx (CFD, fit)    : {dpdx:.6f} Pa/m")
    print(f"Darcy f (CFD)       : {f_cfd:.5f}")
    print(f"Darcy f (theory)    : {f_theory:.5f}   [{theory_label}]")
    print(f"Difference          : {error_pct:+.2f} %")
    print("---------------------------------------------------------------")
    print("\nA difference of a few percent is normal and expected (numerical")
    print("discretisation, mesh resolution, finite domain/entrance length).")
    print("If the difference is large (>15-20%), check: mesh convergence,")
    print("whether the flow is fully developed by your fit_xmin, and (for the")
    print("turbulent case) the yPlus values from postProcess -func yPlus.")


if __name__ == "__main__":
    main()
