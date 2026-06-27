"""
velocity_profile.py
---------------------
Plots the CFD velocity profile across the channel half-gap (sampled at
x = 2.5 m, in the fully developed region) against the analytical /
empirical profile, and saves a PNG.

USAGE (run from inside the case folder, AFTER simpleFoam has converged):

    python3 ../postProcessing_scripts/velocity_profile.py --regime laminar
    python3 ../postProcessing_scripts/velocity_profile.py --regime turbulent
"""

import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_latest_profile_file():
    candidates = glob.glob("postProcessing/profileSample/*/profile_x2p5_U.xy")
    if not candidates:
        raise FileNotFoundError(
            "Could not find postProcessing/profileSample/*/profile_x2p5_U.xy\n"
            "Make sure you have run simpleFoam in this case folder first."
        )
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
    parser.add_argument("--h", type=float, default=0.01, help="half-gap height [m]")
    parser.add_argument("--nu", type=float, default=1.5e-5)
    parser.add_argument("--U", type=float, default=None)
    parser.add_argument("--out", default="velocity_profile.png")
    args = parser.parse_args()

    U = args.U
    if U is None:
        U = 0.1875 if args.regime == "laminar" else 7.5

    path = find_latest_profile_file()
    print(f"Reading: {path}")
    # .xy columns for a vector field: y  Ux  Uy  Uz
    data = np.loadtxt(path)
    y, Ux = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(Ux, y, "o", ms=3, label="CFD (OpenFOAM)", color="#1f77b4")

    if args.regime == "laminar":
        # Analytical parabola for parallel-plate Poiseuille flow over the
        # FULL gap [-h, h], mapped onto our half-domain y in [0, h]:
        # u(y) = u_max * (1 - (y/h)^2),  u_max = 1.5 * U_mean
        u_max = 1.5 * U
        y_th = np.linspace(0, args.h, 100)
        u_th = u_max * (1 - (y_th / args.h) ** 2)
        ax.plot(u_th, y_th, "-", lw=2, color="#d62728",
                 label="Analytical parabola (Poiseuille)")
    else:
        # Simple log-law sketch for reference (not a rigorous fit):
        # u+ = (1/kappa)*ln(y+) + B,  kappa=0.41, B=5.0
        kappa, B = 0.41, 5.0
        # rough utau estimate from Blasius friction factor
        Dh = 2 * 2 * args.h
        Re = U * Dh / args.nu
        f = 0.316 * Re ** -0.25
        tau_w = f / 8 * 1.2 * U ** 2
        u_tau = (tau_w / 1.2) ** 0.5
        y_th = np.linspace(args.h * 1e-4, args.h, 200)
        yplus = y_th * u_tau / args.nu
        uplus = (1 / kappa) * np.log(yplus) + B
        u_th = uplus * u_tau
        ax.plot(u_th, y_th, "-", lw=2, color="#d62728",
                 label="Log-law sketch (reference only)")

    ax.set_xlabel("Velocity U_x [m/s]")
    ax.set_ylabel("Distance from centreline, y [m]")
    ax.set_title(f"Velocity profile at x = 2.5 m  ({args.regime})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
