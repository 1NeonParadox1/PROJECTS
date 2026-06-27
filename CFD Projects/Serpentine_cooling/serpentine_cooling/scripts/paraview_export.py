#!/usr/bin/env python3
"""
paraview_export.py
==================
Automates ParaView screenshot export for the CFD report.
Generates temperature contours and velocity streamlines.

Run AFTER simulation is complete:
  pvpython scripts/paraview_export.py

pvpython is bundled with ParaView installation.
"""

try:
    from paraview.simple import *
except ImportError:
    print("This script must be run with pvpython (ParaView's Python):")
    print("  pvpython scripts/paraview_export.py")
    print("")
    print("Alternatively, use ParaView GUI and manually:")
    print("  1. File > Open > case_Re500.foam")
    print("  2. Apply > Colour by T")
    print("  3. File > Save Screenshot")
    import sys; sys.exit(0)

import os

CASES = {
    'Re500':  '../case_Re500',
    'Re1000': '../case_Re1000',
    'Re2000': '../case_Re2000',
    'Rib':    '../case_rib',
}

def export_temperature_contour(case_name, case_path, output_dir='../report/figures'):
    """Export temperature contour screenshot."""
    os.makedirs(output_dir, exist_ok=True)

    foam_file = os.path.join(case_path, f'{os.path.basename(case_path)}.foam')
    if not os.path.exists(foam_file):
        # Create trigger file
        open(foam_file, 'w').close()

    # Load case
    reader = OpenFOAMReader(FileName=foam_file)
    reader.MeshRegions = ['internalMesh']
    reader.CellArrays = ['T', 'U', 'p']
    UpdatePipeline()

    # Go to last timestep
    animScene = GetAnimationScene()
    animScene.GoToLast()

    # Temperature display
    display = Show(reader, GetActiveViewOrCreate('RenderView'))
    display.ColorArrayName = ['CELLS', 'T']

    # Set colour range 300-350 K
    lut = GetColorTransferFunction('T')
    lut.RescaleTransferFunction(300.0, 350.0)

    # Blue-to-red temperature scale
    lut.ApplyPreset('Cool to Warm', True)

    # Screenshot
    view = GetActiveView()
    view.ResetCamera()
    SaveScreenshot(
        os.path.join(output_dir, f'temperature_{case_name}.png'),
        view,
        ImageResolution=[1200, 400],
        FontScaling='Scale fonts proportionally'
    )
    print(f"  Saved: temperature_{case_name}.png")

    Delete(reader)

print("Exporting ParaView figures...")
for name, path in CASES.items():
    print(f"\nProcessing {name}...")
    try:
        export_temperature_contour(name, path)
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone. Figures saved in report/figures/")
print("Insert into your report with:")
print("  \\includegraphics[width=\\textwidth]{figures/temperature_Re500.png}")
