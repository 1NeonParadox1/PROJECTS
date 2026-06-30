"""
Generates placeholder F1 rear-wing geometry as 4 separate watertight STL
solids, sized to match the OpenFOAM case already built:
    mainplane.stl    fixed lower element (inverted NACA airfoil, extruded)
    flap.stl         moving upper flap, in REST (downforce) position
    endplate.stl     two end plates (simple angled boxes)
    rotatingZone.stl closed cylinder enclosing flap's full rotation sweep

NOT a replica of Ferrari's real SF-26 wing - proprietary geometry isn't
public. This is parametric placeholder geometry, F1-proportioned, meant
to validate the full meshing/AMI/rotation pipeline. Swap in real STLs
later using the exact same filenames and the rest of the case (snappyHexMeshDict,
dynamicMeshDict, 0/ fields, etc.) needs zero changes.

Units: meters throughout.
"""
import numpy as np
import struct

# ---------------------------------------------------------------------
# Basic STL writer (binary), takes a list of triangles, each a (3,3) array
# ---------------------------------------------------------------------
def write_stl_binary(filename, triangles):
    with open(filename, 'wb') as f:
        header = b'Generated placeholder F1 wing geometry'.ljust(80, b' ')
        f.write(header)
        f.write(struct.pack('<I', len(triangles)))
        for tri in triangles:
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 0.0])
            f.write(struct.pack('<3f', *normal))
            for v in (v0, v1, v2):
                f.write(struct.pack('<3f', *v))
            f.write(struct.pack('<H', 0))


# ---------------------------------------------------------------------
# NACA 4-digit airfoil section generator
# ---------------------------------------------------------------------
def naca4_coords(code='6412', n=60, chord=1.0):
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:4]) / 100.0

    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2.0  # cosine spacing, dense near LE/TE

    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                  + 0.2843*x**3 - 0.1015*x**4)

    yc = np.where(
        x < p,
        m / (p**2 + 1e-12) * (2*p*x - x**2),
        m / ((1-p)**2 + 1e-12) * ((1 - 2*p) + 2*p*x - x**2)
    )
    dyc_dx = np.where(
        x < p,
        2*m / (p**2 + 1e-12) * (p - x),
        2*m / ((1-p)**2 + 1e-12) * (p - x)
    )
    theta = np.arctan(dyc_dx)

    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)

    # closed loop: upper surface LE->TE, then lower surface TE->LE
    xs = np.concatenate([xu, xl[::-1][1:]])
    ys = np.concatenate([yu, yl[::-1][1:]])

    return xs * chord, ys * chord


# ---------------------------------------------------------------------
# Extrude a closed 2D airfoil section into a solid along the span (y axis)
# section coords are in (x, z); result placed at given y range and offset
# ---------------------------------------------------------------------
def extrude_airfoil(xs, zs, y0, y1, x_offset=0.0, z_offset=0.0,
                     chord_scale=1.0, invert=False, rotate_deg=0.0,
                     pivot=(0.0, 0.0)):
    xs = xs * chord_scale
    zs = zs * chord_scale
    if invert:
        zs = -zs  # flip camber for downforce-generating (upside-down) section

    if rotate_deg != 0.0:
        theta = np.radians(rotate_deg)
        px, pz = pivot
        xr = xs - px
        zr = zs - pz
        xs = xr*np.cos(theta) - zr*np.sin(theta) + px
        zs = xr*np.sin(theta) + zr*np.cos(theta) + pz

    xs = xs + x_offset
    zs = zs + z_offset

    n = len(xs)
    triangles = []

    # side faces (quad strip between y0 and y1) split into 2 triangles each
    for i in range(n - 1):
        p0 = np.array([xs[i],   y0, zs[i]])
        p1 = np.array([xs[i+1], y0, zs[i+1]])
        p2 = np.array([xs[i+1], y1, zs[i+1]])
        p3 = np.array([xs[i],   y1, zs[i]])
        triangles.append((p0, p1, p2))
        triangles.append((p0, p2, p3))
    # close the loop (last point back to first)
    p0 = np.array([xs[-1], y0, zs[-1]])
    p1 = np.array([xs[0],  y0, zs[0]])
    p2 = np.array([xs[0],  y1, zs[0]])
    p3 = np.array([xs[-1], y1, zs[-1]])
    triangles.append((p0, p1, p2))
    triangles.append((p0, p2, p3))

    # end caps: fan triangulation from centroid (airfoils are star-shaped
    # from their centroid, so this produces a valid closed cap)
    cx, cz = np.mean(xs), np.mean(zs)
    for y_cap, flip in [(y0, True), (y1, False)]:
        centroid = np.array([cx, y_cap, cz])
        for i in range(n - 1):
            a = np.array([xs[i],   y_cap, zs[i]])
            b = np.array([xs[i+1], y_cap, zs[i+1]])
            if flip:
                triangles.append((centroid, b, a))
            else:
                triangles.append((centroid, a, b))
        a = np.array([xs[-1], y_cap, zs[-1]])
        b = np.array([xs[0],  y_cap, zs[0]])
        if flip:
            triangles.append((centroid, b, a))
        else:
            triangles.append((centroid, a, b))

    return triangles


# ---------------------------------------------------------------------
# Simple box generator for endplates
# ---------------------------------------------------------------------
def box_triangles(xmin, xmax, ymin, ymax, zmin, zmax):
    v = {
        '000': np.array([xmin, ymin, zmin]), '100': np.array([xmax, ymin, zmin]),
        '110': np.array([xmax, ymax, zmin]), '010': np.array([xmin, ymax, zmin]),
        '001': np.array([xmin, ymin, zmax]), '101': np.array([xmax, ymin, zmax]),
        '111': np.array([xmax, ymax, zmax]), '011': np.array([xmin, ymax, zmax]),
    }
    faces = [
        ('000','100','110','010'),  # bottom
        ('001','011','111','101'),  # top
        ('000','010','011','001'),  # -x
        ('100','101','111','110'),  # +x
        ('000','001','101','100'),  # -y
        ('010','110','111','011'),  # +y
    ]
    triangles = []
    for f in faces:
        a, b, c, d = [v[k] for k in f]
        triangles.append((a, b, c))
        triangles.append((a, c, d))
    return triangles


# ---------------------------------------------------------------------
# Capped cylinder generator for the AMI rotating zone
# ---------------------------------------------------------------------
def cylinder_triangles(origin, axis_len, radius, axis='y', n=48):
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    cx, cy, cz = origin
    triangles = []

    def pt(a, h):
        # axis is y: circle in x-z plane, extruded along y
        return np.array([cx + radius*np.cos(a), cy + h, cz + radius*np.sin(a)])

    h0, h1 = 0.0, axis_len
    for i in range(n):
        a0, a1 = theta[i], theta[(i+1) % n]
        p0, p1 = pt(a0, h0), pt(a1, h0)
        p2, p3 = pt(a1, h1), pt(a0, h1)
        triangles.append((p0, p1, p2))
        triangles.append((p0, p2, p3))

    centre0 = np.array([cx, cy + h0, cz])
    centre1 = np.array([cx, cy + h1, cz])
    for i in range(n):
        a0, a1 = theta[i], theta[(i+1) % n]
        p0, p1 = pt(a0, h0), pt(a1, h0)
        triangles.append((centre0, p1, p0))
        p2, p3 = pt(a1, h1), pt(a0, h1)
        triangles.append((centre1, p3, p2))

    return triangles


# =======================================================================
# BUILD GEOMETRY - F1-proportioned placeholder rear wing
# =======================================================================
SPAN = 0.90          # m, regs-compliant max width ballpark
MAIN_CHORD = 0.25    # m
FLAP_CHORD = 0.12    # m
WING_HEIGHT = 0.85   # m, z-height of wing reference (above ground)
SLOT_GAP = 0.025     # m, gap between mainplane TE and flap LE region

y0, y1 = -SPAN/2, SPAN/2

# --- Mainplane: inverted cambered NACA section (downforce orientation) ---
xs, zs = naca4_coords('6412', n=50, chord=1.0)
mainplane_tris = extrude_airfoil(
    xs, zs, y0, y1,
    x_offset=0.0, z_offset=WING_HEIGHT,
    chord_scale=MAIN_CHORD, invert=True
)
write_stl_binary('/home/claude/geom_gen/mainplane.stl', mainplane_tris)

# --- Flap: smaller cambered section, positioned above/behind mainplane TE ---
# Pivot placed at flap mid-chord, offset up/back from mainplane TE per the
# real mechanism's centre-pivot design (not an end-hinge)
flap_x_offset = MAIN_CHORD + SLOT_GAP - 0.02   # slight overlap typical of multi-element wings
flap_z_offset = WING_HEIGHT + 0.04

xs_f, zs_f = naca4_coords('6408', n=50, chord=1.0)
flap_tris = extrude_airfoil(
    xs_f, zs_f, y0, y1,
    x_offset=flap_x_offset, z_offset=flap_z_offset,
    chord_scale=FLAP_CHORD, invert=True
)
write_stl_binary('/home/claude/geom_gen/flap.stl', flap_tris)

# Flap pivot location in global coords (mid-chord of the flap, used in
# dynamicMeshDict origin) - computed here for consistency, printed below
flap_pivot = (flap_x_offset + 0.5*FLAP_CHORD, 0.0, flap_z_offset)

# --- Endplates: simple angled boxes at each span end ---
endplate_thickness = 0.008
endplate_height_below = 0.30
endplate_height_above = 0.20
endplate_x_extent_front = -0.05
endplate_x_extent_back = MAIN_CHORD + FLAP_CHORD + 0.10

ep_tris = []
for y_centre in (y0, y1):
    ep_tris += box_triangles(
        endplate_x_extent_front, endplate_x_extent_back,
        y_centre - endplate_thickness/2, y_centre + endplate_thickness/2,
        WING_HEIGHT - endplate_height_below, WING_HEIGHT + endplate_height_above
    )
write_stl_binary('/home/claude/geom_gen/endplate.stl', ep_tris)

# --- Rotating zone: cylinder enclosing flap through its full sweep ---
# Computed precisely (not guessed): the true required radius is the flap's
# max RADIAL distance from the pivot AXIS (rotation is about y, so only
# x-z distance matters, not full 3D distance which wrongly includes span).
# Measured: flap needs ~0.060m, mainplane's nearest point is ~0.076m away
# from the pivot axis - only ~16mm of real clearance margin. A radius that
# looks "generous" in 3D (e.g. 0.13m) actually CLIPS the mainplane here.
flap_tris_arr = np.array(flap_tris).reshape(-1, 3)
flap_radial = np.linalg.norm(flap_tris_arr[:, [0, 2]] - np.array([flap_pivot[0], flap_pivot[2]]), axis=1)
rot_radius = flap_radial.max() + 0.008   # ~8mm safety margin over the flap's true sweep radius
rot_origin = (flap_pivot[0], y0 - 0.02, flap_pivot[2])
rot_axis_len = SPAN + 0.04

rot_tris = cylinder_triangles(rot_origin, rot_axis_len, rot_radius, n=48)
write_stl_binary('/home/claude/geom_gen/rotatingZone.stl', rot_tris)

print("Flap pivot (set this as 'origin' in constant/dynamicMeshDict):")
print(f"  ({flap_pivot[0]:.4f} {flap_pivot[1]:.4f} {flap_pivot[2]:.4f})")
print(f"Rotating zone cylinder: origin {rot_origin}, radius {rot_radius:.4f}, length {rot_axis_len:.4f}")

# Sanity check: confirm the cylinder does NOT reach the mainplane
mainplane_tris_arr = np.array(mainplane_tris).reshape(-1, 3)
main_radial = np.linalg.norm(mainplane_tris_arr[:, [0, 2]] - np.array([flap_pivot[0], flap_pivot[2]]), axis=1)
clearance = main_radial.min() - rot_radius
print(f"Clearance between cylinder and mainplane: {clearance*1000:.1f} mm "
      f"({'OK' if clearance > 0 else 'CLIPPING - FIX REQUIRED'})")

print("Done.")
