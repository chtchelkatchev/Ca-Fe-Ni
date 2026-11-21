"""Two-potential screening diagnostics from the Ca–Fe–Ni Supplement (Sec. S6)."""
import math


LABELS = ("D_e", "D_h", "D_F", "D_q")
DEFAULT_THRESHOLDS = (0.015, 0.015, 0.010, 0.020)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def force_rows(prediction):
    forces = prediction["forces_eV_A"]
    if not forces or any(len(row) != 3 for row in forces):
        raise ValueError("Forces must have shape (N, 3), with N > 0")
    return [tuple(finite(value) for value in row) for row in forces]


def model_scalars(prediction, volume, natoms):
    stress = prediction["stress_eV_A3"]
    if len(stress) != 6:
        raise ValueError("Supply six ASE Voigt stress components: xx, yy, zz, yz, xz, xy")
    stress = tuple(finite(value) for value in stress)
    pressure = -sum(stress[:3]) / 3
    q = pressure * volume / natoms
    e = finite(prediction["energy_eV"]) / natoms
    return e, e + q, q


def diagnostics(volume_A3, first, second):
    """Return D_e, D_h, D_F, D_q for two models on one cell and atom ordering."""
    volume = finite(volume_A3)
    if volume <= 0:
        raise ValueError("Cell volume must be positive")
    fa, fb = force_rows(first), force_rows(second)
    n = len(fa)
    if len(fb) != n:
        raise ValueError("The two models must supply the same number of forces")
    a = model_scalars(first, volume, n)
    b = model_scalars(second, volume, n)
    de, dh, dq = ((u - v) ** 2 / 4 for u, v in zip(a, b))
    squared_force_difference = sum(
        (u - v) ** 2 for row_a, row_b in zip(fa, fb) for u, v in zip(row_a, row_b)
    )
    return de, dh, squared_force_difference / (36 * n), dq


def flags(scores, thresholds=DEFAULT_THRESHOLDS):
    """The historical driver used strict greater-than and selected the union."""
    return tuple(value > cutoff for value, cutoff in zip(scores, thresholds, strict=True))
