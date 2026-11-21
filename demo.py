"""Run two active-learning rounds on synthetic dimers using only the Python standard library."""
import argparse
from bisect import bisect_right
from math import exp
from pathlib import Path
from statistics import mean

from pipeline import jsonlines, publish, screen, update, volume


def dimer(r):
    return {"id": f"pair-{round(100 * r):03}", "geometry": {
        "symbols": ["Ar", "Ar"], "cell_A": [[9, 0, 0], [0, 9, 0], [0, 0, 9]],
        "positions_A": [[2, 4, 4], [2 + r, 4, 4]]}}


def distance(row):
    a, b = row["geometry"]["positions_A"]
    return b[0] - a[0]  # This teaching set has one isolated dimer oriented along +x.


def evaluate(row, potential):
    r = distance(row)
    energy, slope = potential(r)
    stress_xx = r * slope / volume(row["geometry"]["cell_A"])  # tensile-positive
    return {**row, "energy_eV": energy, "forces_eV_A": [[slope, 0, 0], [-slope, 0, 0]],
            "stress_eV_A3": [stress_xx, 0, 0, 0, 0, 0]}


def reference(r):
    # Illustrative Morse curve in eV and angstroms with arbitrary parameters.
    depth, alpha, minimum = 1.3, 1.6, 2.4
    u = exp(-alpha * (r - minimum))
    return depth * (u * u - 2 * u), 2 * depth * alpha * u * (1 - u)


def label(row):
    return {**evaluate(row, reference), "reference_method": "analytic Morse demo (no DFT)"}


def fit_models(training):
    data = sorted((distance(row), row["energy_eV"], row["forces_eV_A"][0][0]) for row in training)
    r, energy, slope = zip(*data)
    q = [x**-2 for x in r]
    slope_q = [s / (-2 * x**-3) for x, s in zip(r, slope)]
    q0, s0, e0 = mean(q), mean(slope_q), mean(energy)
    variance = mean((v - q0) ** 2 for v in q)
    k = mean((v - q0) * (s - s0) for v, s in zip(q, slope_q)) / variance

    def inverse_power(x):
        # Quadratic E(q) in q = 1/r^2. Fit dE/dq and the mean energy separately.
        dq = x**-2 - q0
        return e0 + s0 * dq + 0.5 * k * (dq * dq - variance), (s0 + k * dq) * (-2 * x**-3)

    def tabulated(x):
        # Piecewise cubic Hermite potential with a tangent continuation outside the table.
        if x <= r[0] or x >= r[-1]:
            i = 0 if x <= r[0] else -1
            return energy[i] + slope[i] * (x - r[i]), slope[i]
        i = bisect_right(r, x) - 1
        dx, width = x - r[i], r[i + 1] - r[i]
        secant = (energy[i + 1] - energy[i]) / width
        a = (3 * secant - 2 * slope[i] - slope[i + 1]) / width
        b = (slope[i] + slope[i + 1] - 2 * secant) / width**2
        return (energy[i] + dx * (slope[i] + dx * (a + dx * b)),
                slope[i] + dx * (2 * a + 3 * dx * b))

    return inverse_power, tabulated


def predictions(directory, pool, models):
    names = ("inverse-power.jsonl", "table.jsonl")
    data = {name: jsonlines([evaluate(row, model) for row in pool])
            for name, model in zip(names, models, strict=True)}
    publish(directory, data)
    return tuple(directory / name for name in names)


def run(out):
    # Initial reference data exist before round 1. The candidate pool is unlabeled.
    seed = [label(dimer(r)) for r in (2.2, 2.4, 2.6)]
    pool = [dimer(r) for r in (1.45, 1.60, 1.75, 1.90, 2.05, 2.30, 2.50, 2.70, 2.85, 3.00, 3.15)]
    publish(out / "seed", {"training.jsonl": jsonlines(seed)})
    initial_models = fit_models(seed)
    a, b = predictions(out / "predictions-1", pool, initial_models)
    chosen = screen(a, b, out / "round-1")

    # Only now call the teaching reference on exactly the identifiers selected above.
    publish(out / "returned", {"labels.jsonl": jsonlines(label(row) for row in chosen)})
    training = update(out / "round-1/to_label.jsonl", out / "returned/labels.jsonl",
                      out / "seed/training.jsonl", out / "updated")
    new_models = fit_models(training)  # Both fits receive the same ordered reference records.

    known = {row["id"] for row in training}
    next_pool = [row for row in pool if row["id"] not in known] + [dimer(1.85), dimer(2.95)]
    a, b = predictions(out / "predictions-2", next_pool, new_models)
    next_chosen = screen(a, b, out / "round-2")
    print("Teaching reference: analytic Morse dimer (no DFT).")
    print(f"Round 1: selected {len(chosen)}/{len(pool)} before obtaining new labels.")
    print(f"Both models refitted on {len(seed)} initial + {len(chosen)} returned configurations.")
    print(f"Round 2: selected {len(next_chosen)}/{len(next_pool)}; geometries ready for labeling.")
    print(f"Files: {out}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("demo-output"))
    run(cli.parse_args().out)
