"""Rank the 44 benchmark cells using the two predicted pressure-volume terms.

Run from any directory with standard Python 3:
    python rank_pressure.py
Pass another table of the same format as the first argument if required.
Energies and pressure-volume terms in the input are in eV/atom.
"""
import csv
import sys
from math import isfinite
from pathlib import Path
from statistics import mean

def read_table(path):
    with path.open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    ids = [int(row["frame"]) for row in table]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Frame identifiers must be unique in a nonempty table.")
    predicted = {}
    reference = {}
    for frame, row in zip(ids, table):
        first = float(row["q_dpa3_eV_atom"])
        second = float(row["q_local_eV_atom"])
        q_dft = float(row["q_DFT_eV_atom"])
        if not all(map(isfinite, (first, second, q_dft))):
            raise ValueError(f"Nonfinite pressure-volume term in frame {frame}")
        predicted[frame] = (first, second)
        reference[frame] = q_dft
    return predicted, reference

def largest(values, count):
    return sorted(values, key=lambda frame: (-values[frame], frame))[:count]

def main():
    default = Path(__file__).with_name("benchmark_pair_50GPa.csv")
    predictions, reference = read_table(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
    count = len(predictions)
    if count % 4:
        raise ValueError("This upper-quarter comparison requires a multiple of four cells.")
    budget = count // 4
    gaps = {frame: abs(a - b) for frame, (a, b) in predictions.items()}
    selected = largest(gaps, budget)
    print(f"Selected {budget}/{count} in descending model disagreement: {selected}")
    print(f"Expected overlap under uniform selection: {budget**2/count:.2f}")
    for column, model in enumerate(("DPA-3", "Local DeepMD")):
        error = {i: abs(pair[column] - reference[i]) * 1000 for i, pair in predictions.items()}
        worst = set(largest(error, budget))
        overlap = len(worst.intersection(selected))
        selected_mae = mean(error[i] for i in selected)
        print(f"{model}: overlap {overlap}/{budget}; MAE in selected/all cells "
              f"{selected_mae:.2f}/{mean(error.values()):.2f} meV/atom")

if __name__ == "__main__":
    main()

