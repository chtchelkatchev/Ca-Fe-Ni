# Active learning with two interatomic potentials

Choose periodic structures for reference calculations by comparing the energy, forces and stress predicted by two different potentials on the same atomic geometry. Train both models on the same returned reference data. The screening rule comes from Sec. S6 of the Ca–Fe–Ni Supplement.

## Run the example

From the root of a fresh clone, with Python 3.10 or later:

```sh
python demo.py
```

The demo uses only the standard library. It fits a global inverse-power potential and a local cubic interpolation to three synthetic dimer configurations. Both learn from energies and forces and predict a consistent tensile-positive stress. They rank an unlabeled pool, request labels, refit on the combined data and screen the next pool. The reference is an analytic Morse curve with illustrative parameters. The demo does not run DFT or reproduce the Ca–Fe–Ni phase diagram.

The first round writes scores for all 11 candidates and selects 7. Both models then fit the same 10 records. The second round selects 1 of 6 candidates. Inspect `demo-output/round-1/to_label.xyz` for the selected structures and `demo-output/updated/training.xyz` for the common labeled set. Reruns reproduce the same files.

To try the separate retrospective check on the 44 published pressure-benchmark records, run `python paper-example/rank_pressure.py`.

## Use two existing calculators

Produce one JSON Lines prediction file per model. Each line carries a unique string ID, ordered species, Cartesian positions and three lattice vectors in Å, total energy in eV, forces in eV/Å and ASE tensile-positive stress in eV/Å³ (Voigt order xx, yy, zz, yz, xz, xy). IDs may contain letters, digits, dots, hyphens and underscores. Supply fully periodic cells and use the same elemental energy reference for both models. The screen matches IDs and checks the cell, species and coordinates exactly, including atom order. Write the geometry actually evaluated by each calculator.

With ASE and your calculators already installed, this adapter builds one prediction record:

```python
def predict(frame_id, atoms, calculator):
    a = atoms.copy()
    a.calc = calculator
    assert a.pbc.all(), "The screen expects a three-dimensionally periodic cell"
    geometry = dict(symbols=a.get_chemical_symbols(), cell_A=a.cell.array.tolist(),
                    positions_A=a.get_positions().tolist())
    return dict(id=frame_id, geometry=geometry, energy_eV=a.get_potential_energy(apply_constraint=False),
                forces_eV_A=a.get_forces(apply_constraint=False).tolist(),
                stress_eV_A3=a.get_stress(voigt=True, apply_constraint=False).tolist())
```

Call it on the same unrelaxed coordinates with each calculator and write one `json.dumps(record)` per line. Calculator-specific fitting and inference stay in their usual programs.

```sh
python pipeline.py screen predictions_a.jsonl predictions_b.jsonl --out round-01
```

For each model let `e = E/N`, `q = -V tr(stress)/(3N)` and `h = e + q`. For `x = e, h, q`, the score is `D_x = (x_a - x_b)²/4`. The force score is `D_F = Σ(F_a - F_b)²/(36N)`, summed over all atoms and Cartesian components. A structure is selected if **any** of these strict bounds is exceeded:

| Score | D_e | D_h | D_F | D_q |
|---|---:|---:|---:|---:|
| Threshold | 0.015 | 0.015 | 0.010 | 0.020 |

Scalar thresholds are in (eV/atom)² and the force threshold is in (eV/Å)². These defaults reproduce the Ca–Fe–Ni screen. For another application, pass four positive values via `--thresholds D_e D_h D_F D_q`. Use the model-predicted stress to evaluate `q`. `scores.jsonl` records the four diagnostics, triggered criteria and maximum score divided by its own threshold. This ratio orders candidates for inspection. The threshold union determines selection.

## Locate force disagreement by atom

Every `screen` writes `atom_scores.jsonl` for all candidates before reference labeling. Each record gives the structure ID, 1-based atom index, element, signed Cartesian force difference `ΔF = F_A − F_B` and its component RMS `d_i = √[(ΔFx²+ΔFy²+ΔFz²)/3]` in eV/Å. `scores.jsonl` reports the atom with the largest `d_i` (first in input order for ties), its element, the maximum and the cell RMS `√[Σ_i d_i²/N]`. The historical cell variance `D_F` is the square of that cell RMS divided by 12.

Optionally select a geometry when any atom exceeds a local threshold:

```sh
python pipeline.py screen predictions_a.jsonl predictions_b.jsonl --atom-force-cutoff 0.25 --out round-local
```

This threshold is in eV/Å. The four structure criteria still apply. A local exceedance adds `atom_force` to `triggered` and records **all** matching atom indices. When the option is set, ordering also accounts for the maximum local RMS divided by the local threshold. The original Ca–Fe–Ni selection used the four cell scores. No atomic cutoff is enabled by default.

Consider a 64-atom illustration with zero total force difference. At one site the difference is `(0.60, 0, 0)` eV/Å. At each of the other 63 sites it is `(−0.60/63, 0, 0)` eV/Å. The component RMS is **0.346410 eV/Å** at the first site and **0.005499 eV/Å** at each of the others. The cell RMS is **0.043644 eV/Å**, and `D_F = 0.00015873 (eV/Å)²`, well below the historical cutoff of `0.01`. With identical scalar predictions, the optional `0.25 eV/Å` threshold selects the structure and identifies exactly one atom. `test_pipeline.py` reproduces this constructed example.

After reference forces have been obtained, compare each prediction with the reference on the same sites:

```sh
python pipeline.py audit-forces predictions_a.jsonl predictions_b.jsonl returned.jsonl --out audit-01
```

`atom_errors.jsonl` gives both per-site force errors (component RMS) and the model with the larger error, with `tie` for equal values. Model A is the first prediction file. The audit examines labeled structures only and checks their cells, coordinates and atom order against both predictions. The example run from `demo.py` writes this comparison against its analytic reference in `demo-output/force-audit/`.

## Return reference labels and start the next round

`to_label.jsonl` and `to_label.xyz` contain only the selected IDs and geometries. Run your reference code on these **fixed** cells and positions. For each completed calculation use the same JSONL fields as for a prediction and add a `reference_method` string identifying the calculation. Retain the original atom order. The existing training set uses the same schema and may be empty. Use consistent reference settings across rounds, including functional, pseudopotentials, energy zero and stress convention.

```sh
python pipeline.py update round-01/to_label.jsonl returned.jsonl base.jsonl --out trained-01
```

The update checks IDs and geometries, rejects missing or unexpected labels and detects conflicts with existing records. If only part of the selected set is to be labeled, explicitly provide that subset as the request. Both trainers receive the same `trained-01/training.jsonl` or `training.xyz`. The extended XYZ contains total energy, forces and the full symmetric stress tensor. Import it with each trainer's usual reader, refit and run `screen` again on new candidates. Model agreement can conceal a shared error, so reference calculations still determine final phase rankings.

## Associated manuscript

N. M. Chtchelkatchev, M. V. Magnitskaya and R. E. Ryltsev, *Disagreement between two interatomic potentials guides active learning in evolutionary crystal structure searches*. An [earlier version of the manuscript](https://arxiv.org/abs/2606.30870) describes the Ca–Fe–Ni search.
