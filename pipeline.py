"""Screen paired fixed-geometry predictions and explicitly incorporate returned labels."""
import argparse
import filecmp
import json
import math
import re
import tempfile
from pathlib import Path

from screening import (DEFAULT_THRESHOLDS, LABELS, component_rms, finite, flags, force_differences,
                       force_rows, model_scalars, scores_and_force_differences)


def volume(cell):
    (a, b, c), (d, e, f), (g, h, i) = cell
    return abs(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def matrix(values, rows):
    if len(values) != rows or any(len(row) != 3 for row in values):
        raise ValueError(f"expected a {rows} by 3 array")
    return [[finite(x) for x in row] for row in values]


def geometry(row):
    item_id = row["id"]
    if not isinstance(item_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", item_id):
        raise ValueError("id must contain only letters, digits, '.', '_' or '-'")
    atoms = row["geometry"]
    symbols = atoms["symbols"]
    if not isinstance(symbols, list) or not symbols or any(
        not isinstance(s, str) or not re.fullmatch(r"[A-Z][a-z]?", s) for s in symbols
    ):
        raise ValueError(f"{item_id}: invalid ordered list of chemical symbols")
    cell = matrix(atoms["cell_A"], 3)
    if finite(volume(cell)) <= 0:
        raise ValueError(f"{item_id}: singular cell")
    return {"id": item_id, "geometry": {"symbols": symbols, "cell_A": cell,
            "positions_A": matrix(atoms["positions_A"], len(symbols))}}


def read_records(path, kind):
    rows = {}
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                row = geometry(raw)
                item_id = row["id"]
                if item_id in rows:
                    raise ValueError(f"duplicate id {item_id}")
                if kind == "prediction" and "reference_method" in raw:
                    raise ValueError("a prediction file contains a reference label")
                if kind != "geometry":
                    n = len(row["geometry"]["symbols"])
                    forces = force_rows(raw)
                    if len(forces) != n:
                        raise ValueError(f"{item_id}: force count differs from atom count")
                    model_scalars(raw, volume(row["geometry"]["cell_A"]), n)
                    row.update(energy_eV=finite(raw["energy_eV"]),
                               forces_eV_A=[list(v) for v in forces],
                               stress_eV_A3=[finite(x) for x in raw["stress_eV_A3"]])
                if kind == "reference":
                    method = raw["reference_method"]
                    if not isinstance(method, str) or not method.strip():
                        raise ValueError("reference_method must name the labeling method")
                    row["reference_method"] = method
                rows[item_id] = row
            except (KeyError, TypeError, ValueError, OverflowError) as err:
                raise ValueError(f"{path}:{line_no}: {err}") from err
    return rows


def same_geometry(a, b):
    if a["id"] != b["id"] or a["geometry"] != b["geometry"]:
        raise ValueError(f"{a['id']}: cell, species or ordered coordinates differ")


def jsonlines(rows):
    for row in rows:
        yield json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"


def xyz(rows, labeled=False):
    for row in rows:
        atoms = row["geometry"]
        lattice = " ".join(str(x) for v in atoms["cell_A"] for x in v)
        props = "species:S:1:pos:R:3" + (":forces:R:3" if labeled else "")
        header = f'Lattice="{lattice}" Properties={props} pbc="T T T" frame_id={row["id"]}'
        if labeled:
            xx, yy, zz, yz, xz, xy = row["stress_eV_A3"]
            tensor = " ".join(map(str, (xx, xy, xz, xy, yy, yz, xz, yz, zz)))
            header += f' energy={row["energy_eV"]} stress="{tensor}"'
        yield f"{len(atoms['symbols'])}\n{header}\n"
        for i, (symbol, pos) in enumerate(zip(atoms["symbols"], atoms["positions_A"], strict=True)):
            values = [*pos, *row["forces_eV_A"][i]] if labeled else pos
            yield symbol + " " + " ".join(map(str, values)) + "\n"


def publish(directory, files):
    """Commit a complete result directory. Replaying identical outputs is safe."""
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        staged = Path(tmp, "result")
        staged.mkdir()
        for name, lines in files.items():
            with (staged / name).open("w", encoding="utf-8", newline="\n") as stream:
                stream.writelines(lines)
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
            if not all((target / f).is_file() and filecmp.cmp(staged / f, target / f, shallow=False)
                       for f in files):
                raise ValueError(f"{target} already exists with different contents; use a new directory")


def atom_score_rows(first, second, rankings, cutoff):
    for record in rankings:
        item_id = record["id"]
        differences = force_differences(first[item_id], second[item_id])
        symbols = first[item_id]["geometry"]["symbols"]
        for index, (symbol, delta) in enumerate(zip(symbols, differences, strict=True), start=1):
            rms = component_rms(delta)
            yield {"id": item_id, "atom_index": index, "element": symbol,
                   "delta_force_eV_A": delta, "rms_difference_eV_A": rms,
                   "above_atom_cutoff": None if cutoff is None else rms > cutoff}


def screen(first_file, second_file, directory, thresholds=DEFAULT_THRESHOLDS, atom_force_cutoff=None):
    thresholds = tuple(finite(t) for t in thresholds)
    if len(thresholds) != len(LABELS) or min(thresholds) <= 0:
        raise ValueError("supply four positive finite thresholds in order D_e, D_h, D_F, D_q")
    if atom_force_cutoff is not None:
        atom_force_cutoff = finite(atom_force_cutoff)
        if atom_force_cutoff <= 0:
            raise ValueError("atom force cutoff must be positive in eV/Å")
    first = read_records(first_file, "prediction")
    second = read_records(second_file, "prediction")
    if first.keys() != second.keys():
        raise ValueError("the two prediction files must cover identical candidate IDs")
    rankings = []
    for item_id, a in first.items():
        b = second[item_id]
        same_geometry(a, b)
        values, differences = scores_and_force_differences(volume(a["geometry"]["cell_A"]), a, b)
        hits = flags(values, thresholds)
        reasons = [k for k, hit in zip(LABELS, hits) if hit]
        rms = [component_rms(delta) for delta in differences]
        peak = max(range(len(rms)), key=rms.__getitem__)
        rank_score = max(v / t for v, t in zip(values, thresholds))
        flagged_atoms = []
        if atom_force_cutoff is not None:
            flagged_atoms = [i for i, value in enumerate(rms, start=1) if value > atom_force_cutoff]
            if flagged_atoms:
                reasons.append("atom_force")
            rank_score = max(rank_score, rms[peak] / atom_force_cutoff)
        rankings.append({"id": item_id, **dict(zip(LABELS, values)),
                         "cell_force_rms_eV_A": math.sqrt(12 * values[2]),
                         "max_atom_index": peak + 1, "max_atom_element": a["geometry"]["symbols"][peak],
                         "max_atom_rms_eV_A": rms[peak], "atom_force_cutoff_eV_A": atom_force_cutoff,
                         "triggered_atom_indices": flagged_atoms, "triggered": reasons,
                         "rank_score": rank_score})
    rankings.sort(key=lambda x: (-x["rank_score"], x["id"]))
    selected = [geometry(first[r["id"]]) for r in rankings if r["triggered"]]
    atoms = atom_score_rows(first, second, rankings, atom_force_cutoff)
    publish(directory, {"scores.jsonl": jsonlines(rankings), "atom_scores.jsonl": jsonlines(atoms),
                        "to_label.jsonl": jsonlines(selected),
                        "to_label.xyz": xyz(selected)})
    return selected


def audit_forces(first_file, second_file, labels_file, directory):
    """Evaluate each model's site error only after labels are available."""
    first = read_records(first_file, "prediction")
    second = read_records(second_file, "prediction")
    labels = read_records(labels_file, "reference")

    def records():
        for item_id, ref in labels.items():
            if item_id not in first or item_id not in second:
                raise ValueError(f"{item_id}: labeled structure missing from one or both predictions")
            errors = []
            for model in (first[item_id], second[item_id]):
                same_geometry(ref, model)
                errors.append([component_rms(v) for v in force_differences(model, ref)])
            for index, (symbol, ea, eb) in enumerate(zip(ref["geometry"]["symbols"], *errors,
                                                        strict=True), start=1):
                larger = "tie" if ea == eb else ("A" if ea > eb else "B")
                yield {"id": item_id, "atom_index": index, "element": symbol,
                       "reference_method": ref["reference_method"], "rms_error_A_eV_A": ea,
                       "rms_error_B_eV_A": eb, "larger_error_model": larger}

    publish(directory, {"atom_errors.jsonl": jsonlines(records())})


def update(request_file, labels_file, base_file, directory):
    requested = read_records(request_file, "geometry")
    labels = read_records(labels_file, "reference")
    base = read_records(base_file, "reference")
    missing = sorted(requested.keys() - labels.keys())
    unexpected = sorted(labels.keys() - requested.keys())
    if missing or unexpected:
        raise ValueError(f"requested/returned IDs differ: missing={missing}, unexpected={unexpected}")
    for item_id, label in labels.items():
        same_geometry(requested[item_id], label)
        if item_id in base and base[item_id] != label:
            raise ValueError(f"{item_id}: conflicting reference already present in training data")
        base[item_id] = label
    rows = [base[key] for key in sorted(base)]
    publish(directory, {"training.jsonl": jsonlines(rows), "training.xyz": xyz(rows, labeled=True)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paired = commands.add_parser("screen", help="score and export geometries before labeling")
    paired.add_argument("first_model")
    paired.add_argument("second_model")
    paired.add_argument("--thresholds", nargs=4, type=float, default=DEFAULT_THRESHOLDS,
                        metavar=("D_e", "D_h", "D_F", "D_q"), help="four positive variance thresholds")
    paired.add_argument("--atom-force-cutoff", type=float, metavar="EV_PER_ANGSTROM",
                        help="also select a structure when a site's force-component RMS exceeds this value")
    audit = commands.add_parser("audit-forces", help="compare stored model forces with returned references by atom")
    audit.add_argument("first_model")
    audit.add_argument("second_model")
    audit.add_argument("labels")
    ingest = commands.add_parser("update", help="merge returned reference labels with the base set")
    ingest.add_argument("request")
    ingest.add_argument("labels")
    ingest.add_argument("base")
    for command in (paired, ingest, audit):
        command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "screen":
            chosen = screen(args.first_model, args.second_model, args.out, args.thresholds, args.atom_force_cutoff)
            print(f"Selected {len(chosen)} configurations.")
        elif args.command == "audit-forces":
            audit_forces(args.first_model, args.second_model, args.labels, args.out)
            print(f"Site errors written to {args.out / 'atom_errors.jsonl'}")
        else:
            print(f"Shared training set: {len(update(args.request, args.labels, args.base, args.out))} records.")
    except (OSError, ValueError) as err:
        parser.exit(1, f"{err}\n")


if __name__ == "__main__":
    main()
