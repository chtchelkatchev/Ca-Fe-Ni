import contextlib
import io
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import demo
from pipeline import audit_forces, jsonlines, read_records, screen, update
from screening import DEFAULT_THRESHOLDS, diagnostics, flags


class PipelineTests(unittest.TestCase):
    def test_force_balanced_local_outlier_in_64_and_returned_reference(self):
        n = 64
        symbols = ["Ca"] * n
        symbols[36] = "Fe"
        atom = {"id": "illustration", "geometry": {"symbols": symbols,
                "cell_A": [[12, 0, 0], [0, 12, 0], [0, 0, 12]],
                "positions_A": [[i % 4, (i // 4) % 4, i // 16] for i in range(n)]},
                "energy_eV": 0.0, "stress_eV_A3": [0] * 6, "forces_eV_A": [[0, 0, 0] for _ in range(n)]}
        first_record = deepcopy(atom)
        # One large discrepancy, balanced by small, equal, opposite discrepancies.
        for force in first_record["forces_eV_A"]:
            force[0] = -0.60 / (n - 1)
        first_record["forces_eV_A"][36] = [0.60, 0, 0]
        self.assertAlmostEqual(math.fsum(f[0] for f in first_record["forces_eV_A"]), 0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / "a", root / "b"
            a.write_text(json.dumps(first_record), encoding="utf8")
            b.write_text(json.dumps(atom), encoding="utf8")
            self.assertEqual(screen(a, b, root / "historical"), [])
            summary = json.loads((root / "historical/scores.jsonl").read_text())
            self.assertEqual((summary["max_atom_index"], summary["max_atom_element"]), (37, "Fe"))
            self.assertAlmostEqual(summary["max_atom_rms_eV_A"], 0.3464101615)
            self.assertAlmostEqual(summary["cell_force_rms_eV_A"], 0.04364357805)
            self.assertAlmostEqual(summary["D_F"], 0.00015873015873)
            self.assertAlmostEqual(summary["cell_force_rms_eV_A"] ** 2 / 12, summary["D_F"])
            sites = [json.loads(line) for line in (root / "historical/atom_scores.jsonl").read_text().splitlines()]
            self.assertEqual(len(sites), n)
            self.assertTrue(all(s["above_atom_cutoff"] is None for s in sites))
            self.assertEqual(sites[36]["delta_force_eV_A"], [0.6, 0, 0])
            self.assertAlmostEqual(sites[0]["rms_difference_eV_A"], 0.00549857399)
            self.assertAlmostEqual(sum(s["rms_difference_eV_A"]**2 for s in sites) / n,
                                   summary["cell_force_rms_eV_A"] ** 2)
            chosen = screen(a, b, root / "local", atom_force_cutoff=0.25)
            self.assertEqual([r["id"] for r in chosen], ["illustration"])
            local = json.loads((root / "local/scores.jsonl").read_text())
            self.assertEqual((local["triggered"], local["triggered_atom_indices"]), (["atom_force"], [37]))
            local_sites = (json.loads(l) for l in (root / "local/atom_scores.jsonl").read_text().splitlines())
            self.assertEqual([s["atom_index"] for s in local_sites if s["above_atom_cutoff"]], [37])
            self.assertEqual(screen(a, b, root / "equal", atom_force_cutoff=local["max_atom_rms_eV_A"]), [])
            for invalid in (0, -1, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    screen(a, b, root / "invalid", atom_force_cutoff=invalid)

            ref = deepcopy(atom)
            ref["reference_method"] = "analytic example"
            for force in ref["forces_eV_A"]:
                force[0] = -0.1 / (n - 1)
            ref["forces_eV_A"][36] = [0.1, 0, 0]
            labels = root / "labels"
            labels.write_text(json.dumps(ref), encoding="utf8")
            audit_forces(a, b, labels, root / "audit")
            errors = [json.loads(line) for line in (root / "audit/atom_errors.jsonl").read_text().splitlines()]
            self.assertEqual(errors[36]["larger_error_model"], "A")
            self.assertAlmostEqual(errors[36]["rms_error_A_eV_A"], 0.5 / math.sqrt(3))
            self.assertAlmostEqual(errors[36]["rms_error_B_eV_A"], 0.1 / math.sqrt(3))
            audit_forces(b, b, labels, root / "tie")
            ties = (root / "tie/atom_errors.jsonl").read_text().splitlines()
            self.assertTrue(all(json.loads(line)["larger_error_model"] == "tie" for line in ties))
            with self.assertRaisesRegex(ValueError, "reference label"):
                screen(labels, b, root / "leak", atom_force_cutoff=0.25)
            ref["geometry"]["positions_A"].reverse()
            labels.write_text(json.dumps(ref), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "ordered coordinates differ"):
                audit_forces(a, b, labels, root / "incorrect")
            self.assertFalse((root / "incorrect").exists())

    def test_physical_scores_and_strict_thresholds(self):
        a = dict(energy_eV=4, forces_eV_A=[[1, 2, 0], [0, 0, 0]], stress_eV_A3=[-1, -2, 0, 8, 0, 0])
        b = dict(energy_eV=0, forces_eV_A=[[0, 0, 0], [0, 0, 0]], stress_eV_A3=[0] * 6)
        # N=2, V=12; e=2, pressure=1, q=6, h=8. Shear does not enter q.
        self.assertEqual(diagnostics(12, a, b), (1, 16, 5 / 72, 9))
        self.assertFalse(any(flags(DEFAULT_THRESHOLDS)))

    def test_two_rounds_and_input_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                demo.run(out)
                demo.run(out)  # the identical round can be replayed safely
            first = out / "round-1/to_label.jsonl"
            returned = out / "returned/labels.jsonl"
            base = out / "seed/training.jsonl"
            new = out / "updated/training.jsonl"
            targets = read_records(first, "geometry")
            self.assertEqual((len(targets), len(read_records(new, "reference"))), (7, 10))
            self.assertEqual(len(read_records(out / "round-2/to_label.jsonl", "geometry")), 1)
            a, b = (out / "predictions-1" / name for name in ("inverse-power.jsonl", "table.jsonl"))
            self.assertEqual(screen(a, b, out / "strict", [1e8] * 4), [])
            for thresholds in ([0] * 4, [math.inf] * 4, [math.nan] * 4):
                with self.assertRaises(ValueError):
                    screen(a, b, out / "bad-cutoffs", thresholds)
            labels = read_records(returned, "reference")
            bad_file = out / "incorrect.jsonl"
            altered = deepcopy(labels)
            changed = next(iter(altered.values()))
            changed["geometry"]["positions_A"].reverse()
            bad_file.write_text("".join(jsonlines(altered.values())), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordered coordinates differ"):
                update(first, bad_file, base, out / "bad")
            bad_file.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing="):
                update(first, bad_file, base, out / "bad")
            with self.assertRaisesRegex(ValueError, "reference label"):
                screen(returned, a, out / "bad")
            row = json.loads(a.read_text().splitlines()[0])
            row["geometry"]["positions_A"].reverse()
            lines = a.read_text().splitlines()
            bad_file.write_text("\n".join([json.dumps(row), *lines[1:]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "ordered coordinates differ"):
                screen(bad_file, b, out / "bad")
            self.assertFalse((out / "bad").exists())


if __name__ == "__main__":
    unittest.main()
