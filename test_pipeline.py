import contextlib
import io
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import demo
from pipeline import jsonlines, read_records, screen, update
from screening import DEFAULT_THRESHOLDS, diagnostics, flags


class PipelineTests(unittest.TestCase):
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
