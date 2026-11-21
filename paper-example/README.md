# Pressure errors in 44 Ca–Fe–Ni structures

These are the 44 fixed DFT geometries from the 50 GPa pressure benchmark. Each row contains the two predictions for the pressure–volume term per atom and the DFT reference at the same geometry. At fixed volume per atom, an error in this term is proportional to the hydrostatic pressure error.

From the repository root, run:

~~~sh
python paper-example/rank_pressure.py
~~~

The script selects the 11 structures with the largest absolute difference between the DPA-3 and local DeepMD predictions, then consults the DFT column. These 11 contain 5 of the 11 largest errors for DPA-3 and 8 of the 11 largest errors for local DeepMD. Random selection of 11 out of 44 has an expected overlap of 2.75 with either set. The mean absolute errors for the selected cells are 22.15 and 134.27 meV/atom. Across all 44 they are 12.26 and 58.54 meV/atom.

This retrospective check evaluates a ranking on a fixed panel. The DPA-3 export is the benchmark model. The overlap of the local model's training data with these cells is unknown. The selection rule has no access to DFT when it ranks candidates. DFT defines the errors used to assess that ranking. The test does not measure a gain from retraining or a change in phase discovery rate.
