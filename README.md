# multi_yoked_surface_codes

Codes, circuits, simulation tools and data for **multi-yoked surface codes**: surface-code patches protected by one, two or three levels of outer ("yoke") codes.

The repository contains:

- the outer codes: classical seed codes, quantum chain codes, and the tools that build and check them,
- Stim circuits of the surface-code patches under SI1000 noise,
- a sampler for the complementary gap of a patch, and the calibration that turns gaps into log-likelihood ratios,
- simulators of the yoked memory that draw from the sampled gaps and decode the outer code,
- the data and plotting scripts behind the paper figures.

## Repository layout

```
.
├── circuits/SI1000/        Stim circuits of the surface-code patches
├── ecc/
│   ├── iceberg/            [[n, n-2, 2]] iceberg codes (one-yoke outer codes)
│   ├── seed_codes/         classical seed codes (level 2 and level 3)
│   └── chain_codes/        quantum CSS chain codes (2-yoke and 3-yoke)
├── src/
│   ├── code_maker/         build 3-yoke codes, check distance, reduce column weight
│   ├── gap_sampling/       sample gaps from the circuits and fit the gap -> LLR calibration
│   └── gap_simulator/      one-, two- and three-yoke simulators
├── data/                   working data produced by the scripts in src/
│   ├── sampled_gap/        sampled gaps, one gap_d{d}_p{p}.mmap/ per distance
│   ├── calibration/        gap -> LLR calibration, one JSON per distance
│   ├── simulated/          logical error rates (Sinter CSV), one file per code
│   ├── plots/              figures written by the plot scripts
│   └── plot_*_yoke_all.py  plotting scripts for data/simulated/
├── paper_data/             data and plotting scripts of the paper figures
├── pyproject.toml
└── uv.lock
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

Run every script **from the repository root**, for example `uv run python src/code_maker/l3_code_maker.py`. The scripts in `src/` take most of their settings from constants at the top of each file (or in its `if __name__ == "__main__":` block) and not from the command line.

`tesseract-decoder` is pinned to `0.1.1.dev20260822020007` in `pyproject.toml`: newer releases have no wheel for older glibc versions (2.31).

## Error-correcting codes (`ecc/`)

All code files store binary matrices as rows of `0`/`1` characters, one row per line.

| Directory | Naming | Content |
|---|---|---|
| `iceberg/` | `q{n}_{k}_{d}.txt` | 8 iceberg codes, `n` = 4 to 24 |
| `seed_codes/class_candidates_level_2/` | `c{n}_{k}_{d}.txt` | 6 classical codes, `n` = 2 to 22, `d` = 2 to 3 |
| `seed_codes/class_candidates_level_3/` | `c{n}_{k}_{d}.txt` | 33 classical codes, `n` = 30 to 216, `d` = 5 to 8 |
| `chain_codes/2_yoke/` | `q{n}_{k}_{d}.txt` | 29 quantum codes with `d = 4`, `n` = 24 to 256 |
| `chain_codes/3_yoke/` | `q{n}_{k}_{d}.txt` | quantum codes with `d = 8`, `n` = 96 to 512 |

- A classical file `c{n}_{k}_{d}.txt` holds the parity-check matrix of an `[n, k, d]` code.
- A quantum file `q{n}_{k}_{d}.txt` holds an `[[n, k, d]]` CSS code as two blocks, `Z checks` and `X checks`, each a parity-check matrix with `n` columns.
- In the chain codes the rows are ordered by level: the level-1 checks first (pairwise disjoint supports), then level 2, then (3-yoke only) level 3.
- `*_reduced.txt` is the same code after the column-weight reduction described below.

## Circuits (`circuits/SI1000/`)

`circuit_12dxdxd_d{d}_p0.001.stim` is a memory experiment of a distance-`d` rotated surface code run for `12d` rounds (spacetime volume `12d × d × d`), under the SI1000 noise model at `p = 0.001`. Distances 5 to 11 are included.

## Building the codes (`src/code_maker/`)

| Script | What it does |
|---|---|
| `l3_code_maker.py` | Builds a 3-yoke code from a 2-yoke code and a level-3 classical code, and saves it to `ecc/chain_codes/3_yoke/` once its distance reaches 8 |
| `reduce_column_weight.py` | Lowers the column weight of a 3-yoke code without changing what the decoder sees; writes `<name>_reduced.txt` |
| `_distance_checker.py` | Exact distance of a CSS code (meet-in-the-middle search); `distance(Hz, Hx)` returns `(dz, dx)` |
| `_extend.py` | Reads a code file and doubles a 2-yoke code into two independent copies |
| `_qubit_identifier.py` | Logical Z and X operators of a CSS code (`logical_qubits(Hz, Hx)`) |
| `_code_concatenator.py` | Writes the classical checks on the logical operators of the inner code and stacks them under the doubled inner code |

The construction of `l3_code_maker.py`, for a 2-yoke code with Z checks `H` (`l_1` level-1 rows and `l_2` level-2 rows) and a classical check matrix `C`:

1. Double the 2-yoke code: the level-1 and level-2 rows become block diagonal over two copies.
2. Compute `M = C · L`, where the rows of `L` are the logical operators of the 2-yoke code, and append `[M, M]` as the level-3 rows. The X checks are built the same way from the X logicals.
3. Check that no logical operator has weight below 8. If one does, permute the columns of `C` at random and try again.

Set the inner code, the classical code and `l_1`, `l_2` in the call at the bottom of the file:

```bash
uv run python src/code_maker/l3_code_maker.py
uv run python src/code_maker/reduce_column_weight.py
```

`reduce_column_weight.py` only accepts files straight from `l3_code_maker.py` (it detects the levels from their structure). Its algorithm is explained in [src/code_maker/reduce_column_weight.md](src/code_maker/reduce_column_weight.md).

## Sampling gaps and fitting the calibration (`src/gap_sampling/`)

The simulators do not run the surface-code circuits themselves. They draw from a set of pre-sampled outcomes of one patch: the decoder's complementary gap and whether its prediction was right.

| Step | Command | Output |
|---|---|---|
| 1. Sample gaps for one distance (`D`, `P`, `NUM_SHOTS` at the top of the file) | `uv run python src/gap_sampling/gap_sampler.py` | `data/sampled_gap/gap_d{d}_p{p}.csv` |
| 2. Convert the CSVs to memory-mapped arrays | `uv run python src/gap_sampling/csv2mmap.py` | `data/sampled_gap/gap_d{d}_p{p}.mmap/` |
| 3. Fit the gap → LLR calibration | `uv run python src/gap_sampling/fit_gap.py` | `data/calibration/gap_d{d}_p{p}.json` |

- Step 1 appends to the CSV, so it can be stopped and resumed.
- Step 3 fits every sample set it finds; `--d 7 --p 0.001` fits one. It writes an affine map (`scale`, `offset`), plus the knots of a piecewise-linear map (`knots_gap`, `knots_llr`, `tail_scale`) when that map wins a 2-fold holdout. The procedure is explained in [src/gap_sampling/fit_gap.md](src/gap_sampling/fit_gap.md).
- Without a calibration file the simulators fall back to `llr = 0.95 × gap` and print a warning.

## Simulating the yoked memory (`src/gap_simulator/`)

| Script | Outer code (`MATRIX_PATH` in the file) | Extra options |
|---|---|---|
| `sim_one_yoke.py` | an iceberg code, e.g. `ecc/iceberg/q8_6_2.txt` | – |
| `sim_two_yoke.py` | a 2-yoke code; set `n_l1` to its number of level-1 rows | `--l1-check-rate` |
| `sim_three_yoke.py` | a 3-yoke code; set `n_l1` and `n_l2` | `--l1-check-rate`, `--l2-check-rate` |

```bash
uv run python src/gap_simulator/sim_one_yoke.py --d 7 --t-interval 8 --shots 100000 --trivial-first
uv run python src/gap_simulator/sim_two_yoke.py --d 7 --t-interval 48 --l1-check-rate 2 --shots 100000 --trivial-first
```

- Only the Z checks of the code are simulated, with noiseless check measurements. The observable is one logical Z operator of the outer code.
- Each run appends to a single CSV per code, `data/simulated/<protocol>.csv` (for example `one_yoke_q8_6_2.csv`). Every parameter is stored in the row metadata, so runs with different `d`, `t_interval` or check rates share the file.
- `--max-outer-errors N` stops a run once the CSV holds `N` outer errors for that parameter set; errors from earlier runs count.
- `--processes`, `--chunk-size` and `--csv` set the worker count, the shots per task and the output path. Run a script with `--help` for the full list.
- `run_one_yoke.sh` and `run_two_yoke.sh` loop over distances and intervals (or check rates) with a target number of outer errors per point; edit the settings block at the top.

The outer decoder ([tesseract](https://github.com/quantumlib/tesseract-decoder)) is compiled afresh for every decode (`REBUILD_EVERY_SHOT` in `util/_outer_decoder.py`). Reusing one compiled decoder and only rewriting its costs keeps a stale search heuristic with the standard tesseract build and gives too many outer errors.

### CSV metadata

Each run appears twice, as an `_inner` and an `_outer` entry (the `level` field). The `json_metadata` column holds:

| Key | Meaning |
|---|---|
| `protocol` | simulator and code, e.g. `two_yoke_q48_30_4` |
| `d`, `p` | surface-code distance and physical error rate |
| `t_interval` | interval between outer rounds, in units of `12d` rounds |
| `l1_check_rate`, `l2_check_rate` | how many times the level-1 (level-2) checks are measured per outer round |
| `outer_rounds` | number of outer rounds |
| `num_ticks` | number of inner rounds, used to normalize the error rate |
| `calibration` | `piecewise` or `affine`: which gap → LLR map decoded the run |

### Plotting

```bash
uv run python data/plot_one_yoke_all.py
uv run python data/plot_two_yoke_all.py
```

They read `data/simulated/one_yoke_*.csv` and `data/simulated/two_yoke_*.csv` and write to `data/plots/`. By default the error rate is divided by `num_ticks` to give a rate per inner round. The defaults are constants at the top of each script, and `--help` lists the command-line options (filters such as `--d-min`, the x axis `--x` of the two-yoke script, output formats).

## Paper data (`paper_data/`)

The data behind the paper figures, with their own plotting scripts. These files are independent of `data/` and `src/`.

| Directory | Content |
|---|---|
| `calibration/` | gap → LLR calibration for distances 5 to 11 |
| `sampled_gap/` | histograms of the sampled gaps (`gap_d{d}_p0.001_dist.csv`, columns `gap_lo,gap_hi,count`) and `plot_gap.py` |
| `LERs/` | logical error rates (Sinter CSV) and plotting scripts |

| LER file | Protocols |
|---|---|
| `one_yoke_combined.csv` | `[[8,6,2]]`, `[[16,14,2]]` |
| `two_yoke_combined.csv` | `[[8,6,2]]->[[36,30,2]]`, `[[16,14,2]]->[[112,104,2]]` |
| `three_yoke_combined.csv` | `[[16,14,2]]->[[112,104,2]]->[[208,166,2]]` |

```bash
uv run python paper_data/sampled_gap/plot_gap.py
uv run python paper_data/LERs/plot_one_yoke_all.py
uv run python paper_data/LERs/plot_two_yoke_all.py --x d --t 12
uv run python paper_data/LERs/plot_three_yoke_all.py --x d
```

The figures are written to the `plots/` folder next to each script. The LER scripts take `--x {d,t,r,r2,p,tr}` for the x axis (two- and three-yoke), the filters `--d`, `--d-min`, `--d-max`, `--p`, `--t`, `--r`, `--r2`, and `--formats png pdf`; `--help` gives the full list.
